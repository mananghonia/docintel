"""Evaluation: field-level metrics, GroupKFold comparison harness.

Two levels of truth:

- token level: macro-F1 over BIO tags (never accuracy — ~85% of tokens are O,
  so accuracy is a lie; trap #3).
- field level: for each document, does the extracted field VALUE match the
  ground truth after normalisation? This is the number the business sees.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field as dc_field

import numpy as np

from ml.labeling import Document, FIELDS, extract_field_values
from ml.postprocess import parse_amount, parse_date


# ---------------------------------------------------------------------------
# Value normalisation for comparison
# ---------------------------------------------------------------------------

def _norm_value(field_name: str, value) -> str:
    s = str(value).strip()
    if field_name in ("subtotal", "tax_amount", "total_amount"):
        a = parse_amount(s)
        return f"{a:.2f}" if a is not None else s.lower()
    if field_name.endswith("_date"):
        d = parse_date(s)
        return d.isoformat() if d is not None else s.lower()
    return " ".join(s.lower().replace(" ", " ").split())


# ---------------------------------------------------------------------------
# Field-level scoring
# ---------------------------------------------------------------------------

@dataclass
class FieldScores:
    tp: dict = dc_field(default_factory=lambda: defaultdict(int))
    fp: dict = dc_field(default_factory=lambda: defaultdict(int))
    fn: dict = dc_field(default_factory=lambda: defaultdict(int))

    def add_document(self, truth: dict[str, str], predicted: dict[str, str]) -> None:
        """Exact-match field scoring. A wrong value counts as BOTH a false
        positive (a wrong value was emitted) and a false negative (the true
        value was not produced) — there is no partial credit for a field.

        This is intentionally strict: it penalises precision and recall for
        the same error, so these F1 numbers are LOWER than token-level or
        partial-overlap span-F1 and should not be compared to those directly.
        The threshold that matters operationally is "did the reviewer have to
        touch this field", and a wrong value fails that just as a miss does.
        """
        for f in FIELDS:
            has_t, has_p = f in truth and str(truth[f]).strip() != "", f in predicted
            if has_p and has_t:
                if _norm_value(f, predicted[f]) == _norm_value(f, truth[f]):
                    self.tp[f] += 1
                else:
                    self.fp[f] += 1  # extracted, but wrong value
                    self.fn[f] += 1
            elif has_p:
                self.fp[f] += 1
            elif has_t:
                self.fn[f] += 1

    def per_field(self) -> dict[str, dict[str, float]]:
        out = {}
        for f in FIELDS:
            tp, fp, fn = self.tp[f], self.fp[f], self.fn[f]
            if tp + fp + fn == 0:
                continue
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            out[f] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}
        return out

    def macro_f1(self) -> float:
        per = self.per_field()
        return float(np.mean([m["f1"] for m in per.values()])) if per else 0.0


def evaluate_field_extraction(docs: list[Document],
                              predicted_tags: list[list[str]]) -> FieldScores:
    """Score predicted tag sequences against each doc's meta['truth'] (or
    values reconstructed from its gold annotations)."""
    scores = FieldScores()
    for doc, tags in zip(docs, predicted_tags):
        truth = doc.meta.get("truth")
        if truth is None:
            truth = {a.field: a.value for a in doc.annotations}
        # Re-tag copies of the tokens with predictions, extract values.
        import copy
        toks = copy.deepcopy(doc.tokens)
        for t, tag in zip(toks, tags):
            t.tag = tag
        predicted = extract_field_values(toks)
        scores.add_document(truth, predicted)
    return scores


# ---------------------------------------------------------------------------
# Token-level scoring
# ---------------------------------------------------------------------------

def per_document_field_f1(docs: list[Document],
                          predicted_tags: list[list[str]]) -> list[float]:
    """One field macro-F1 per document — the per-unit statistic a paired
    bootstrap needs to decide whether one model really beats another."""
    return [evaluate_field_extraction([d], [t]).macro_f1()
            for d, t in zip(docs, predicted_tags)]


def bootstrap_win_rate(challenger_scores: list[float], champion_scores: list[float],
                       n_boot: int = 1000, random_state: int = 0) -> float:
    """Paired bootstrap over documents: fraction of resamples in which the
    challenger's mean per-doc F1 exceeds the champion's. ~1.0 = confident
    win, ~0.5 = indistinguishable. Guards against promoting on holdout noise
    when the holdout is small."""
    diffs = np.asarray(challenger_scores) - np.asarray(champion_scores)
    if len(diffs) == 0:
        return 0.0
    rng = np.random.RandomState(random_state)
    means = [rng.choice(diffs, size=len(diffs), replace=True).mean()
             for _ in range(n_boot)]
    return float((np.asarray(means) > 0).mean())


def token_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the tags that actually appear in y_true, excluding O."""
    from sklearn.metrics import f1_score

    from ml.labeling import TAG2ID

    labels = sorted(set(int(t) for t in np.unique(y_true)) - {TAG2ID["O"]})
    if not labels:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=labels, average="macro",
                          zero_division=0))


def token_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    from sklearn.metrics import classification_report

    from ml.labeling import ID2TAG

    labels = sorted(set(int(t) for t in np.unique(np.concatenate([y_true, y_pred]))))
    return classification_report(
        y_true, y_pred, labels=labels,
        target_names=[ID2TAG[i] for i in labels], zero_division=0)


# ---------------------------------------------------------------------------
# GroupKFold model comparison (trap #1: split by DOCUMENT, never token)
# ---------------------------------------------------------------------------

def compare_models(models: dict, docs: list[Document], n_splits: int = 4,
                   verbose: bool = True) -> dict[str, dict]:
    """Cross-validate sklearn-style classifiers on featurized documents.

    `models` maps name -> unfitted estimator with fit/predict on (X, y).
    Returns {name: {token_macro_f1_mean/std, field_macro_f1_mean/std, fit_seconds}}.
    """
    import time

    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    from ml.features import featurize_dataset

    X, y, groups = featurize_dataset(docs)
    gkf = GroupKFold(n_splits=n_splits)
    results: dict[str, dict] = {}

    for name, proto in models.items():
        tok_f1s, field_f1s, fit_times = [], [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            model = clone(proto)
            t0 = time.perf_counter()
            model.fit(X[train_idx], y[train_idx])
            fit_times.append(time.perf_counter() - t0)
            y_pred = model.predict(X[test_idx])
            tok_f1s.append(token_macro_f1(y[test_idx], y_pred))

            # Field-level on the held-out documents.
            test_docs_idx = sorted(set(int(g) for g in groups[test_idx]))
            from ml.labeling import ID2TAG
            tag_lists, fold_docs = [], []
            for di in test_docs_idx:
                mask = groups == di
                pred_ids = model.predict(X[mask])
                tag_lists.append([ID2TAG[int(p)] for p in pred_ids])
                fold_docs.append(docs[di])
            field_f1s.append(
                evaluate_field_extraction(fold_docs, tag_lists).macro_f1())

        results[name] = {
            "token_macro_f1_mean": float(np.mean(tok_f1s)),
            "token_macro_f1_std": float(np.std(tok_f1s)),
            "field_macro_f1_mean": float(np.mean(field_f1s)),
            "field_macro_f1_std": float(np.std(field_f1s)),
            "fit_seconds": float(np.mean(fit_times)),
        }
        if verbose:
            r = results[name]
            print(f"{name:>22}  token-F1 {r['token_macro_f1_mean']:.3f}±{r['token_macro_f1_std']:.3f}"
                  f"  field-F1 {r['field_macro_f1_mean']:.3f}±{r['field_macro_f1_std']:.3f}"
                  f"  fit {r['fit_seconds']:.1f}s")

    return results

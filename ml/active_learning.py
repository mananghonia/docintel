"""Active learning: which documents should a human label next?

Acquisition functions score a document's informativeness from the model's
token probabilities; the diversity stage (PCA + KMeans over layout
signatures) stops the picker from spending its whole budget on 50
near-identical invoices from one vendor.

The headline experiment lives in simulate_active_learning(): F1 vs number
of labeled documents, one curve per strategy.
"""

from __future__ import annotations

import numpy as np

from ml.labeling import Document, O_TAG, TAG2ID

# ---------------------------------------------------------------------------
# Acquisition functions (document score = how much we'd learn by labeling it)
# ---------------------------------------------------------------------------

def _informative_mask(probs: np.ndarray) -> np.ndarray:
    """Weight tokens the model thinks are NOT background: O-dominated tokens
    carry little field information, so score over likely-entity tokens."""
    o_id = TAG2ID[O_TAG]
    return probs[:, o_id] < 0.98


def least_confidence(probs: np.ndarray) -> float:
    """Mean (1 - max prob) over informative tokens. Higher = more uncertain."""
    mask = _informative_mask(probs)
    p = probs[mask] if mask.any() else probs
    return float((1.0 - p.max(axis=1)).mean())


def margin(probs: np.ndarray) -> float:
    """Mean (1 - (p1 - p2)): small top-2 gap means the model is torn between
    two labels — more informative than flat low confidence. Higher = pick me."""
    mask = _informative_mask(probs)
    p = probs[mask] if mask.any() else probs
    part = np.sort(p, axis=1)
    return float((1.0 - (part[:, -1] - part[:, -2])).mean())


def entropy(probs: np.ndarray) -> float:
    mask = _informative_mask(probs)
    p = probs[mask] if mask.any() else probs
    return float(-(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1).mean())


ACQUISITIONS = {
    "least_confidence": least_confidence,
    "margin": margin,
    "entropy": entropy,
}


# ---------------------------------------------------------------------------
# Layout signatures + diversity selection
# ---------------------------------------------------------------------------

def layout_signature(doc: Document, grid: int = 8) -> np.ndarray:
    """A doc's layout as a grid x grid ink-density histogram (flattened).
    Documents from the same vendor/template land close together."""
    sig = np.zeros((grid, grid), dtype=np.float32)
    for tok in doc.tokens:
        gx = min(int(tok.cx / max(doc.page_width, 1) * grid), grid - 1)
        gy = min(int(tok.cy / max(doc.page_height, 1) * grid), grid - 1)
        sig[gy, gx] += 1
    total = sig.sum()
    return (sig / total).ravel() if total else sig.ravel()


def select_batch(
    docs: list[Document],
    doc_probs: list[np.ndarray],
    batch_size: int,
    acquisition: str = "margin",
    diversity: bool = True,
    random_state: int = 0,
) -> list[int]:
    """Pick indices of the docs to send for labeling.

    diversity=True: PCA the layout signatures, KMeans into batch_size
    clusters, take the most-uncertain doc FROM EACH cluster. Uncertainty
    alone collapses onto one confusing template.
    """
    acq = ACQUISITIONS[acquisition]
    scores = np.array([acq(p) if len(p) else 0.0 for p in doc_probs])

    if not diversity or batch_size >= len(docs):
        return list(np.argsort(-scores)[:batch_size])

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    sigs = np.stack([layout_signature(d) for d in docs])
    n_comp = min(10, sigs.shape[1], len(docs) - 1)
    reduced = PCA(n_components=n_comp, random_state=random_state).fit_transform(sigs)
    clusters = KMeans(n_clusters=batch_size, n_init=4,
                      random_state=random_state).fit_predict(reduced)

    chosen: list[int] = []
    for c in range(batch_size):
        members = np.flatnonzero(clusters == c)
        if len(members):
            chosen.append(int(members[np.argmax(scores[members])]))
    # KMeans can produce empty clusters; top up with best unchosen scores.
    if len(chosen) < batch_size:
        for i in np.argsort(-scores):
            if int(i) not in chosen:
                chosen.append(int(i))
            if len(chosen) == batch_size:
                break
    return chosen


# ---------------------------------------------------------------------------
# The headline experiment: label-efficiency curves
# ---------------------------------------------------------------------------

def _simulate_one(train_docs, test_docs, strategy, seed_size, batch_size,
                  n_rounds, random_state):
    """One AL run of one strategy at one seed. Returns list of per-round F1."""
    from ml.evaluate import evaluate_field_extraction
    from ml.features import featurize_dataset, featurize_document
    from ml.labeling import ID2TAG
    from ml.models_classical import make_models

    rng = np.random.RandomState(random_state)
    X_all, y_all, groups = featurize_dataset(train_docs)
    labeled = list(rng.choice(len(train_docs), size=seed_size, replace=False))
    curve = []
    for _ in range(n_rounds + 1):
        mask = np.isin(groups, labeled)
        model = make_models(fast=True)["xgboost"]
        model.fit(X_all[mask], y_all[mask])
        classes = model.classes_
        tag_lists = [[ID2TAG[int(p)] for p in model.predict(featurize_document(doc))]
                     for doc in test_docs]
        f1 = evaluate_field_extraction(test_docs, tag_lists).macro_f1()
        curve.append({"n_labeled": len(labeled), "field_f1": f1})

        pool = [i for i in range(len(train_docs)) if i not in labeled]
        if not pool:
            break
        k = min(batch_size, len(pool))
        if strategy == "random":
            picked = list(rng.choice(pool, size=k, replace=False))
        else:
            acq = "margin" if strategy.startswith("margin") else strategy
            pool_docs = [train_docs[i] for i in pool]
            pool_probs = [_expand(model.predict_proba(featurize_document(d)), classes)
                          for d in pool_docs]
            local = select_batch(pool_docs, pool_probs, k, acquisition=acq,
                                 diversity=strategy.endswith("+diversity"),
                                 random_state=random_state)
            picked = [pool[i] for i in local]
        labeled.extend(int(i) for i in picked)
    return curve


def simulate_active_learning(
    train_docs: list[Document],
    test_docs: list[Document],
    strategies: list[str] = ("random", "least_confidence", "margin", "margin+diversity"),
    seed_size: int = 20,
    batch_size: int = 10,
    n_rounds: int = 8,
    seeds: tuple = (0, 1, 2),
) -> dict[str, list[dict]]:
    """Simulate AL rounds with an XGBoost tagger, averaged over `seeds`.

    Averaging is not optional: with one seed the strategy curves cross inside
    the run-to-run noise band and a single run 'proves' whichever seed was
    lucky. Returns {strategy: [{n_labeled, f1_mean, f1_std}, ...]}.
    """
    results: dict[str, list[dict]] = {}
    for strategy in strategies:
        per_seed = [_simulate_one(train_docs, test_docs, strategy, seed_size,
                                  batch_size, n_rounds, s) for s in seeds]
        n = min(len(c) for c in per_seed)
        curve = []
        for r in range(n):
            f1s = [per_seed[si][r]["field_f1"] for si in range(len(seeds))]
            curve.append({"n_labeled": per_seed[0][r]["n_labeled"],
                          "f1_mean": float(np.mean(f1s)),
                          "f1_std": float(np.std(f1s))})
        results[strategy] = curve
        print(f"{strategy:>20}: " +
              " ".join(f"{c['f1_mean']:.2f}±{c['f1_std']:.2f}" for c in curve))
    return results


def _expand(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Expand (n, len(classes)) proba to (n, NUM_TAGS) with zeros elsewhere."""
    from ml.labeling import NUM_TAGS

    full = np.zeros((probs.shape[0], NUM_TAGS), dtype=probs.dtype)
    for j, c in enumerate(classes):
        full[:, int(c)] = probs[:, j]
    return full

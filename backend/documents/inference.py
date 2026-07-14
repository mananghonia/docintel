"""Prediction glue for the backend.

Order of preference:
1. Remote model server (MODEL_SERVER_URL set) — production path.
2. In-process champion model loaded from MODEL_DIR (joblib bundle written
   by training) — dev/worker path.
3. Rule baseline — always works, day-one path.

All three return the same shape: (tags, probs) with probs (n_tokens, NUM_TAGS).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from django.conf import settings

from ml.labeling import Document, ID2TAG, NUM_TAGS, TAG2ID

CHAMPION_BUNDLE = "champion.joblib"


@lru_cache(maxsize=1)
def _load_champion(cache_key: str):
    """cache_key = file mtime string, so promotion invalidates the cache."""
    import joblib

    return joblib.load(Path(settings.MODEL_DIR) / CHAMPION_BUNDLE)


def champion_available() -> bool:
    return (Path(settings.MODEL_DIR) / CHAMPION_BUNDLE).exists()


def predict_document(doc: Document) -> tuple[list[str], np.ndarray, str]:
    """Returns (tags, probs, engine): engine in {remote, champion, baseline}."""
    if settings.MODEL_SERVER_URL:
        tags, probs, engine = _predict_remote(doc)
    elif champion_available():
        tags, probs, engine = _predict_champion(doc)
    else:
        return _predict_baseline(doc)
    # The champion is trained on synthetic GST invoices and misses fields on
    # out-of-distribution real invoices (different label wording, layout,
    # currency). The rule extractor keys on regex/anchors and is layout-
    # robust, so use it to backfill fields the champion found nowhere.
    return _merge_rules(doc, tags, probs, engine)


def _merge_rules(doc: Document, tags, probs, engine):
    """Combine champion output with the rule extractor.

    Two cases where the rule tag wins over the champion's:
    1. The champion produced no token for this field at all (a miss).
    2. The champion did tag this field somewhere, but only with confidence
       below RULES_OVERRIDE_BELOW — on out-of-distribution invoices the
       champion is often confidently wrong on a stray token, so a
       low-confidence learned guess should defer to the layout-robust rule.

    Rule-supplied tokens get a fixed modest confidence so they always surface
    for human review. High-confidence champion spans are never overwritten.
    """
    from ml.baseline import predict as baseline_predict
    from ml.labeling import tag_field

    # Best champion confidence seen per field (max over its tokens).
    champ_conf: dict[str, float] = {}
    for t, p in zip(tags, probs):
        f = tag_field(t)
        if f:
            champ_conf[f] = max(champ_conf.get(f, 0.0), float(np.max(p)))

    override_below = settings.RULES_OVERRIDE_BELOW
    rule_tags = baseline_predict(doc)
    merged = list(tags)
    probs = np.asarray(probs, dtype=np.float64).copy()
    filled = False
    for i, rt in enumerate(rule_tags):
        rf = tag_field(rt)
        if not rf:
            continue
        champ_has_confident = champ_conf.get(rf, 0.0) >= override_below
        if not champ_has_confident and merged[i] == "O":
            merged[i] = rt
            probs[i] = 0.0
            probs[i, TAG2ID[rt]] = 0.55  # detected by rules; review it
            filled = True
    # Drop any weak champion span the rules replaced, so it doesn't linger
    # as a duplicate low-confidence guess for the same field.
    replaced = {tag_field(rt) for rt in rule_tags if tag_field(rt)
                and champ_conf.get(tag_field(rt), 0.0) < override_below}
    for i, t in enumerate(tags):
        f = tag_field(t)
        if f in replaced and merged[i] == t and float(np.max(probs[i])) < override_below:
            merged[i] = "O"
    return merged, probs, (engine + "+rules" if filled else engine)


def _predict_baseline(doc: Document):
    from ml.baseline import predict as baseline_predict

    tags = baseline_predict(doc)
    # Rules have no probabilities; use a flat, deliberately modest confidence
    # so everything routes to review until a learned champion exists.
    probs = np.full((len(tags), NUM_TAGS), 0.0, dtype=np.float32)
    for i, t in enumerate(tags):
        probs[i, TAG2ID[t]] = 0.75
    return tags, probs, "baseline"


def _predict_champion(doc: Document):
    from ml.calibration import probs_to_logits
    from ml.features import featurize_document

    path = Path(settings.MODEL_DIR) / CHAMPION_BUNDLE
    bundle = _load_champion(str(path.stat().st_mtime_ns))
    model, scaler = bundle["model"], bundle.get("scaler")

    X = featurize_document(doc)
    raw = model.predict_proba(X)
    if scaler is not None:
        raw = scaler.transform(probs_to_logits(raw))
    # Columns follow model.classes_ (original tag ids); expand to full space.
    probs = np.zeros((raw.shape[0], NUM_TAGS), dtype=np.float64)
    for j, c in enumerate(model.classes_):
        probs[:, int(c)] = raw[:, j]
    tags = [ID2TAG[int(i)] for i in probs.argmax(axis=1)]
    return tags, probs, f"champion:{bundle.get('version', '?')}"


def _predict_remote(doc: Document):
    import urllib.request

    req = urllib.request.Request(
        settings.MODEL_SERVER_URL.rstrip("/") + "/predict",
        data=json.dumps({"document": doc.to_dict()}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read())
    return (payload["tags"], np.asarray(payload["probs"]),
            f"remote:{payload.get('model_version', '?')}")


# ---------------------------------------------------------------------------
# tags+probs -> reviewable field list
# ---------------------------------------------------------------------------

def build_fields(doc: Document, tags: list[str], probs: np.ndarray) -> list[dict]:
    """Group tagged tokens into field spans with value, confidence and bbox,
    then run business-rule postprocessing on values and confidences."""
    from ml.labeling import tag_field
    from ml.postprocess import postprocess_fields

    # Collect every span for each field, then keep the highest-confidence one.
    # (First-span-wins silently drops a page-2 total when a stray page-1 token
    # matched, and loses the stronger of two candidate spans.)
    all_spans: dict[str, list[dict]] = {}
    current: str | None = None
    cur_span: dict | None = None
    for i, (tok, tag) in enumerate(zip(doc.tokens, tags)):
        f = tag_field(tag)
        if f is None:
            current, cur_span = None, None
            continue
        fresh = tag.startswith("B-") or f != current
        if fresh:
            cur_span = {"tokens": [], "confs": [], "page": tok.page}
            all_spans.setdefault(f, []).append(cur_span)
        current = f
        cur_span["tokens"].append(tok)
        cur_span["confs"].append(float(probs[i].max()))

    spans = {f: max(cands, key=lambda s: float(np.mean(s["confs"])))
             for f, cands in all_spans.items()}

    values = {f: " ".join(t.text for t in s["tokens"]) for f, s in spans.items()}
    confs = {f: float(np.mean(s["confs"])) for f, s in spans.items()}
    processed = postprocess_fields(values, confs)

    fields = []
    for f, s in spans.items():
        toks = s["tokens"]
        p = processed[f]
        fields.append({
            "field": f,
            "raw": p["raw"],
            "value": p["value"],
            "confidence": round(p["confidence"], 4),
            "flags": p["flags"],
            "bbox": {
                "x0": min(t.x0 for t in toks), "y0": min(t.y0 for t in toks),
                "x1": max(t.x1 for t in toks), "y1": max(t.y1 for t in toks),
                "page": s["page"],
            },
        })
    return fields

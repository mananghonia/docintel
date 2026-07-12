"""FastAPI model server: holds the champion model, serves /predict.

Separated from Django on purpose: the app tier can redeploy without dropping
the model from memory, and the model can be promoted (champion.joblib
replaced) without touching the app. The server checks the artifact's mtime
per request and hot-reloads when training promotes a new version.

Run:  uvicorn model_server.main:app --port 8001
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.labeling import Document, ID2TAG, NUM_TAGS, TAG2ID  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "data" / "models"))
BUNDLE = MODEL_DIR / "champion.joblib"

app = FastAPI(title="DocIntel model server", version="0.1.0")

_lock = threading.Lock()
_state: dict = {"bundle": None, "mtime": None}


def _current_bundle():
    """Load champion.joblib, hot-reloading if the file changed (promotion)."""
    if not BUNDLE.exists():
        return None
    mtime = BUNDLE.stat().st_mtime_ns
    with _lock:
        if _state["mtime"] != mtime:
            import joblib

            _state["bundle"] = joblib.load(BUNDLE)
            _state["mtime"] = mtime
        return _state["bundle"]


class PredictRequest(BaseModel):
    document: dict


class PredictResponse(BaseModel):
    tags: list[str]
    probs: list[list[float]]
    model_version: str
    engine: str


@app.get("/health")
def health():
    bundle = _current_bundle()
    return {"status": "ok",
            "champion": None if bundle is None else f"v{bundle.get('version', '?')}"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        doc = Document.from_dict(req.document)
    except (KeyError, TypeError) as e:
        raise HTTPException(422, f"bad document payload: {e}")

    bundle = _current_bundle()
    if bundle is None:
        # No trained model yet: serve the rule baseline so the system works
        # from day one (same behaviour as backend in-process fallback).
        from ml.baseline import predict as baseline_predict

        tags = baseline_predict(doc)
        probs = np.zeros((len(tags), NUM_TAGS), dtype=np.float32)
        for i, t in enumerate(tags):
            probs[i, TAG2ID[t]] = 0.75
        return PredictResponse(tags=tags, probs=probs.tolist(),
                               model_version="none", engine="baseline")

    from ml.calibration import probs_to_logits
    from ml.features import featurize_document

    model, scaler = bundle["model"], bundle.get("scaler")
    X = featurize_document(doc)
    raw = model.predict_proba(X)
    if scaler is not None:
        raw = scaler.transform(probs_to_logits(raw))
    probs = np.zeros((raw.shape[0], NUM_TAGS), dtype=np.float64)
    for j, c in enumerate(model.classes_):
        probs[:, int(c)] = raw[:, j]
    tags = [ID2TAG[int(i)] for i in probs.argmax(axis=1)]
    return PredictResponse(tags=tags, probs=probs.tolist(),
                           model_version=f"v{bundle.get('version', '?')}",
                           engine="champion")

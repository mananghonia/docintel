# DocIntel

Invoice extraction that reads messy documents, outputs structured data with a
**confidence score per field**, routes uncertain fields to a human reviewer,
and **retrains itself on those corrections** — measurably improving over time.

The extraction is table stakes. The self-improving loop is the project.

## Architecture

```
upload / synthetic ingest
   │
   ▼
Django + DRF (backend/) ──► Celery worker: OCR → predict → postprocess
   │                              │
   │                              ▼
   │                        FastAPI model server (model_server/)
   │                        holds champion.joblib, hot-reloads on promotion
   ▼
React review UI (frontend/)
   confidence-coded bboxes · low-confidence fields pre-focused ·
   draw-a-box for missed fields (hard negatives) · every delta stored
   │
   ▼
Celery Beat: ≥50 new verified docs → retrain → champion/challenger gate
   challenger promoted ONLY if it beats the champion on a frozen holdout;
   rejections are logged in TrainingRun, not silently discarded
```

Three services on purpose: Django holds a model badly (memory per worker,
slow deploys). A separate model server hot-swaps model versions without
redeploying the app.

## Quick start (no Docker, no OCR, no GPU needed)

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt

# 1. Prove the ML pipeline end to end (synth → features → models → calibration)
python scripts/smoke_test.py

# 2. Prove the full product loop (ingest → review → retrain → champion serves)
python scripts/backend_e2e.py

# 3. Run the app
cd backend && python manage.py migrate && python manage.py runserver   # :8000
cd frontend && npm install && npm run dev                              # :5173
```

Without a Redis broker configured, Celery runs tasks eagerly in-process —
the whole system works with zero services. `docker-compose up` brings up the
full stack (Postgres, Redis, MinIO, MLflow, worker, beat, model server).

Use **"Ingest 10 synthetic"** in the UI to demo without any real files:
synthetic invoices carry their own tokens, so no OCR install is needed.
Real uploads need `tesseract` (+ `poppler` for PDFs) on PATH.

## Experiments

```bash
python scripts/train.py compare            # 6-model GroupKFold table
python scripts/train.py ablation           # RNN vs LSTM vs BiLSTM (needs torch)
python scripts/train.py active             # label-efficiency curves — the money slide
python scripts/train.py fit                # train + calibrate, write champion.joblib
```

`--real DIR` mixes in hand-labeled real documents (Document-dict JSONs);
evaluation then becomes **real-only** — synthetic-only models shatter on
reality, and knowing that is part of the point.

## The model stack

| Tier | Model                                   | Module                  |
|------|-----------------------------------------|-------------------------|
| 0    | Regex + keyword anchors                 | `ml/baseline.py`        |
| 1    | ~150 engineered features → RF / XGBoost | `ml/features.py`, `ml/models_classical.py` |
| 2    | RNN → LSTM → BiLSTM tagger (ablation)   | `ml/models_bilstm.py`   |

Plus the pieces that make confidence mean something:

- **Temperature scaling** (`ml/calibration.py`): one scalar fitted on
  validation; ECE reported before/after. Review routing depends on honest
  confidence.
- **Business rules** (`ml/postprocess.py`): GSTIN mod-36 checksum, date and
  amount normalisation, and cross-field arithmetic — if subtotal + tax ≠
  total, every amount field's confidence is cut, even when the model was sure.
- **Active learning** (`ml/active_learning.py`): margin sampling + PCA/KMeans
  layout diversity, so the label budget isn't spent on 50 near-identical
  invoices from one vendor.
- **Drift** (`backend/monitoring/drift.py`): PSI over the confidence
  distribution — needs no ground truth, so it's the earliest warning signal.

## The three traps (encoded in the code, not just the docs)

1. **Never split train/test by token.** Tokens from one invoice are massively
   correlated; token splits leak layout and inflate F1 by ~10 points.
   `ml/evaluate.py` only exposes GroupKFold by document.
2. **Verify label alignment visually.** `ml.labeling.visualize_alignment()`
   renders tokens + annotation boxes; look at 5 documents before training
   anything. The smoke test writes one to `data/synthetic/`.
3. **Never report accuracy.** ~85% of tokens are `O`; predicting `O`
   everywhere scores 85%. Everything here reports macro-F1, per field.

## Repository layout

```
ml/                 the ML package (imported by worker + model server)
backend/            Django + DRF + Celery: documents, training, monitoring
model_server/       FastAPI /predict, hot-reloads champion.joblib
frontend/           React review UI: canvas, confidence routing, dashboards
scripts/            smoke_test, backend_e2e, train
data/models/        champion.joblib + experiment outputs (gitignored)
```

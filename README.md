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
python scripts/train.py compare --hard     # 6-model GroupKFold table
python scripts/train.py ablation --hard    # RNN vs LSTM vs BiLSTM (needs torch)
python scripts/train.py visual --hard      # BiLSTM vs BiLSTM+CNN (Tier 3)
python scripts/train.py active --hard      # label-efficiency curves — the money slide
python scripts/train.py fit                # train + calibrate, write champion.joblib
python scripts/benchmark_onnx.py           # torch vs ONNX fp32 vs ONNX INT8
```

`--hard` switches the generator to what scanned invoices actually look like:
8 vendor template families on a Zipf distribution (layouts cluster; the tail
is rare), label-above layouts that break same-line anchors, distractor
fields (quotation no, delivery date, IRN, bank a/c), OCR character
corruption, anchor-token dropout, bbox jitter. On easy data everything
saturates ≥0.94 F1 and comparisons are meaningless.

## Results (hard synthetic, field macro-F1, GroupKFold by document)

| Tier | Model | Field F1 |
|------|-------|----------|
| 0 | Regex + keyword anchors | 0.37 (0.72 on easy data — anchors shatter) |
| 1 | Naive Bayes | 0.53 |
| 1 | KNN | 0.73 |
| 1 | XGBoost / LogReg / LinearSVM | 0.77 |
| 1 | Random Forest | 0.81 |
| 2 | vanilla RNN | 0.74 |
| 2 | LSTM | 0.73 |
| 2 | **BiLSTM** | **0.77** |
| 3 | **BiLSTM + CNN visual stream** | **0.78** |

Reading the table honestly: Tier 1 models receive hand-engineered context
(±2-token window, keyword-anchor distances); Tier 2/3 receive **context-free
per-token features only** and must learn context through recurrence. The
BiLSTM matching XGBoost while learning its own context — and the visual
stream adding +2.7 points on top — is the point. (First lesson learned the
hard way: feeding the context window TO the sequence models made the
RNN/LSTM/BiLSTM comparison completely flat; the architecture only matters
when it has a job to do.)

**Active learning** (XGBoost, seed 20 docs, batch 10): random sampling
plateaus at ~0.70 after 100 labels; least-confidence reaches the same F1
with ~50 — **half the labeling budget**. Margin+diversity *underperforms*
plain margin on this data: with only 8 layout families, KMeans-forced
cluster coverage wastes budget on already-easy templates. Diversity should
pay off when layouts number in the hundreds (real vendors), and the
simulation exists to test exactly that once real data lands.

**Serving** (BiLSTM, single CPU thread, per document):

| Engine | p50 | p95 | p99 | Field F1 | Size |
|--------|-----|-----|-----|----------|------|
| torch eager | 3.2ms | 4.3ms | 4.6ms | 0.7753 | — |
| ONNX fp32 | 2.5ms | 3.0ms | 3.2ms | 0.7753 | 1.34 MB |
| ONNX INT8 | 1.0ms | 1.1ms | 1.1ms | 0.7718 | 0.35 MB |

INT8 quantization: **3.9× faster at p95, 3.9× smaller, −0.0035 F1**.

## Real data: the gap, measured

150 real scanned receipts (CORD v2, `scripts/import_real.py cord`), three
training regimes, one real test set (`scripts/train.py gap`):

| Training data | Field macro-F1 on real docs |
|---------------|------------------------------|
| synthetic only | **0.11** — shatters, as predicted |
| real only (100 docs) | 0.40 |
| synthetic + real | **0.43** |

"Synthetic-only models shatter on reality" is a measurement here, not a
slogan: 0.11 vs 0.43. Synthetic data still earns +0.03 as pretraining on
top of real labels. (CORD is receipts, so only the amount fields map onto
the invoice schema — per-field numbers are what matter.)

Active learning on the real pool is *inconclusive at this scale*: with a
120-doc pool, a 30-doc test set and one seed, the strategy curves cross
inside the noise band. The synthetic study shows the mechanism; validating
it on real data needs several hundred docs and multiple seeds — documented
here so nobody mistakes one noisy run for a result.

**The OCR upload path** (tesseract) is exercised end to end: a rotated,
blurred, noisy scan produced mangled tokens, the GSTIN checksum penalty cut
that field's confidence to 0.30, and the document routed to review instead
of being silently accepted — the failure path working as designed.
Preprocessing lesson learned the measured way: median filtering + a fixed
binarisation threshold destroyed thin strokes (65 OCR tokens → 6); tesseract
binarises internally, so preprocessing now only does autocontrast + deskew.

To label your own invoices, the review UI is the tool: upload, correct,
then `python scripts/import_real.py export-verified`.

`--real DIR` mixes in hand-labeled real documents (Document-dict JSONs);
evaluation then becomes **real-only** — synthetic-only models shatter on
reality, and knowing that is part of the point.

## The model stack

| Tier | Model                                   | Module                  |
|------|-----------------------------------------|-------------------------|
| 0    | Regex + keyword anchors                 | `ml/baseline.py`        |
| 1    | ~150 engineered features → RF / XGBoost | `ml/features.py`, `ml/models_classical.py` |
| 2    | RNN → LSTM → BiLSTM tagger (ablation)   | `ml/models_bilstm.py`   |
| 3    | + CNN visual stream on token crops      | `ml/models_cnn.py`      |

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

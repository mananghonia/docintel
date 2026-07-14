# DocIntel

Invoice extraction that reads messy documents, outputs structured data with a
**confidence score per field**, routes uncertain fields to a human reviewer,
and **retrains itself on those corrections** — measurably improving over time.

The extraction is table stakes. The self-improving loop is the project.

**[→ Results dashboard](docs/results.html)** — the model ladder, the real-data
self-improving loop (real-invoice F1 0.41 → 0.57 after one retrain), active-learning
label efficiency, and ONNX INT8 serving, all from measured numbers. Open the file
locally, or serve `docs/` and view in a browser.

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
pip install -r requirements.txt                    # fully working core
# optional: deep-learning tiers (Tier 2/3 + ONNX) — needs the CPU torch index
pip install -r requirements-dl.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 1. Prove the ML pipeline end to end (synth → features → models → calibration)
python scripts/smoke_test.py

# 2. Prove the full product loop (ingest → review → retrain → champion serves)
python scripts/backend_e2e.py

# 3. Run the app
cd backend && python manage.py migrate && python manage.py runserver   # :8000
cd frontend && npm install && npm run dev                              # :5174
```

`runserver` (no `--noreload`) auto-reloads on code changes — edit backend code
and it restarts itself; only `--noreload` needs a manual restart to pick up edits.

Without a Redis broker configured, Celery runs tasks eagerly in-process —
the whole system works with zero services. `docker-compose up` brings up the
full stack (Postgres, Redis, MinIO, MLflow, worker, beat, model server).

## Deploy (one container, runs anywhere)

The whole app ships as a **single self-contained image**: Django (gunicorn)
serves the built React SPA via WhiteNoise, predicts in-process, and runs tasks
eagerly — no Redis/worker/model-server needed. A champion model is trained at
build time and baked in, so extraction works on the first request; a few
synthetic docs are seeded on first boot so the dashboard isn't empty.

```bash
docker build -t docintel .
docker run -p 8000:8000 docintel        # open http://localhost:8000
# or, with persistence (DB + model + uploads survive restarts):
docker compose -f docker-compose.deploy.yml up --build -d
```

Demo-open by default (`DEBUG=0`, auth off). To lock it down, set
`REQUIRE_AUTH=1` and mint a token with `manage.py create_api_token`.

**[→ DEPLOY.md](DEPLOY.md)** has a copy-paste Fly.io walkthrough (`fly.toml` is
in the repo) plus the generic Docker-host path. On any Docker-capable host this
is a one-service deploy; point it at the `Dockerfile` and expose port 8000. For
the full multi-service architecture instead, use `docker-compose.yml`.

*(Both compose files and the image are written and verified by inspection and
by running every build step locally — the Django prod-serving, collectstatic,
model bake, and gunicorn config — but not yet run inside Docker on this
machine, which has no Docker daemon.)*

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
| 0 | Regex + keyword anchors | 0.38 (0.73 on easy data — anchors shatter) |
| 1 | Naive Bayes | 0.56 |
| 1 | KNN | 0.78 |
| 1 | XGBoost / LogReg / LinearSVM | 0.81–0.82 |
| 1 | Random Forest | **0.86** |
| 2 | vanilla RNN | 0.794 ± 0.015 (3 seeds) |
| 2 | LSTM | 0.804 ± 0.009 |
| 2 | BiLSTM | 0.793 ± 0.024 |
| 3 | BiLSTM + CNN visual stream | 0.821 ± 0.041 (vs 0.834 ± 0.034 without CNN, same setup) |

Reading the table honestly — this experiment taught two lessons the hard way:

1. **Feature design shapes what an ablation can measure.** Tier 1 models get
   hand-engineered context (±2-token window, keyword-anchor distances);
   Tier 2/3 get context-free per-token features and must learn context
   through recurrence — otherwise the RNN comparison measures nothing.
2. **Single-seed neural comparisons lie.** The first single-seed run "showed"
   BiLSTM beating RNN by 3 points and the CNN adding 2.7 more. Re-run over
   3 seeds, every architecture gap collapsed inside the noise band
   (RNN ≈ LSTM ≈ BiLSTM; CNN gain not measurable on clean synthetic
   renders). At a few-hundred-document scale, **Random Forest on engineered
   features beats every neural model here** — sequence models need more
   data than this to pay for themselves, and any claimed neural win without
   seed variance is a coin flip wearing a lab coat.

The sequence models do learn context (0.79–0.80 with context-free input vs
0.56 for the also-context-free Naive Bayes), they just don't beat
hand-engineered context at this data size. On real-scale data (thousands of
docs, real scan artifacts) the expected ordering may flip — that experiment
is wired and waiting for data.

**Active learning** (XGBoost, seed 20 docs, batch 10, **averaged over 3
seeds** — single-seed AL curves cross inside the noise band): random
sampling plateaus at ~0.74 after 100 labels; least-confidence reaches that
same F1 with ~40 labels — **~60% less labeling budget** — and keeps climbing
to 0.80. Margin+diversity *underperforms* plain margin on this data: with
only 8 layout families, KMeans-forced cluster coverage wastes budget on
already-easy templates. Diversity should pay off when layouts number in
the hundreds (real vendors), and the simulation exists to test exactly
that once real data lands.

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

**The loop works on real data, end to end** (`scripts/seed_real.py`). Loading
those 150 real documents into the app as verified, then retraining through
the champion/challenger gate on a *frozen real holdout*:

- real-invoice field macro-F1 **0.405 → 0.567 (+0.16)**
- calibration **ECE 0.058 → 0.026**
- promoted only after clearing a paired-bootstrap significance gate
  (win-rate 0.86 over the holdout documents)

This is the whole thesis reduced to one command: real corrections in, a
measurably better *and better-calibrated* model out, promoted only when the
improvement is real. It also surfaced a genuine lesson — the first attempt
was rejected because a large leftover *synthetic* holdout drowned the real
signal: **your holdout must reflect the distribution you actually serve.**

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

**Getting more real data.** Three importers into the same `data/labeled/` format:

- **DocILE** (`import_real.py docile`) — real invoices mapped onto the full
  12-field schema, the right dataset to push real-invoice F1 past the CORD
  ceiling. One-time: get a token at docile.rossum.ai, `pip install
  docile-benchmark`, run their `download_dataset.sh`, then
  `python scripts/import_real.py docile --docile-path data/docile`.
- **CORD** (`import_real.py cord`) — receipts; only amount fields map.
- **Your own invoices** — upload through the review UI, correct, then
  `python scripts/import_real.py export-verified`. Best data of all: it
  specialises the model to exactly what you'll serve.

Then `python scripts/seed_real.py` retrains through the gate on the new data.
(Downloaded dataset JSONs are gitignored — licenses vary; rerun the importer.)

## Failure modes, named

Every one of these was hit for real during development, not imagined:

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| OOD documents | US receipt → `vendor_name="Receipt"` at 1.00 confidence, auto-accepted | critical-field gate: no total/invoice number → review regardless of confidence; PSI drift alert fired correctly |
| OOD real invoice | champion (synthetic-trained) found 1 of 12 fields on a real US invoice | hybrid extractor: rules backfill fields the champion misses AND override its low-confidence guesses |
| OCR value corruption | `total=1019067.69` read as `1.0` | unfixable downstream; arithmetic check flags it, reviewer corrects |
| Overzealous preprocessing | binarise+median destroyed thin strokes, 65 tokens → 6 | measured, removed; tesseract binarises internally |
| Dense-page rendering | 1806-token research report drawn at fixed 16px = overlapping smear | scale each glyph to its OCR bbox |
| Anchor dropout | faint print eats the "Total:" label, rules find nothing | learned models key on many signals; rules baseline honestly collapses (0.73 → 0.38) |
| Truth for unprinted fields | generator claimed due_date even when not rendered → recall deflated for every model | caught by the test suite; truth now only contains rendered fields |
| Split leakage | token-level splits leak layout, inflate F1 ~10 pts | GroupKFold by document is the only split the harness offers |
| Promotion on noise | 3-doc holdout, tie promoted a "new" model | paired bootstrap win-rate gate + larger holdout minimum |

Not yet handled (would be next): handwriting, stamps/watermarks over text,
carried-over totals across pages, rotated scans beyond ±4°.

## Known limitations

Honest inventory of what this project does **not** yet do — stated so the
numbers above aren't mistaken for more than they are:

- **The learned model does not generalise to real invoices.** The champion is
  trained on synthetic GST invoices and scores ~0.11 F1 on real receipts; the
  rule-based hybrid is what makes real uploads usable today. The real fix is
  labeling a few hundred real invoices through the review UI and retraining —
  the loop is built and waiting for that data.
- **No real invoice training set yet.** CORD (the one real dataset wired up)
  is receipts, so only 3 of 12 fields overlap the schema.
- **Active-learning superiority is shown only on synthetic data.** Averaged
  over 3 seeds the ranking is stable there, but on the small real pool the
  curves still cross within noise — needs several hundred real docs to settle.
- **`docker-compose.yml` is written but unrun** (no Docker on the dev machine).
  Treat the single-service local run as the verified path; the compose stack
  likely needs the same IPv4/loopback fixes that surfaced locally.
- **No multi-tenancy.** Token auth exists (see below) but there's no per-user
  data isolation — every authenticated user sees every document.
- **Single-worker assumptions.** The in-process model cache and champion
  hot-reload assume one worker; horizontal scaling needs a shared model store.

## Authentication

Open in local dev, enforced in production. Token + session auth; the API is
`AllowAny` while `DEBUG` is on and flips to `IsAuthenticated` when you set
`DJANGO_DEBUG=0` (or `REQUIRE_AUTH=1`). To use it:

```bash
python backend/manage.py create_api_token --username reviewer --password secret
# -> prints a token; call the API with  Authorization: Token <token>
# or exchange credentials:  POST /api/auth/token/  {username, password}
```

The React client sends the token automatically if one is stored
(`localStorage.setItem("docintel_token", "<token>")`); with nothing stored it
sends no header, so local dev stays friction-free.

## Tests

```bash
python -m pytest tests -q     # 30 tests
```

Coverage: bbox→token alignment, GSTIN checksum, date/amount parsing,
arithmetic consistency, generator determinism, feature shapes, temperature
scaling, field-scoring logic, the **champion/challenger bootstrap gate**, the
**rules-merge / best-span inference glue**, and **PSI drift**.

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

"""Offline training / experimentation entrypoint.

    python scripts/train.py compare [--docs 300]      six-model GroupKFold table
    python scripts/train.py ablation [--docs 300]     RNN vs LSTM vs BiLSTM (needs torch)
    python scripts/train.py active [--docs 400]       label-efficiency curves (the money slide)
    python scripts/train.py fit [--docs 500]          fit + calibrate XGBoost, write champion.joblib

All commands use synthetic data by default; pass --real DIR to load labeled
real documents (one JSON per doc, ml.labeling.Document dict format) — then
training is synthetic+real and EVALUATION IS REAL-ONLY.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_docs(args):
    from ml.labeling import Document, assign_labels
    from ml.synth.generator import generate_dataset

    synth = generate_dataset(args.docs, seed=args.seed, hard=args.hard)
    for d in synth:
        assign_labels(d)
    real = []
    if args.real:
        for p in sorted(Path(args.real).glob("*.json")):
            real.append(assign_labels(Document.from_dict(json.loads(p.read_text()))))
        print(f"loaded {len(real)} real + {len(synth)} synthetic docs "
              f"(evaluation will be real-only)")
    return synth, real


def cmd_compare(args):
    from ml.evaluate import compare_models
    from ml.models_classical import make_models

    synth, real = load_docs(args)
    docs = real if real else synth  # CV over real docs when available
    compare_models(make_models(fast=args.fast), docs, n_splits=4)


def cmd_ablation(args):
    from ml.models_bilstm import run_ablation

    synth, real = load_docs(args)
    if real:
        train, test = synth + real[len(real) // 3:], real[: len(real) // 3]
    else:
        n_test = args.docs // 5
        train, test = synth[n_test:], synth[:n_test]
    run_ablation(train, test, epochs=args.epochs)


def cmd_visual(args):
    from ml.models_cnn import run_visual_ablation

    synth, real = load_docs(args)
    docs = synth + real
    n_test = len(docs) // 5
    run_visual_ablation(docs[n_test:], docs[:n_test], epochs=args.epochs)


def cmd_active(args):
    from ml.active_learning import simulate_active_learning

    synth, real = load_docs(args)
    docs = synth + real
    n_test = len(docs) // 5
    results = simulate_active_learning(docs[n_test:], docs[:n_test])
    out = ROOT / "data" / "models" / "active_learning_curves.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"curves written to {out}")


def cmd_fit(args):
    import joblib
    import numpy as np

    from ml.calibration import TemperatureScaler, probs_to_logits
    from ml.evaluate import evaluate_field_extraction
    from ml.features import featurize_dataset, featurize_document
    from ml.labeling import ID2TAG
    from ml.models_classical import make_models

    synth, real = load_docs(args)
    test = real[: max(len(real) // 3, 1)] if real else synth[: args.docs // 5]
    train = (synth + real[len(real) // 3:]) if real else synth[args.docs // 5:]

    n_val = max(len(train) // 5, 2)
    X, y, _ = featurize_dataset(train[n_val:])
    model = make_models(fast=args.fast)["xgboost"]
    model.fit(X, y)

    Xv, yv, _ = featurize_dataset(train[:n_val])
    pos = {int(c): i for i, c in enumerate(model.classes_)}
    scaler = TemperatureScaler().fit(
        probs_to_logits(model.predict_proba(Xv)),
        np.array([pos.get(int(t), 0) for t in yv]))

    tag_lists = [[ID2TAG[int(p)] for p in model.predict(featurize_document(d))]
                 for d in test]
    f1 = evaluate_field_extraction(test, tag_lists).macro_f1()
    print(f"holdout field macro-F1: {f1:.4f}   T={scaler.temperature:.3f}")

    out = ROOT / "data" / "models" / "champion.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "version": 0}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["compare", "ablation", "visual", "active", "fit"])
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--real", help="directory of labeled real-doc JSONs")
    ap.add_argument("--hard", action="store_true",
                    help="hard synthetic data: template families, OCR noise, distractors")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    {"compare": cmd_compare, "ablation": cmd_ablation, "visual": cmd_visual,
     "active": cmd_active, "fit": cmd_fit}[args.command](args)

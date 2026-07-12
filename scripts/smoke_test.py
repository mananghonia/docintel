"""End-to-end smoke test of the ML pipeline (no OCR, no torch needed).

    synth generator -> assign_labels -> alignment PNG
    -> baseline field-F1
    -> XGBoost vs LogReg (GroupKFold by document)
    -> calibration ECE before/after
    -> postprocess arithmetic check on one doc

Run:  python scripts/smoke_test.py [n_docs]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main(n_docs: int = 80) -> None:
    from ml.baseline import predict as baseline_predict
    from ml.evaluate import compare_models, evaluate_field_extraction
    from ml.labeling import assign_labels, visualize_alignment
    from ml.synth.generator import generate_dataset

    print(f"== 1. Generating {n_docs} synthetic invoices ==")
    docs = generate_dataset(n_docs, seed=42)
    for d in docs:
        assign_labels(d)
    n_labeled = sum(1 for d in docs for t in d.tokens if t.tag != "O")
    n_total = sum(len(d.tokens) for d in docs)
    print(f"   {n_total} tokens, {n_labeled} labeled ({n_labeled / n_total:.1%})")

    out_dir = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = visualize_alignment(docs[0], str(out_dir / "alignment_check.png"))
    print(f"   alignment check image (trap #2 — LOOK at it): {png}")

    print("\n== 2. Tier 0 rule baseline ==")
    tag_lists = [baseline_predict(d) for d in docs]
    scores = evaluate_field_extraction(docs, tag_lists)
    print(f"   baseline field macro-F1: {scores.macro_f1():.3f}")
    for f, m in sorted(scores.per_field().items(), key=lambda kv: kv[1]["f1"]):
        print(f"     {f:>16}  P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}")
    baseline_f1 = scores.macro_f1()

    print("\n== 3. Classical models, GroupKFold by document ==")
    from ml.models_classical import make_models
    models = make_models(fast=True)
    subset = {k: models[k] for k in ("logreg", "xgboost")}
    results = compare_models(subset, docs, n_splits=3)
    assert results["xgboost"]["field_macro_f1_mean"] > baseline_f1, \
        "XGBoost failed to beat the rule baseline — investigate before continuing"

    print("\n== 4. Calibration (temperature scaling) ==")
    from sklearn.model_selection import GroupKFold

    from ml.calibration import TemperatureScaler, probs_to_logits
    from ml.features import featurize_dataset

    X, y, groups = featurize_dataset(docs)
    train_idx, val_idx = next(GroupKFold(n_splits=3).split(X, y, groups))
    xgb = make_models(fast=True)["xgboost"].fit(X[train_idx], y[train_idx])
    probs = xgb.predict_proba(X[val_idx])
    # Map val labels into the encoder's class index space for NLL.
    import numpy as _np
    class_pos = {int(c): i for i, c in enumerate(xgb.classes_)}
    y_val = _np.array([class_pos.get(int(t), 0) for t in y[val_idx]])
    logits = probs_to_logits(probs)
    scaler = TemperatureScaler().fit(logits, y_val)
    rep = scaler.report(logits, y_val)
    print(f"   T = {rep['temperature']:.3f}   ECE {rep['ece_before']:.4f} -> {rep['ece_after']:.4f}")

    print("\n== 5. Post-processing business rules ==")
    from ml.postprocess import postprocess_fields
    truth = docs[0].meta["truth"]
    values = {f: str(truth[f]) for f in ("subtotal", "tax_amount", "total_amount", "vendor_gstin")}
    conf = {f: 0.9 for f in values}
    out = postprocess_fields(values, conf)
    print(f"   consistent doc:  total conf {out['total_amount']['confidence']:.2f} "
          f"flags {out['total_amount']['flags']}")
    values_bad = dict(values, total_amount=str(round(float(truth['total_amount']) * 1.10, 2)))
    out_bad = postprocess_fields(values_bad, conf)
    print(f"   corrupted total: total conf {out_bad['total_amount']['confidence']:.2f} "
          f"flags {out_bad['total_amount']['flags']}")
    assert out_bad["total_amount"]["confidence"] < out["total_amount"]["confidence"]

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)

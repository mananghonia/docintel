"""Deep evaluation of the *deployed extraction pipeline* (champion + rules +
postprocess), per field and per region, on fresh labeled synthetic invoices.
Surfaces the weakest fields to target.

    python scripts/model_eval.py [n]
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import numpy as np


def main(n=200):
    import django
    django.setup()

    from documents.inference import build_fields, predict_document
    from ml.evaluate import FieldScores, _norm_value
    from ml.labeling import FIELDS, assign_labels
    from ml.synth.generator import generate_dataset

    print(f"== evaluating deployed pipeline on {n} fresh hard invoices ==")
    docs = [assign_labels(d) for d in generate_dataset(n, seed=31337, hard=True)]

    by_region = {"in": FieldScores(), "intl": FieldScores()}
    overall = FieldScores()
    conf_correct = []  # (confidence, is_correct) for calibration on fields

    engines = defaultdict(int)
    for d in docs:
        region = "in" if d.meta.get("region") == "in" else "intl"
        truth = d.meta["truth"]
        tags, probs, engine = predict_document(d)
        engines[engine.split(":")[0]] += 1
        fields = build_fields(d, tags, probs)
        predicted = {f["field"]: f["value"] for f in fields}
        conf = {f["field"]: f["confidence"] for f in fields}

        overall.add_document(truth, predicted)
        by_region[region].add_document(truth, predicted)
        for f in FIELDS:
            if f in predicted and f in truth:
                correct = _norm_value(f, predicted[f]) == _norm_value(f, str(truth[f]))
                conf_correct.append((conf[f], 1.0 if correct else 0.0))

    print(f"engines used: {dict(engines)}")

    print(f"\n== overall: field macro-F1 = {overall.macro_f1():.3f} ==")
    per = overall.per_field()
    print(f"{'field':>16}  {'P':>5} {'R':>5} {'F1':>5}  support")
    for f in sorted(per, key=lambda k: per[k]['f1']):
        m = per[f]
        print(f"{f:>16}  {m['precision']:.2f}  {m['recall']:.2f}  {m['f1']:.2f}   {m['support']}")

    print("\n== by region (macro-F1) ==")
    for r, s in by_region.items():
        print(f"  {r:>5}: {s.macro_f1():.3f}")

    print("\n== calibration on field confidences ==")
    from ml.calibration import expected_calibration_error
    if conf_correct:
        c = np.array([x[0] for x in conf_correct])
        y = np.array([x[1] for x in conf_correct])
        ece = expected_calibration_error(c, y)
        print(f"  fields evaluated: {len(c)}  mean conf {c.mean():.3f}  "
              f"accuracy {y.mean():.3f}  ECE {ece:.4f}")

    weak = sorted(per, key=lambda k: per[k]["f1"])[:3]
    weak_str = ", ".join("{} ({:.2f})".format(w, per[w]["f1"]) for w in weak)
    print("\n== weakest fields: " + weak_str + " ==")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)

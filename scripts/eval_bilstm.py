"""Train a BiLSTM and measure per-field F1 through the SAME extraction pipeline
(build_fields + name repair + postprocess) as the XGBoost champion, so the
numbers are directly comparable to scripts/model_eval.py.

    python scripts/eval_bilstm.py [n_train] [n_test] [epochs]
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import numpy as np


def main(n_train=600, n_test=200, epochs=14):
    import django
    django.setup()

    from documents.inference import build_fields
    from ml.evaluate import FieldScores
    from ml.labeling import ID2TAG, assign_labels
    from ml.models_bilstm import SequenceTagger
    from ml.synth.generator import generate_dataset

    print(f"== training BiLSTM on {n_train} hard docs ({epochs} epochs) ==")
    train = [assign_labels(d) for d in generate_dataset(n_train, seed=7, hard=True)]
    test = [assign_labels(d) for d in generate_dataset(n_test, seed=31337, hard=True)]

    tagger = SequenceTagger(rnn_type="bilstm", epochs=epochs, seed=0)
    tagger.fit(train, verbose=False)

    overall = FieldScores()
    for d in test:
        probs = tagger.predict_proba(d)          # (n_tokens, NUM_TAGS)
        tags = [ID2TAG[int(i)] for i in probs.argmax(axis=1)]
        fields = build_fields(d, tags, probs)     # same repair + postprocess
        predicted = {f["field"]: f["value"] for f in fields}
        overall.add_document(d.meta["truth"], predicted)

    per = overall.per_field()
    print(f"\n== BiLSTM overall field macro-F1 = {overall.macro_f1():.3f} ==")
    print(f"{'field':>16}  {'F1':>5}   (XGBoost champion)")
    xgb = {"vendor_name": 0.55, "buyer_name": 0.61, "due_date": 0.66,
           "invoice_date": 0.76, "tax_amount": 0.84, "total_amount": 0.86,
           "subtotal": 0.88, "vendor_gstin": 0.93, "buyer_gstin": 0.94,
           "invoice_number": 0.97, "currency": 0.99, "po_number": 1.00}
    for f in sorted(per, key=lambda k: per[k]["f1"]):
        base = xgb.get(f)
        delta = f"  vs {base:.2f}  ({per[f]['f1'] - base:+.2f})" if base else ""
        print(f"{f:>16}  {per[f]['f1']:.2f}{delta}")


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]) if len(a) > 1 else 600,
         int(a[2]) if len(a) > 2 else 200,
         int(a[3]) if len(a) > 3 else 14)

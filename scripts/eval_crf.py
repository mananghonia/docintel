"""Evaluate a CRF tagger (transition constraints on the engineered features)
per-field, through the same pipeline, vs the XGBoost champion (0.844).

    python scripts/eval_crf.py [n_train] [n_test]
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import numpy as np


def _reading_order(doc):
    return sorted(range(len(doc.tokens)),
                  key=lambda i: (doc.tokens[i].page, doc.tokens[i].cy, doc.tokens[i].cx))


def _seq(doc):
    """Feature-dict sequence + reading-order index list for one document."""
    from ml.features import featurize_document
    X = featurize_document(doc, context=True)
    order = _reading_order(doc)
    feats = [{f"x{j}": float(v) for j, v in enumerate(X[i])} for i in order]
    return feats, order


def main(n_train=600, n_test=200):
    import django
    django.setup()
    import sklearn_crfsuite

    from documents.inference import build_fields
    from ml.evaluate import FieldScores
    from ml.labeling import NUM_TAGS, TAG2ID, assign_labels
    from ml.synth.generator import generate_dataset

    print(f"== training CRF on {n_train} hard docs ==")
    train = [assign_labels(d) for d in generate_dataset(n_train, seed=7, hard=True)]
    test = [assign_labels(d) for d in generate_dataset(n_test, seed=31337, hard=True)]

    Xtr, ytr = [], []
    for d in train:
        feats, order = _seq(d)
        Xtr.append(feats)
        ytr.append([d.tokens[i].tag for i in order])

    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1,
                               max_iterations=80, all_possible_transitions=True)
    crf.fit(Xtr, ytr)

    overall = FieldScores()
    for d in test:
        feats, order = _seq(d)
        marg = crf.predict_marginals_single(feats)  # list of {tag: prob} in reading order
        # map back to doc.tokens order as tags + a (n, NUM_TAGS) prob matrix
        tags = [""] * len(d.tokens)
        probs = np.zeros((len(d.tokens), NUM_TAGS))
        for pos, i in enumerate(order):
            dist = marg[pos]
            best = max(dist, key=dist.get)
            tags[i] = best
            for tag, p in dist.items():
                probs[i, TAG2ID[tag]] = p
        fields = build_fields(d, tags, probs)
        predicted = {f["field"]: f["value"] for f in fields}
        overall.add_document(d.meta["truth"], predicted)

    per = overall.per_field()
    print(f"\n== CRF overall field macro-F1 = {overall.macro_f1():.3f}  (XGBoost 0.844) ==")
    xgb = {"vendor_name": 0.60, "buyer_name": 0.62, "due_date": 0.71,
           "invoice_date": 0.73, "tax_amount": 0.84, "total_amount": 0.86,
           "subtotal": 0.88, "vendor_gstin": 0.93, "buyer_gstin": 0.94,
           "invoice_number": 0.97, "currency": 0.99, "po_number": 1.00}
    for f in sorted(per, key=lambda k: per[k]["f1"]):
        base = xgb.get(f)
        d = f"  vs {base:.2f} ({per[f]['f1'] - base:+.2f})" if base else ""
        print(f"{f:>16}  {per[f]['f1']:.2f}{d}")


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]) if len(a) > 1 else 600, int(a[2]) if len(a) > 2 else 200)

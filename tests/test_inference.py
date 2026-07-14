"""Inference glue: rules-merge (issues #5) and best-span selection (#6).

These exercise pure functions in backend/documents/inference.py; conftest
configures Django so `from django.conf import settings` resolves, but no DB
is touched.
"""

import numpy as np

from ml.labeling import NUM_TAGS, TAG2ID, Token, assign_labels
from ml.synth.generator import generate_document


def _probs_for(tags):
    """One-hot-ish prob matrix giving each tag 0.99 on its own token."""
    p = np.zeros((len(tags), NUM_TAGS))
    for i, t in enumerate(tags):
        p[i, TAG2ID[t]] = 0.99
    return p


def test_build_fields_keeps_highest_confidence_span():
    from documents.inference import build_fields
    from ml.labeling import Document

    # Two spans claim total_amount; the second is more confident and must win.
    toks = [
        Token("100.00", 10, 10, 60, 30),
        Token("999.00", 10, 100, 60, 120),
    ]
    doc = Document("d", toks, 600, 800)
    tags = ["B-total_amount", "B-total_amount"]
    probs = np.zeros((2, NUM_TAGS))
    probs[0, TAG2ID["B-total_amount"]] = 0.55
    probs[1, TAG2ID["B-total_amount"]] = 0.95
    fields = build_fields(doc, tags, probs)
    total = next(f for f in fields if f["field"] == "total_amount")
    assert total["raw"] == "999.00"


def test_merge_rules_fills_missed_field():
    from documents.inference import _merge_rules
    from ml.labeling import Document

    # Champion tags nothing; a GSTIN token should be recovered by rules.
    toks = [Token("GSTIN:", 10, 10, 60, 30),
            Token("27AAPFU0939F1ZV", 70, 10, 260, 30)]
    doc = Document("d", toks, 600, 800)
    tags = ["O", "O"]
    probs = np.zeros((2, NUM_TAGS))
    probs[:, TAG2ID["O"]] = 0.9
    merged, mprobs, engine = _merge_rules(doc, tags, probs, "champion:1")
    assert "+rules" in engine
    assert any(t != "O" for t in merged)


def test_merge_rules_keeps_confident_champion():
    from documents.inference import _merge_rules
    from ml.labeling import Document

    # A confident champion field must not be disturbed by rules.
    toks = [Token("Acme", 10, 10, 60, 30), Token("Ltd", 70, 10, 110, 30)]
    doc = Document("d", toks, 600, 800)
    tags = ["B-vendor_name", "I-vendor_name"]
    probs = np.zeros((2, NUM_TAGS))
    probs[0, TAG2ID["B-vendor_name"]] = 0.97
    probs[1, TAG2ID["I-vendor_name"]] = 0.97
    merged, _, engine = _merge_rules(doc, tags, probs, "champion:1")
    assert merged[0] == "B-vendor_name"
    assert engine == "champion:1"  # rules did not fire

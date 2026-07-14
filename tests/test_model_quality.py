"""Guards for extraction quality: the name-span repair (fixes the weakest
fields) and a model-quality floor so a future change can't silently regress
field-F1."""

import numpy as np
import pytest

from ml.labeling import NUM_TAGS, TAG2ID, Document, Token


def _probs(tags):
    p = np.zeros((len(tags), NUM_TAGS))
    for i, t in enumerate(tags):
        p[i, TAG2ID[t]] = 0.85
    return p


def test_name_repair_recovers_suffix_fragment():
    from documents.inference import build_fields
    # Model tagged only the suffix "Inc." — repair must grow it to the name.
    doc = Document("d", [Token("Soylent", 50, 40, 120, 58),
                         Token("Inc.", 125, 40, 165, 58)], 600, 800)
    tags = ["O", "B-vendor_name"]
    fields = build_fields(doc, tags, _probs(tags))
    v = next(f for f in fields if f["field"] == "vendor_name")
    assert v["raw"] == "Soylent Inc."


def test_name_repair_recovers_generic_word_fragment():
    from documents.inference import build_fields
    doc = Document("d", [Token("Cyberdyne", 50, 40, 140, 58),
                         Token("Systems", 145, 40, 210, 58)], 600, 800)
    tags = ["O", "B-buyer_name"]
    fields = build_fields(doc, tags, _probs(tags))
    v = next(f for f in fields if f["field"] == "buyer_name")
    assert v["raw"] == "Cyberdyne Systems"


def test_name_repair_leaves_full_name_untouched():
    from documents.inference import build_fields
    # A span that already holds a proper name must not be altered.
    doc = Document("d", [Token("Acme", 50, 40, 110, 58),
                         Token("Corporation", 115, 40, 240, 58)], 600, 800)
    tags = ["B-buyer_name", "I-buyer_name"]
    fields = build_fields(doc, tags, _probs(tags))
    v = next(f for f in fields if f["field"] == "buyer_name")
    assert v["raw"] == "Acme Corporation"


def test_name_repair_does_not_cross_column_gap():
    from documents.inference import build_fields
    # "Inc." with a far-away Title-case token (a different column) must NOT
    # be swallowed — the horizontal gap guard prevents it.
    doc = Document("d", [Token("Inc.", 50, 40, 90, 58),
                         Token("INVOICE", 700, 40, 800, 58)], 600, 800)
    tags = ["B-vendor_name", "O"]
    fields = build_fields(doc, tags, _probs(tags))
    v = next(f for f in fields if f["field"] == "vendor_name")
    assert "INVOICE" not in v["raw"]


@pytest.mark.slow
def test_field_f1_floor():
    """Train a small XGBoost and assert a field macro-F1 floor on held-out
    hard invoices — a regression tripwire, not a benchmark."""
    from ml.evaluate import evaluate_field_extraction
    from ml.features import featurize_dataset, featurize_document
    from ml.labeling import ID2TAG, assign_labels
    from ml.models_classical import make_models
    from ml.synth.generator import generate_dataset

    docs = [assign_labels(d) for d in generate_dataset(160, seed=5, hard=True)]
    train, test = docs[40:], docs[:40]
    X, y, _ = featurize_dataset(train)
    model = make_models(fast=True)["xgboost"].fit(X, y)
    tag_lists = [[ID2TAG[int(p)] for p in model.predict(featurize_document(d))]
                 for d in test]
    f1 = evaluate_field_extraction(test, tag_lists).macro_f1()
    assert f1 > 0.70, f"field macro-F1 regressed to {f1:.3f}"

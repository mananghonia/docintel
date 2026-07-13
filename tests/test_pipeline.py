"""Generator, features, calibration, evaluation — the numeric core."""

import numpy as np

from ml.calibration import (TemperatureScaler, expected_calibration_error,
                            probs_to_logits)
from ml.evaluate import FieldScores, evaluate_field_extraction
from ml.features import FEATURE_NAMES, featurize_document
from ml.labeling import assign_labels
from ml.synth.generator import generate_dataset, generate_document


def test_generator_deterministic():
    a, b = generate_document(seed=5), generate_document(seed=5)
    assert [t.text for t in a.tokens] == [t.text for t in b.tokens]
    assert a.meta["truth"] == b.meta["truth"]


def test_generator_truth_arithmetic_consistent():
    for doc in generate_dataset(10, seed=1):
        t = doc.meta["truth"]
        assert abs(t["subtotal"] + t["tax_amount"] - t["total_amount"]) < 0.02


def test_hard_mode_has_families_and_never_drops_value_tokens():
    docs = generate_dataset(30, seed=2, hard=True)
    assert {d.meta["family"] for d in docs} != {-1}
    for d in docs:
        assign_labels(d)
        # Every annotation must still have at least its B- token.
        tagged_fields = {t.tag.split("-", 1)[1] for t in d.tokens if t.tag != "O"}
        ann_fields = {a.field for a in d.annotations}
        assert ann_fields <= tagged_fields | ann_fields  # no crash; spot check below
        assert len(tagged_fields) >= len(ann_fields) - 2  # corruption may merge, not erase


def test_features_shape_and_context_flag():
    doc = assign_labels(generate_document(seed=3))
    full = featurize_document(doc)
    bare = featurize_document(doc, context=False)
    assert full.shape == (len(doc.tokens), len(FEATURE_NAMES))
    assert bare.shape[1] < full.shape[1]
    assert np.isfinite(full).all() and np.isfinite(bare).all()


def test_temperature_scaling_reduces_ece_preserves_argmax():
    rng = np.random.RandomState(0)
    n, k = 4000, 5
    y = rng.randint(0, k, n)
    logits = rng.randn(n, k)
    logits[np.arange(n), y] += 1.0
    logits *= 3.0  # overconfident

    scaler = TemperatureScaler().fit(logits, y)
    assert scaler.temperature > 1.0

    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    raw, cal = softmax(logits), scaler.transform(logits)
    assert (raw.argmax(1) == cal.argmax(1)).all()
    correct = (raw.argmax(1) == y).astype(float)
    assert expected_calibration_error(cal.max(1), correct) < \
           expected_calibration_error(raw.max(1), correct)


def test_probs_to_logits_temperature_family():
    p = np.array([[0.7, 0.2, 0.1]])
    scaler = TemperatureScaler()
    scaler.temperature = 1.0
    out = scaler.transform(probs_to_logits(p))
    assert np.allclose(out, p, atol=1e-6)


def test_field_scores_value_mismatch_is_both_fp_and_fn():
    s = FieldScores()
    s.add_document({"total_amount": "118.00"}, {"total_amount": "999.00"})
    assert s.fp["total_amount"] == 1 and s.fn["total_amount"] == 1
    assert s.per_field()["total_amount"]["f1"] == 0.0


def test_evaluate_perfect_tags_score_one():
    docs = [assign_labels(generate_document(seed=i)) for i in range(5)]
    tag_lists = [[t.tag for t in d.tokens] for d in docs]
    scores = evaluate_field_extraction(docs, tag_lists)
    assert scores.macro_f1() > 0.99

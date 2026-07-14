"""The champion/challenger gate's statistical guard (issue #1) and the
per-document scorer + paired bootstrap it relies on."""

import numpy as np

from ml.evaluate import bootstrap_win_rate, per_document_field_f1
from ml.labeling import assign_labels
from ml.synth.generator import generate_document


def test_per_document_field_f1_length_and_range():
    docs = [assign_labels(generate_document(seed=i)) for i in range(4)]
    perfect = [[t.tag for t in d.tokens] for d in docs]
    scores = per_document_field_f1(docs, perfect)
    assert len(scores) == 4
    assert all(0.99 <= s <= 1.0 for s in scores)


def test_bootstrap_win_rate_clear_winner():
    # Challenger strictly better on every document -> win rate ~1.
    champ = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    chal = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    assert bootstrap_win_rate(chal, champ) > 0.99


def test_bootstrap_win_rate_tie_is_uncertain():
    # Identical scores -> the gate must NOT be confident (this is the noise
    # case that used to promote models on a tie).
    same = [0.8, 0.7, 0.9, 0.85, 0.75]
    assert bootstrap_win_rate(same, same) < 0.6


def test_bootstrap_win_rate_tiny_noisy_holdout_not_confident():
    # 3-doc holdout, challenger better on average but noisy -> not >= 0.9,
    # so the gate correctly refuses to promote.
    champ = [0.6, 0.9, 0.6]
    chal = [0.7, 0.5, 0.9]
    assert bootstrap_win_rate(chal, champ) < 0.9


def test_bootstrap_empty():
    assert bootstrap_win_rate([], []) == 0.0

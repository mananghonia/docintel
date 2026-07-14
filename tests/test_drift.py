"""PSI drift metric used by the monitoring dashboard."""

import numpy as np

from monitoring.drift import psi


def test_psi_zero_for_same_distribution():
    rng = np.random.RandomState(0)
    a = rng.normal(0.9, 0.05, 500)
    b = rng.normal(0.9, 0.05, 500)
    assert psi(a, b) < 0.1  # same distribution -> stable


def test_psi_high_for_shifted_distribution():
    rng = np.random.RandomState(0)
    ref = rng.normal(0.9, 0.05, 500)   # confident model
    cur = rng.normal(0.5, 0.15, 500)   # confidence collapsed (OOD input)
    assert psi(ref, cur) > 0.25        # alert territory


def test_psi_guards_tiny_samples():
    assert psi([0.9, 0.9], [0.5, 0.5]) == 0.0

"""Confidence calibration: temperature scaling, ECE, reliability curves.

Raw softmax/probability outputs are systematically overconfident. The entire
review-routing design (green/amber/red thresholds) assumes confidence means
what it says, so we calibrate.

Temperature scaling: divide logits by a single scalar T fitted on a held-out
validation set. T > 1 softens overconfident distributions. It cannot change
the argmax, so accuracy is untouched — only honesty improves.
"""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray,
                               n_bins: int = 15) -> float:
    """ECE: |accuracy - confidence| averaged over equal-width confidence bins,
    weighted by bin population. 0 = perfectly calibrated."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(confidences)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def reliability_curve(confidences: np.ndarray, correct: np.ndarray,
                      n_bins: int = 15) -> list[dict]:
    """Per-bin stats for plotting a reliability diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        out.append({
            "bin_low": float(lo), "bin_high": float(hi),
            "count": int(mask.sum()),
            "mean_confidence": float(confidences[mask].mean()) if mask.any() else None,
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
        })
    return out


class TemperatureScaler:
    """Fit a single temperature T on validation logits by minimising NLL."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        from scipy.optimize import minimize_scalar

        logits = np.asarray(logits, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=int)

        def nll(t: float) -> float:
            p = _softmax(logits / max(t, 1e-3))
            return float(-np.log(np.clip(p[np.arange(len(y_true)), y_true],
                                         1e-12, 1.0)).mean())

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _softmax(np.asarray(logits, dtype=np.float64) / self.temperature)

    def report(self, logits: np.ndarray, y_true: np.ndarray) -> dict:
        """ECE before/after — the headline calibration numbers."""
        y_true = np.asarray(y_true, dtype=int)
        raw = _softmax(np.asarray(logits, dtype=np.float64))
        cal = self.transform(logits)
        raw_conf, raw_pred = raw.max(axis=1), raw.argmax(axis=1)
        cal_conf = cal.max(axis=1)
        correct = (raw_pred == y_true).astype(float)  # argmax unchanged by T
        return {
            "temperature": self.temperature,
            "ece_before": expected_calibration_error(raw_conf, correct),
            "ece_after": expected_calibration_error(cal_conf, correct),
            "reliability_before": reliability_curve(raw_conf, correct),
            "reliability_after": reliability_curve(cal_conf, correct),
        }


def probs_to_logits(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """For models exposing only predict_proba: log-probs work as logits
    (softmax(log p / T) is a proper temperature family over p)."""
    return np.log(np.clip(probs, eps, 1.0))

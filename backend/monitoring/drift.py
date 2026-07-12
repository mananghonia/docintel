"""Drift detection utilities.

PSI (Population Stability Index) between a reference window and a current
window of any 1-D statistic. Rule of thumb: < 0.1 stable, 0.1-0.25 drifting,
> 0.25 investigate.

The confidence distribution needs no ground truth, so it's the earliest
warning signal: when a new vendor's layout arrives, confidence sags long
before anyone labels a document.
"""

from __future__ import annotations

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) < 10 or len(current) < 10:
        return 0.0
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_frac = np.histogram(current, bins=edges)[0] / len(current)
    ref_frac = np.clip(ref_frac, 1e-4, None)
    cur_frac = np.clip(cur_frac, 1e-4, None)
    return float(((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)).sum())

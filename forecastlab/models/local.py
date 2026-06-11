"""Per-series ("local") forecasters for intermittent demand.

All produce a flat horizon-h forecast (the per-period expected demand), which is
the natural output of these methods: Croston-family models estimate a constant
demand *rate* until the next update. For intermittent demand a flat mean over
the lead time is exactly what feeds a min/max or safety-stock calculation.

Implemented from first principles (no statsforecast dependency) so the logic is
inspectable in the app and the package stays light.
"""

from __future__ import annotations

import numpy as np


def naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value. The honest floor every method must beat."""
    last = float(history[-1]) if len(history) else 0.0
    return np.full(horizon, last)


def moving_average(history: np.ndarray, horizon: int, window: int = 8) -> np.ndarray:
    """Mean of the trailing ``window`` periods (zeros included)."""
    if len(history) == 0:
        return np.zeros(horizon)
    w = history[-window:]
    return np.full(horizon, float(w.mean()))


def ses(history: np.ndarray, horizon: int, alpha: float = 0.2) -> np.ndarray:
    """Simple exponential smoothing — the conventional smooth-quadrant method."""
    if len(history) == 0:
        return np.zeros(horizon)
    level = float(history[0])
    for x in history[1:]:
        level = alpha * float(x) + (1 - alpha) * level
    return np.full(horizon, level)


def _croston_core(history: np.ndarray, alpha: float, kind: str) -> float:
    """Shared Croston / SBA / TSB recursion, returning the per-period rate.

    kind:
      'croston' — classic Croston (1972): smooth demand size z and interval p
                  on demand epochs; forecast = z / p.
      'sba'     — Syntetos-Boylan Approximation: multiply by (1 - alpha/2) to
                  remove Croston's known positive bias.
      'tsb'     — Teunter-Syntetos-Babai: smooth demand *probability* every
                  period (including zeros), so obsolescent parts decay to zero.
    """
    history = np.asarray(history, dtype=float)
    nz_idx = np.nonzero(history > 0)[0]
    if nz_idx.size == 0:
        return 0.0

    if kind == "tsb":
        # Smooth probability and size; probability updates every period.
        prob = (history > 0).mean()
        size = history[history > 0].mean()
        alpha_p, alpha_z = 0.1, alpha
        for x in history:
            if x > 0:
                prob = prob + alpha_p * (1.0 - prob)
                size = size + alpha_z * (float(x) - size)
            else:
                prob = prob + alpha_p * (0.0 - prob)
        return float(prob * size)

    # Croston / SBA: update size z and inter-arrival interval p on demand epochs.
    z = float(history[nz_idx[0]])
    intervals = np.diff(np.concatenate([[-1], nz_idx]))  # gap since prior demand
    p = float(intervals[0]) if intervals[0] > 0 else 1.0
    for k in range(1, nz_idx.size):
        z = z + alpha * (float(history[nz_idx[k]]) - z)
        gap = float(nz_idx[k] - nz_idx[k - 1])
        p = p + alpha * (gap - p)
    rate = z / p if p > 0 else 0.0
    if kind == "sba":
        rate *= 1.0 - alpha / 2.0
    return float(rate)


def croston(history: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    return np.full(horizon, _croston_core(history, alpha, "croston"))


def sba(history: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    return np.full(horizon, _croston_core(history, alpha, "sba"))


def tsb(history: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    return np.full(horizon, _croston_core(history, alpha, "tsb"))


LOCAL_MODELS = {
    "naive": naive,
    "moving_avg": moving_average,
    "ses": ses,
    "croston": croston,
    "sba": sba,
    "tsb": tsb,
}

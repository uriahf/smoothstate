"""Binary state-probability smoothing."""

from __future__ import annotations

import numpy as np
import polars as pl


def _tricube(u: np.ndarray) -> np.ndarray:
    """Tricube kernel used by LOWESS-style local regression."""
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) < 1.0
    z = 1.0 - np.abs(u[mask]) ** 3
    out[mask] = z**3
    return out


def smooth_binary_state(
    probs: np.ndarray | pl.Series,
    states: np.ndarray | pl.Series,
    *,
    grid: np.ndarray | None = None,
    frac: float = 2 / 3,
) -> pl.DataFrame:
    """Smooth ``P(state=1 | prob)`` with local linear regression."""
    x = np.asarray(probs, dtype=float)
    y = np.asarray(states, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("probs and states must be one-dimensional")
    if len(x) != len(y):
        raise ValueError("probs and states must have equal length")
    if len(x) == 0:
        raise ValueError("probs and states must not be empty")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("probs and states must contain only finite values")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("probs must be between 0 and 1")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("states must contain only 0 and 1")
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")

    xout = np.linspace(0.0, 1.0, 101) if grid is None else np.asarray(grid, dtype=float)
    if xout.ndim != 1:
        raise ValueError("grid must be one-dimensional")

    if np.all(x == x[0]):
        return pl.DataFrame({"x": xout, "y": np.full_like(xout, y.mean(), dtype=float)})

    n_neighbors = max(2, int(np.ceil(frac * len(x))))
    fitted = np.empty_like(xout, dtype=float)

    for i, x0 in enumerate(xout):
        distances = np.abs(x - x0)
        bandwidth = np.partition(distances, min(n_neighbors - 1, len(x) - 1))[n_neighbors - 1]

        if bandwidth <= 0:
            local = distances == 0
            fitted[i] = y[local].mean()
            continue

        weights = _tricube(distances / bandwidth)
        centered = x - x0
        s0 = weights.sum()
        s1 = np.sum(weights * centered)
        s2 = np.sum(weights * centered * centered)
        t0 = np.sum(weights * y)
        t1 = np.sum(weights * centered * y)
        det = s0 * s2 - s1 * s1

        if det <= np.finfo(float).eps:
            fitted[i] = t0 / s0 if s0 > 0 else y.mean()
        else:
            fitted[i] = (t0 * s2 - t1 * s1) / det

    fitted = np.clip(fitted, 0.0, 1.0)
    return pl.DataFrame({"x": xout, "y": fitted})

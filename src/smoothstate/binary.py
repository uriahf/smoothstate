"""Binary state-probability smoothing."""

from __future__ import annotations

import numpy as np
import polars as pl


def _tricube(u: np.ndarray) -> np.ndarray:
    """Tricube kernel used by LOWESS local regression."""
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) < 1.0
    z = 1.0 - np.abs(u[mask]) ** 3
    out[mask] = z**3
    return out


def _bisquare(residuals: np.ndarray) -> np.ndarray:
    """Robust residual weights used by Cleveland LOWESS."""
    median_abs_residual = np.median(np.abs(residuals))
    if median_abs_residual <= np.finfo(float).eps:
        return np.ones_like(residuals, dtype=float)
    u = residuals / (6.0 * median_abs_residual)
    weights = np.zeros_like(u, dtype=float)
    mask = np.abs(u) < 1.0
    weights[mask] = (1.0 - u[mask] ** 2) ** 2
    return weights


def _local_linear(
    x: np.ndarray,
    y: np.ndarray,
    xout: np.ndarray,
    *,
    frac: float,
    residual_weights: np.ndarray,
) -> np.ndarray:
    n_neighbors = max(2, int(np.ceil(frac * len(x))))
    fitted = np.empty_like(xout, dtype=float)

    for i, x0 in enumerate(xout):
        distances = np.abs(x - x0)
        bandwidth = np.partition(distances, n_neighbors - 1)[n_neighbors - 1]

        if bandwidth <= 0:
            local = distances == 0
            local_weights = residual_weights[local]
            if local_weights.sum() > 0:
                fitted[i] = np.average(y[local], weights=local_weights)
            else:
                fitted[i] = y[local].mean()
            continue

        weights = _tricube(distances / bandwidth) * residual_weights
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

    return fitted


def smooth_binary_state(
    probs: np.ndarray | pl.Series,
    states: np.ndarray | pl.Series,
    *,
    grid: np.ndarray | None = None,
    frac: float = 2 / 3,
    it: int = 3,
) -> pl.DataFrame:
    """Smooth ``P(state=1 | prob)`` using Cleveland LOWESS.

    ``it`` controls residual-based robust reweighting and defaults to 3,
    matching ``statsmodels.nonparametric.lowess``.
    """
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
    if not isinstance(it, int) or it < 0:
        raise ValueError("it must be a non-negative integer")

    xout = np.linspace(0.0, 1.0, 101) if grid is None else np.asarray(grid, dtype=float)
    if xout.ndim != 1:
        raise ValueError("grid must be one-dimensional")

    if np.all(x == x[0]):
        return pl.DataFrame({"x": xout, "y": np.full_like(xout, y.mean(), dtype=float)})

    residual_weights = np.ones_like(y, dtype=float)
    for _ in range(it):
        fitted_at_data = _local_linear(
            x, y, x, frac=frac, residual_weights=residual_weights
        )
        residual_weights = _bisquare(y - fitted_at_data)

    fitted = _local_linear(
        x, y, xout, frac=frac, residual_weights=residual_weights
    )
    fitted = np.clip(fitted, 0.0, 1.0)
    return pl.DataFrame({"x": xout, "y": fitted})

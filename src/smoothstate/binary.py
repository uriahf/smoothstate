"""LOWESS smoothing for state values over predicted probabilities."""

from __future__ import annotations

import numpy as np
import polars as pl


def _tricube(u: np.ndarray) -> np.ndarray:
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) < 1.0
    z = 1.0 - np.abs(u[mask]) ** 3
    out[mask] = z**3
    return out


def _local_linear_at_data(x: np.ndarray, y: np.ndarray, *, frac: float) -> np.ndarray:
    """Fit one non-robust Cleveland LOWESS pass at the observed x values."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]
    n = len(xs)
    n_neighbors = max(2, min(n, int(frac * n)))
    fitted = np.empty(n, dtype=float)

    for i, x0 in enumerate(xs):
        distances = np.abs(xs - x0)
        bandwidth = np.partition(distances, n_neighbors - 1)[n_neighbors - 1]
        if bandwidth <= 0:
            local = distances == 0
            fitted[i] = ys[local].mean()
            continue

        weights = _tricube(distances / bandwidth)
        centered = xs - x0
        s0 = weights.sum()
        s1 = np.sum(weights * centered)
        s2 = np.sum(weights * centered * centered)
        t0 = np.sum(weights * ys)
        t1 = np.sum(weights * centered * ys)
        det = s0 * s2 - s1 * s1

        if det <= np.finfo(float).eps:
            fitted[i] = t0 / s0 if s0 > 0 else ys.mean()
        else:
            fitted[i] = (t0 * s2 - t1 * s1) / det

    return np.column_stack([xs, fitted])


def smooth_state_lowess(
    probs: np.ndarray | pl.Series,
    values: np.ndarray | pl.Series,
    *,
    grid: np.ndarray | None = None,
    frac: float = 2 / 3,
) -> pl.DataFrame:
    """Smooth state values with the LOWESS pipeline used by rtichoke.

    This matches rtichoke's historical statsmodels call: fit LOWESS with
    ``it=0`` at the observed probabilities, linearly interpolate the fitted
    values onto the requested grid, then clip the result to ``[0, 1]``.
    """
    x = np.asarray(probs, dtype=float)
    y = np.asarray(values, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("probs and values must be one-dimensional")
    if len(x) != len(y):
        raise ValueError("probs and values must have equal length")
    if len(x) == 0:
        raise ValueError("probs and values must not be empty")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("probs and values must contain only finite values")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("probs must be between 0 and 1")
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")

    xout = np.linspace(0.0, 1.0, 101) if grid is None else np.asarray(grid, dtype=float)
    if xout.ndim != 1:
        raise ValueError("grid must be one-dimensional")

    if np.all(x == x[0]):
        fitted = np.full_like(xout, y.mean(), dtype=float)
        return pl.DataFrame({"x": xout, "y": np.clip(fitted, 0.0, 1.0)})

    smoothed = _local_linear_at_data(x, y, frac=frac)
    fitted = np.interp(xout, smoothed[:, 0], smoothed[:, 1])
    fitted = np.clip(fitted, 0.0, 1.0)
    return pl.DataFrame({"x": xout, "y": fitted})


def smooth_binary_state(
    probs: np.ndarray | pl.Series,
    states: np.ndarray | pl.Series,
    *,
    grid: np.ndarray | None = None,
    frac: float = 2 / 3,
) -> pl.DataFrame:
    """Smooth ``P(state=1 | prob)`` using rtichoke-compatible LOWESS."""
    y = np.asarray(states, dtype=float)
    if np.any((y != 0) & (y != 1)):
        raise ValueError("states must contain only 0 and 1")
    return smooth_state_lowess(probs, y, grid=grid, frac=frac)

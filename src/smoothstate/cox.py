"""Secondary Cox smoothing for time-dependent state probabilities."""

from __future__ import annotations

import numpy as np
import polars as pl


def _rcs_basis_3knots(
    x: np.ndarray, knots: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return Harrell-style 3-knot restricted cubic spline basis."""
    x = np.asarray(x, dtype=float)
    if knots is None:
        knots = np.percentile(x, [10, 50, 90])
    knots = np.sort(np.asarray(knots, dtype=float))
    t1, t2, t3 = knots

    if len(np.unique(knots)) < 3:
        return x[:, None], knots

    def pos_cube(z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0) ** 3

    u1 = (
        pos_cube(x - t1)
        - ((t3 - t1) / (t3 - t2)) * pos_cube(x - t2)
        + ((t2 - t1) / (t3 - t2)) * pos_cube(x - t3)
    ) / ((t3 - t1) ** 2)
    return np.column_stack([x, u1]), knots


def _cox_score_info_efron(
    beta: np.ndarray,
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    penalizer: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score and observed information for a Cox model with Efron ties."""
    eta = np.clip(x @ beta, -50.0, 50.0)
    risk = np.exp(eta)
    p = x.shape[1]
    score = -penalizer * beta
    info = np.eye(p) * penalizer

    for t in np.unique(time[event == 1]):
        deaths = (time == t) & (event == 1)
        at_risk = time >= t
        d = int(deaths.sum())
        if d == 0:
            continue

        r = risk[at_risk]
        xr = x[at_risk]
        s0 = r.sum()
        s1 = (r[:, None] * xr).sum(axis=0)
        s2 = np.einsum("i,ij,ik->jk", r, xr, xr)

        rd = risk[deaths]
        xd = x[deaths]
        d0 = rd.sum()
        d1 = (rd[:, None] * xd).sum(axis=0)
        d2 = np.einsum("i,ij,ik->jk", rd, xd, xd)

        score += xd.sum(axis=0)
        for l in range(d):
            frac = l / d
            den = s0 - frac * d0
            mean = (s1 - frac * d1) / den
            second = (s2 - frac * d2) / den
            score -= mean
            info += second - np.outer(mean, mean)

    return score, info


def _fit_cox_efron(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    penalizer: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Fit a small Cox PH model with Newton-Raphson."""
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(max_iter):
        score, info = _cox_score_info_efron(beta, x, time, event, penalizer)
        step = np.linalg.solve(info, score)
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            return beta_new
        beta = beta_new
    return beta


def _breslow_baseline_cumulative_hazard(
    beta: np.ndarray,
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
) -> float:
    """Estimate Breslow baseline cumulative hazard through ``horizon``."""
    risk = np.exp(np.clip(x @ beta, -50.0, 50.0))
    h0 = 0.0
    for t in np.unique(time[(event == 1) & (time <= horizon)]):
        d = int(((time == t) & (event == 1)).sum())
        den = risk[time >= t].sum()
        if den > 0:
            h0 += d / den
    return float(h0)


def smooth_state_cox(
    probs: np.ndarray | pl.Series,
    times: np.ndarray | pl.Series,
    events: np.ndarray | pl.Series,
    horizon: float,
    *,
    grid: np.ndarray | None = None,
    penalizer: float = 0.01,
) -> pl.DataFrame:
    """Smooth event-state probability using secondary Cox + 3-knot RCS.

    This mirrors the secondary Cox smoother used by ``rtichoke``: predicted
    probabilities are complementary-log-log transformed, expanded with a
    3-knot restricted cubic spline, and used as the sole predictors in a Cox
    proportional hazards model.
    """
    p = np.asarray(probs, dtype=float)
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)

    if p.ndim != 1 or t.ndim != 1 or e.ndim != 1:
        raise ValueError("probs, times, and events must be one-dimensional")
    if not (len(p) == len(t) == len(e)) or len(p) == 0:
        raise ValueError("probs, times, and events must have equal non-zero length")
    if np.any((e != 0) & (e != 1)):
        raise ValueError("events must contain only 0 and 1")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probs must be between 0 and 1")

    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(-np.log(1 - p_clip))
    xout = np.linspace(0.001, 0.999, 101) if grid is None else np.asarray(grid, dtype=float)

    if len(np.unique(x)) <= 1 or e.sum() == 0:
        risk = float(np.mean(e[t <= horizon])) if np.any(t <= horizon) else 0.0
        return pl.DataFrame({"x": xout, "y": np.full(len(xout), risk)})

    basis, knots = _rcs_basis_3knots(x)
    beta = _fit_cox_efron(basis, t, e, penalizer=penalizer)
    h0 = _breslow_baseline_cumulative_hazard(beta, basis, t, e, horizon)

    x_grid = np.log(-np.log(1 - np.clip(xout, 1e-6, 1 - 1e-6)))
    grid_basis, _ = _rcs_basis_3knots(x_grid, knots=knots)
    relative_risk = np.exp(np.clip(grid_basis @ beta, -50.0, 50.0))
    yout = np.clip(1.0 - np.exp(-h0 * relative_risk), 0.0, 1.0)
    return pl.DataFrame({"x": xout, "y": yout})

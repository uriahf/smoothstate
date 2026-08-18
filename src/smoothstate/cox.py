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
    """Score and observed information for a Cox model with Efron ties.

    Risk-set moments are accumulated once while traversing observations from
    largest to smallest time. This avoids rebuilding each risk set from the
    full sample and keeps each Newton iteration approximately O(n).
    """
    eta = np.clip(x @ beta, -50.0, 50.0)
    risk = np.exp(eta)
    p = x.shape[1]
    score = -penalizer * beta
    info = np.eye(p) * penalizer

    order = np.argsort(-time, kind="stable")
    time_s = time[order]
    event_s = event[order]
    x_s = x[order]
    risk_s = risk[order]

    s0 = 0.0
    s1 = np.zeros(p, dtype=float)
    s2 = np.zeros((p, p), dtype=float)

    start = 0
    n = len(time_s)
    while start < n:
        end = start + 1
        current_time = time_s[start]
        while end < n and time_s[end] == current_time:
            end += 1

        x_group = x_s[start:end]
        r_group = risk_s[start:end]
        e_group = event_s[start:end]

        s0 += float(r_group.sum())
        s1 += (r_group[:, None] * x_group).sum(axis=0)
        s2 += np.einsum("i,ij,ik->jk", r_group, x_group, x_group)

        deaths = e_group == 1
        d = int(deaths.sum())
        if d:
            xd = x_group[deaths]
            rd = r_group[deaths]
            d0 = float(rd.sum())
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

        start = end

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
    """Estimate Breslow baseline cumulative hazard through ``horizon`` in O(n)."""
    risk = np.exp(np.clip(x @ beta, -50.0, 50.0))
    order = np.argsort(-time, kind="stable")
    time_s = time[order]
    event_s = event[order]
    risk_s = risk[order]

    cumulative_risk = 0.0
    h0 = 0.0
    start = 0
    n = len(time_s)
    while start < n:
        end = start + 1
        current_time = time_s[start]
        while end < n and time_s[end] == current_time:
            end += 1

        cumulative_risk += float(risk_s[start:end].sum())
        if current_time <= horizon:
            d = int(event_s[start:end].sum())
            if d and cumulative_risk > 0:
                h0 += d / cumulative_risk
        start = end

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

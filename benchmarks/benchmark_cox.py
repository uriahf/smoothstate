"""Benchmark smoothstate secondary Cox smoothing against lifelines."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from smoothstate.cox import _rcs_basis_3knots, smooth_state_cox


def simulate(n: int, seed: int = 20260818) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + n)
    probs = rng.uniform(0.02, 0.75, size=n)
    x = np.log(-np.log(1 - probs))
    event_time = rng.exponential(scale=np.exp(-0.5 * x) * 8.0)
    censor_time = rng.exponential(scale=12.0, size=n)
    times = np.minimum(event_time, censor_time)
    events = (event_time <= censor_time).astype(int)
    return probs, times, events


def lifelines_curve(
    probs: np.ndarray, times: np.ndarray, events: np.ndarray, horizon: float
) -> np.ndarray:
    x = np.log(-np.log(1 - np.clip(probs, 1e-6, 1 - 1e-6)))
    basis, knots = _rcs_basis_3knots(x)
    fit_df = pd.DataFrame({"time": times, "event": events, "rcs_1": basis[:, 0]})
    if basis.shape[1] == 2:
        fit_df["rcs_2"] = basis[:, 1]

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(fit_df, duration_col="time", event_col="event")

    grid = np.linspace(0.001, 0.999, 101)
    x_grid = np.log(-np.log(1 - grid))
    grid_basis, _ = _rcs_basis_3knots(x_grid, knots=knots)
    grid_df = pd.DataFrame({"rcs_1": grid_basis[:, 0]})
    if grid_basis.shape[1] == 2:
        grid_df["rcs_2"] = grid_basis[:, 1]
    return 1.0 - cph.predict_survival_function(grid_df, times=[horizon]).values.ravel()


def timed(fn, repeats: int) -> tuple[float, np.ndarray]:
    durations = []
    out = None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        durations.append(time.perf_counter() - start)
    assert out is not None
    return min(durations), np.asarray(out)


def main() -> None:
    horizon = 5.0
    print("n,smoothstate_s,lifelines_s,speedup,max_abs_diff,mean_abs_diff")
    for n in (1_000, 10_000, 100_000):
        probs, times, events = simulate(n)
        repeats = 3 if n < 100_000 else 1
        smooth_t, smooth_curve = timed(
            lambda: smooth_state_cox(probs, times, events, horizon)["y"].to_numpy(),
            repeats=repeats,
        )
        life_t, life_curve = timed(
            lambda: lifelines_curve(probs, times, events, horizon), repeats=repeats
        )
        diff = np.abs(smooth_curve - life_curve)
        print(
            f"{n},{smooth_t:.6f},{life_t:.6f},{life_t / smooth_t:.2f},"
            f"{diff.max():.6g},{diff.mean():.6g}"
        )


if __name__ == "__main__":
    main()

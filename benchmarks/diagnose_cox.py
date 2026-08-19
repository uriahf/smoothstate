"""Diagnose differences between smoothstate and lifelines Cox fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from smoothstate.cox import (
    _breslow_baseline_cumulative_hazard,
    _fit_cox_efron,
    _rcs_basis_3knots,
)


def simulate(n: int, seed: int = 20260818) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + n)
    probs = rng.uniform(0.02, 0.75, size=n)
    x = np.log(-np.log(1 - probs))
    event_time = rng.exponential(scale=np.exp(-0.5 * x) * 8.0)
    censor_time = rng.exponential(scale=12.0, size=n)
    times = np.minimum(event_time, censor_time)
    events = (event_time <= censor_time).astype(int)
    return probs, times, events


def main() -> None:
    horizon = 5.0
    for n in (1_000, 10_000):
        probs, times, events = simulate(n)
        x = np.log(-np.log(1 - np.clip(probs, 1e-6, 1 - 1e-6)))
        basis, _ = _rcs_basis_3knots(x)

        beta = _fit_cox_efron(basis, times, events, penalizer=0.01)
        h0 = _breslow_baseline_cumulative_hazard(beta, basis, times, events, horizon)

        fit_df = pd.DataFrame({"time": times, "event": events, "rcs_1": basis[:, 0]})
        if basis.shape[1] == 2:
            fit_df["rcs_2"] = basis[:, 1]

        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(fit_df, duration_col="time", event_col="event")

        life_beta = cph.params_.to_numpy()
        means = cph._norm_mean.to_numpy()
        stds = cph._norm_std.to_numpy()
        life_h0 = float(cph.baseline_cumulative_hazard_.loc[:horizon].iloc[-1, 0])

        print(f"n={n}")
        print("smooth_beta", beta)
        print("lifelines_beta", life_beta)
        print("beta_diff", beta - life_beta)
        print("lifelines_norm_mean", means)
        print("lifelines_norm_std", stds)
        print("smooth_h0", h0)
        print("lifelines_h0", life_h0)
        print("h0_diff", h0 - life_h0)
        print()


if __name__ == "__main__":
    main()

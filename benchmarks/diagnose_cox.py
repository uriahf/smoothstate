"""Diagnose differences between smoothstate and lifelines Cox fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from benchmarks.benchmark_cox import simulate
from smoothstate.cox import (
    _breslow_baseline_cumulative_hazard,
    _fit_cox_efron,
    _rcs_basis_3knots,
)


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

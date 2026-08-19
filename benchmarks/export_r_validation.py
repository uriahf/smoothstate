"""Create deterministic cross-language Cox/RCS validation fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smoothstate.cox import (
    _breslow_baseline_cumulative_hazard,
    _fit_cox_efron,
    _rcs_basis_3knots,
)


def main() -> None:
    out = Path("benchmark-results")
    out.mkdir(exist_ok=True)

    rng = np.random.default_rng(20260818)
    n = 2500
    probs = rng.uniform(0.02, 0.75, n)
    x = np.log(-np.log(1 - probs))
    event_time = rng.exponential(scale=np.exp(-0.55 * x) * 8.0)
    censor_time = rng.exponential(scale=12.0, size=n)
    times = np.minimum(event_time, censor_time)
    events = (event_time <= censor_time).astype(int)
    observed_event_times = np.sort(times[events == 1])
    horizon = float(observed_event_times[len(observed_event_times) // 2])

    basis, knots = _rcs_basis_3knots(x)
    beta = _fit_cox_efron(basis, times, events, penalizer=0.0)
    h0 = _breslow_baseline_cumulative_hazard(beta, basis, times, events, horizon)

    grid = np.linspace(0.01, 0.90, 101)
    x_grid = np.log(-np.log(1 - grid))
    grid_basis, _ = _rcs_basis_3knots(x_grid, knots=knots)
    pred = 1 - np.exp(-h0 * np.exp(grid_basis @ beta))

    pd.DataFrame(
        {
            "probs": probs,
            "time": times,
            "event": events,
            "x": x,
            "rcs_1": basis[:, 0],
            "rcs_2": basis[:, 1],
        }
    ).to_csv(out / "r-validation-input.csv", index=False)

    pd.DataFrame(
        {
            "probs": grid,
            "x": x_grid,
            "rcs_1": grid_basis[:, 0],
            "rcs_2": grid_basis[:, 1],
            "smoothstate_risk": pred,
        }
    ).to_csv(out / "r-validation-grid.csv", index=False)

    pd.DataFrame(
        {
            "quantity": ["beta_1", "beta_2", "h0", "knot_1", "knot_2", "knot_3", "horizon"],
            "value": [beta[0], beta[1], h0, knots[0], knots[1], knots[2], horizon],
        }
    ).to_csv(out / "r-validation-reference.csv", index=False)


if __name__ == "__main__":
    main()

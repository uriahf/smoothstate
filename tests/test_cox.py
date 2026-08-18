import numpy as np

from smoothstate.cox import _rcs_basis_3knots, smooth_state_cox


def test_rcs_basis_has_expected_shape():
    x = np.linspace(-3, 2, 100)
    basis, knots = _rcs_basis_3knots(x)
    assert basis.shape == (100, 2)
    np.testing.assert_allclose(knots, np.percentile(x, [10, 50, 90]))


def test_secondary_cox_returns_probabilities():
    rng = np.random.default_rng(42)
    n = 300
    probs = rng.uniform(0.03, 0.8, n)
    times = rng.exponential(4.0, n)
    events = rng.binomial(1, 0.7, n)

    out = smooth_state_cox(probs, times, events, horizon=3.0)

    assert out.columns == ["x", "y"]
    assert out.height == 101
    assert out["y"].is_between(0.0, 1.0, closed="both").all()


def test_secondary_cox_tracks_lifelines():
    """Numerical regression test against the implementation rtichoke uses."""
    import pandas as pd
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(7)
    n = 500
    probs = rng.beta(2.0, 5.0, n)
    x = np.log(-np.log(1 - np.clip(probs, 1e-6, 1 - 1e-6)))
    basis, knots = _rcs_basis_3knots(x)

    linpred = 0.45 * basis[:, 0] - 0.12 * basis[:, 1]
    event_time = rng.exponential(scale=np.exp(-linpred) * 5.0)
    censor_time = rng.exponential(8.0, n)
    times = np.minimum(event_time, censor_time)
    events = (event_time <= censor_time).astype(int)
    horizon = 3.0

    ours = smooth_state_cox(probs, times, events, horizon=horizon)

    fit = pd.DataFrame(
        {"time": times, "event": events, "rcs_1": basis[:, 0], "rcs_2": basis[:, 1]}
    )
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(fit, duration_col="time", event_col="event")

    grid = ours["x"].to_numpy()
    x_grid = np.log(-np.log(1 - grid))
    grid_basis, _ = _rcs_basis_3knots(x_grid, knots=knots)
    grid_df = pd.DataFrame({"rcs_1": grid_basis[:, 0], "rcs_2": grid_basis[:, 1]})
    expected = 1.0 - cph.predict_survival_function(grid_df, times=[horizon]).values.ravel()

    # The custom solver should reproduce the same curve closely enough for
    # calibration plotting. Tighten this as penalty conventions are aligned.
    np.testing.assert_allclose(ours["y"].to_numpy(), expected, atol=0.03, rtol=0.08)

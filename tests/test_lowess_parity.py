import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from smoothstate import smooth_binary_state, smooth_state_lowess


def _rtichoke_statsmodels_pipeline(x, y, grid, frac=2 / 3):
    smoothed = lowess(y, x, frac=frac, it=0)
    return np.clip(np.interp(grid, smoothed[:, 0], smoothed[:, 1]), 0.0, 1.0)


def test_binary_lowess_matches_rtichoke_statsmodels_pipeline():
    x = np.linspace(0.05, 0.95, 30)
    y = (x > 0.45).astype(float)
    grid = np.linspace(0.0, 1.0, 101)

    actual = smooth_binary_state(x, y, grid=grid, frac=2 / 3)["y"].to_numpy()
    expected = _rtichoke_statsmodels_pipeline(x, y, grid, frac=2 / 3)

    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_generic_lowess_matches_rtichoke_for_continuous_pseudo_values():
    x = np.linspace(0.04, 0.96, 40)
    y = np.clip(x + 0.18 * np.sin(5 * np.pi * x), -0.2, 1.2)
    grid = np.linspace(0.0, 1.0, 101)

    actual = smooth_state_lowess(x, y, grid=grid, frac=2 / 3)["y"].to_numpy()
    expected = _rtichoke_statsmodels_pipeline(x, y, grid, frac=2 / 3)

    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_generic_lowess_handles_unsorted_probabilities():
    rng = np.random.default_rng(42)
    x = rng.uniform(0.02, 0.98, 60)
    y = rng.normal(loc=x, scale=0.1)
    grid = np.linspace(0.0, 1.0, 101)

    actual = smooth_state_lowess(x, y, grid=grid, frac=0.5)["y"].to_numpy()
    expected = _rtichoke_statsmodels_pipeline(x, y, grid, frac=0.5)

    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

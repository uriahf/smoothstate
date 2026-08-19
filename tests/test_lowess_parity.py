import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from smoothstate import smooth_binary_state


def _statsmodels_lowess(x, y, grid, frac=2 / 3, it=3):
    return lowess(y, x, frac=frac, it=it, xvals=grid)


def test_lowess_matches_statsmodels_default_robust_fit():
    x = np.linspace(0.02, 0.98, 60)
    y = (x + 0.12 * np.sin(8 * np.pi * x) > 0.5).astype(float)
    grid = np.linspace(0.0, 1.0, 101)

    actual = smooth_binary_state(x, y, grid=grid, frac=2 / 3, it=3)["y"].to_numpy()
    expected = _statsmodels_lowess(x, y, grid, frac=2 / 3, it=3)

    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_lowess_matches_statsmodels_with_outliers():
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0.01, 0.99, 80))
    y = (rng.uniform(size=80) < x).astype(float)
    y[[5, 21, 58]] = 1 - y[[5, 21, 58]]
    grid = np.linspace(0.0, 1.0, 101)

    actual = smooth_binary_state(x, y, grid=grid, frac=0.5, it=3)["y"].to_numpy()
    expected = _statsmodels_lowess(x, y, grid, frac=0.5, it=3)

    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_lowess_it_zero_matches_statsmodels():
    x = np.linspace(0.05, 0.95, 30)
    y = (x > 0.45).astype(float)
    grid = np.linspace(0.0, 1.0, 101)

    actual = smooth_binary_state(x, y, grid=grid, frac=2 / 3, it=0)["y"].to_numpy()
    expected = _statsmodels_lowess(x, y, grid, frac=2 / 3, it=0)

    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

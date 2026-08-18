import numpy as np
import polars as pl

from smoothstate import smooth_binary_state


def test_returns_polars_curve():
    probs = np.linspace(0.05, 0.95, 20)
    states = (probs > 0.5).astype(int)

    result = smooth_binary_state(probs, states)

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["x", "y"]
    assert result.height == 101
    assert result["y"].min() >= 0
    assert result["y"].max() <= 1


def test_constant_prediction_returns_mean_state():
    probs = np.repeat(0.4, 5)
    states = np.array([0, 1, 0, 1, 1])

    result = smooth_binary_state(probs, states, grid=np.array([0.4]))

    assert np.isclose(result["y"][0], states.mean())


def test_rejects_nonbinary_states():
    probs = np.array([0.1, 0.2, 0.3])
    states = np.array([0, 2, 1])

    try:
        smooth_binary_state(probs, states)
    except ValueError as exc:
        assert "0 and 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

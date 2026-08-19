# smoothstate

Fast, dependency-light smoothing of state probabilities for Python, built for Polars.

`smoothstate` provides focused smoothing primitives for model-evaluation workflows. It was created to support the smooth calibration machinery in [`rtichoke`](https://github.com/uriahf/rtichoke_python) without requiring a general-purpose survival-modeling stack at runtime.

The package currently includes:

- `smooth_binary_state()` for smoothing binary state probabilities over a continuous predictor.
- `smooth_state_cox()` for time-dependent smoothing with a complementary log-log transformed probability, a 3-knot restricted cubic spline, and a secondary Cox model.

The Cox implementation is continuously validated against Python `lifelines` and R `survival::coxph()`. In the current GitHub Actions benchmark it reproduces the reference curves to numerical precision while running roughly 26–39× faster than `lifelines` for this deliberately narrow workload.

## Installation

```bash
uv add smoothstate
```

Until the first PyPI release is published, install directly from GitHub:

```bash
uv add git+https://github.com/uriahf/smoothstate
```

## Documentation

See the [Great Docs site](https://uriahf.github.io/smoothstate/) for the user guide, implementation details, benchmarks, and API reference.

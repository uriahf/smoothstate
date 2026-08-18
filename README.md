# smoothstate

Smooth state-occupation probabilities over a continuous predictor.

`smoothstate` is a small Polars-first Python package for smoothing state probabilities, designed initially to support calibration workflows in `rtichoke`.

The current prototype includes binary smoothing and a dependency-light secondary Cox + 3-knot restricted cubic spline smoother for time-dependent calibration.

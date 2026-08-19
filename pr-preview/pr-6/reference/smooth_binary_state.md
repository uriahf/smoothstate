## smooth_binary_state()


Smooth `P(state=1 | prob)` using Cleveland LOWESS.


Usage

``` python
smooth_binary_state(
    probs,
    states,
    *,
    grid=None,
    frac=2 / 3,
    it=3,
)
```


`it` controls residual-based robust reweighting and defaults to 3, matching `statsmodels.nonparametric.lowess`.

## smooth_binary_state()


Smooth `P(state=1 | prob)` using rtichoke-compatible LOWESS.


Usage

``` python
smooth_binary_state(
    probs,
    states,
    *,
    grid=None,
    frac=2 / 3,
)
```

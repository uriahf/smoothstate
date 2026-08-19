## smooth_binary_state()


Smooth `P(state=1 | prob)` with local linear regression.


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

# API Reference


Polars-first smoothing primitives for binary and time-dependent state probabilities.


## Binary state smoothing


[smooth_binary_state()](smooth_binary_state.md#smoothstate.smooth_binary_state)  
Smooth `P(state=1 | prob)` using rtichoke-compatible LOWESS.


## Time-dependent state smoothing


[smooth_state_cox()](smooth_state_cox.md#smoothstate.smooth_state_cox)  
Smooth event-state probability using secondary Cox + 3-knot RCS.

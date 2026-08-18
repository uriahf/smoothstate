---
name: smoothstate
description: >
  Smooth state-occupation probabilities over a continuous predictor. Use when writing Python code that uses the smoothstate package.
compatibility: Requires Python >=3.10.
---

# Smooth State

Smooth state-occupation probabilities over a continuous predictor

## Installation

```bash
pip install smoothstate
```

## API overview

### Binary state smoothing

- `smooth_binary_state`: Smooth ``P(state=1 | prob)`` with local linear regression

### Time-dependent state smoothing

- `smooth_state_cox`: Smooth event-state probability using secondary Cox + 3-knot RCS

## Resources

- [Full documentation](https://uriahf.github.io/smoothstate/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs

---
name: smoothstate
description: >
  Fast smoothing of state probabilities for Polars workflows. Use when writing Python code that uses the smoothstate package.
compatibility: Requires Python >=3.9.
---

# Smooth State

Fast smoothing of state probabilities for Polars workflows

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
- [Source code](https://github.com/uriahf/smoothstate)

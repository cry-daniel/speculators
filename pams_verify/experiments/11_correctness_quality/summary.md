# Phase 12 Correctness and Quality

Token-ID exactness was checked in the offline synthetic audit path. This does not prove live vLLM exactness.

- Exact fallback decision match: `1.0000`
- Exact fallback false accept: `0.0000`
- Approximate sparse false accept: `0.1413`
- Exact fallback token-ID exact match: `True`
- Approximate token-ID exact match: `False`
- No task-quality claim is made because no labeled quality dataset was run.

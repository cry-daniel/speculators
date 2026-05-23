# Integration C: Sparse Prefilter + Dense Fallback

Two fallback policies were evaluated on the trace as an offline proxy. No live vLLM prefilter path ran in this turn.

- Exact fallback decision match: `1.0000`
- Exact fallback false accept: `0.0000`
- Approx fallback false accept: `0.1051`
- Approx fallback dense fallback ratio: `0.2898`

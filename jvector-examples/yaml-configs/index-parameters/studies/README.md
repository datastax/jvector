# Studies

Recipes for reproducible comparisons. See
[Reproducing ASH versus PQ](../../../../docs/ash-pq-benchmark.md) for how to run them
and why there are two forms.

- [ash-vs-pq](ash-vs-pq): 2-bit ASH and PQ at matched code sizes on five public 1M
  datasets, with a separate JVM for construction and for each query core budget.
  Run it with `run-ash-pq-study.py`. This is the form for reported comparisons.
- [ash-vs-pq-single-jvm](ash-vs-pq-single-jvm): the same cases, each in one JVM that
  builds once and runs one search case (8 search workers, overquery 1). Run them with the
  [benchmark launcher](../../../bench-profiles/README.md) for a quicker end-to-end check.

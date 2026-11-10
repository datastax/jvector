# Studies

Recipes for published comparisons. See
[Reproducing ASH versus PQ](../../../../docs/ash-pq-benchmark.md) for how to run them,
and [ASH performance](../../../../docs/ash-performance.md) for the results.

- [ash-vs-pq](ash-vs-pq): ASH and PQ at matched code sizes on 1M and 6M/10M datasets,
  with separate JVMs for construction and each query core budget. This is the
  protocol behind the published results; run it with the study's scripts.
- [ash-vs-pq-single-jvm](ash-vs-pq-single-jvm): the same cases, each in one JVM that
  builds once and queries with 1 and 8 search workers. Run them with the
  [benchmark launcher](../../../bench-profiles/README.md); future studies use this form.

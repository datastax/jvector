Added reusable prepared-query and bounded asynchronous execution APIs. Compatible
ASH queries can use native FP16 batch projection before independent graph searches,
with immediate processing under light traffic and bounded admission under load.
Execution uses host-owned capacity, releases idle threads, contains callback
failures and drains accepted work on close. Existing synchronous search and index
formats remain supported.

BenchYAML supports YAML-selected queued execution, explicit run/index configuration
paths, and source/concurrency metadata. The repository includes fixed fused PQ/ASH
comparison configurations and standalone reproduction commands.

See [query integration](../../ash-query-integration.md) and the
[Reproducing ASH versus PQ](../../ash-pq-benchmark.md).

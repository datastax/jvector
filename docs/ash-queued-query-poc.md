# Queued ASH query POC: review guide

This branch preserves the experimental queued-query architecture for independent
review. It has a reproducible unloaded tail-latency regression and is not a
finished production recommendation. No Cassandra changes are included.

## Base and scope

The base is local `ash-dev` at `cd890861fe707e99775b67ff18b15ef62ab7fa55`.
That base already contains the native FP16 batch kernel (`85699a49`) and the
synchronous caller leader-follower implementation (`cd890861`). Review the new
POC with:

```sh
git diff cd890861fe707e99775b67ff18b15ef62ab7fa55..ash-queued-query-poc
```

The existing synchronous search API and its caller-combining path remain
available. Explicitly prepared query batches call the same native FP16 kernel
directly, bypassing caller combining. Encoding and training remain FP32.

## Execution and ownership

1. A producer submits independently owned requests to `BatchingExecutor`.
   Multiple requests can be outstanding without a blocked caller for each one.
2. Workers take currently available preparation requests sharing an identity
   key. They do not wait for a batch-fill timer.
3. `ASHQueryPreparer` snapshots FP32 inputs and prepares projection, landmark
   terms and LUTs into reusable `PreparedQuery` states. A formed batch uses
   native FP16 projection with FP32 output/accumulation.
4. A single prepared request executes on its preparing worker. Larger groups
   publish individual searches to a shared ready queue; any execution worker
   may execute them. Each traversal exclusively owns its searcher and graph view.
5. Completion returns each result/error through the request's callback. There
   is no requirement to return execution to the submitting thread. The
   synchronous benchmark latency path does wait for its caller to resume.

The generic executor uses caller-supplied execution capacity. BenchYAML supplies
one persistent worker per allocated physical core, shared across its benchmark
phases and overquery cases for a configured index. Default maximum batch size
is 8; total admission is `8 * cores` for batch-size-1 and batch-size-8 controls.
The implementation also permits larger explicitly selected batches. This is a
POC policy, not a completed production sizing study.

A prepared state retains its input snapshot and numerical buffers, not a graph
view. Do not reuse it until all searches/resumable iterators borrowing it finish.
State accounting excludes the shared model. Keep index/model resources alive
until accepted requests finish, including requests whose clients timed out.
Close/drain the batching executor before shutting down its supplied workers.

## Code map

- `graph/similarity/PreparedQuery.java`, `QueryPreparer.java`: numerical state API.
- `quantization/ASHQueryPreparer.java`: input snapshots and numerical preparation.
- `quantization/AsymmetricHashing.java`: direct/supplied-batch projection entry points.
- `graph/ImmutableGraphIndex.java`, `graph/disk/OnDiskGraphIndex.java`:
  preparer access and scoring with supplied preparation.
- `quantization/FusedASHDecoder.java`, `ASHNeighborhoodScorer.java`,
  `graph/disk/feature/FusedASH.java`: reuse supplied projection/LUT in graph scoring.
- `util/BatchingExecutor.java`: admission, compatible preparation, ready searches,
  completion, close/drain and batch counts.
- `example/benchmarks/QueuedQueryExecution.java`: standard BenchYAML queued mode.
- `example/Grid.java`: worker ownership and configured-index lifetime.

## BenchYAML configuration

Select the mode in the existing run YAML; use the usual dataset/index YAMLs:

```yaml
queryExecution:
  mode: queued
  maxBatchSize: 8
```

The default mode is `direct`. Setting `maxBatchSize: 1` is a no-batching control
with the same admission budget; it is not how isolated production traffic is
selected. LatencyBenchmark submits one active request even with maximum batch 8.
The mode uses the existing random warmups, full query passes, timing boundaries,
accuracy checks, and query parameters. There is no separate arrival adapter.

Optional console/logging named metrics:

```yaml
metrics:
  execution: [response_mean_ms, response_p95_ms, response_p99_ms, mean_batch_size]
```

## Collected results and unresolved issue

Latest tested source revision is the recorded POC v5. Performance uses JDK 22,
native defaults, oneDNN sequential runtime, CPU 0 or physical CPUs 0-7, NUMA node
0, saved 1M graphs, C=1 landmarks, 2-bit ASH, matched PQ code budgets and NVQ
reranking. Both 1x and 2x rerank overquery are measured. Numbers below are the
three-sample BenchYAML mean QPS from one process per case, not a confidence claim.

| Dataset | Cores | Overquery | PQ QPS | Queued ASH QPS |
| --- | ---: | ---: | ---: | ---: |
| MPNet | 1 | 1x | 11004.60 | 13730.79 |
| MPNet | 1 | 2x | 8910.11 | 10229.72 |
| OpenAI3072 | 1 | 1x | 3592.86 | 5306.32 |
| OpenAI3072 | 1 | 2x | 3002.15 | 4205.70 |
| OpenAI3072 | 8 | 1x | 29439.43 | 41519.84 |
| OpenAI3072 | 8 | 2x | 24503.91 | 33684.40 |

ASH recall is unchanged from the synchronous reference on these saved graphs:
MPNet 0.82012/0.93911 and OpenAI3072 0.7713/0.8979 at 1x/2x. Corresponding PQ
recalls are 0.65429/0.81957 and 0.6374/0.7960. MPNet eight-core, four-bit cases,
and other datasets are not collected for this final POC revision.

The central unresolved regression is unloaded MPNet 1x latency: synchronous
mean 77.444 us and p99.9 0.137355 ms versus queued mean 92.844 us and p99.9
2.826679 ms. At 2x, the means are 101.034/106.326 us and p99.9 values are
0.180007/0.173778 ms. Do not hide the 1x result by reporting only 2x or averages.

Confirmed structural differences requiring investigation:

- An isolated queued request has caller-to-worker dispatch and worker-to-caller
  wakeup. There is no batch-fill delay and no ready-queue hop for size 1.
- Full batches use `prepareBatch`; isolated requests use `prepare`. Loaded
  completion uses pass bookkeeping; isolated completion uses a semaphore.
  Warm worker threads do not imply both paths/native workspaces are initialized
  and compiled. This is a hypothesis, not an established cause of the regression.
- Loaded `response_*` timestamps are taken on the worker before completion
  delivery. Unloaded latency includes the waiting caller resuming and returning.
  These endpoints differ; loaded p99 and unloaded p99.9 are also distinct metrics.
- Pooling requests does not prove zero allocation in blocking-wait machinery.

The worker-lifetime correction applies only to the new queued BenchYAML mode.
It did not resolve the tail regression and is not a general BenchYAML warmup fix.
Earlier JFR evidence on an older OpenAI POC showed compilation overlapping long
waits; it does not prove the cause of the latest MPNet regression. No change to
GraphSearcher visited-set behavior is part of this POC.

## Validation and evidence

The committed source matches the v5 source manifest used for 61 passing JDK 22
tests: 13 native projection, 45 base/graph/queue, and 3 example/configuration tests.
The previous v4 passed 60 tests on JDK 22 and JDK 25; v5's additional persistent
worker-lifetime test has not yet been run under JDK 25.

Relevant tests: `TestPreparedASHQuery`, `TestBatchingExecutor`,
`TestQueuedWorkerLifetime`, `QueryExecutionConfigTest`,
`NativeASHFP16ProjectionTest`, `TestASHQueryPreparation`, `TestASHGraphSearch`,
`TestASHMemoryAccounting`, and `TestCallerBatchProcessor`.

Raw commands, YAML copies, CSVs, console logs, placement checks, source hashes,
and all earlier experiment results remain on the development server at:

```text
/home/ted_willke/benchmark-runs/ash-queued-query-poc-20261103/
  poc-v5-source.json
  workers-v5-tests-jdk22.log
  gate-results.json
  cases/*-rv5/command.json
  cases/*-rv5/logging/*/experiments.csv
```

No new construction measurements are claimed for this query-only POC. Profiling
preparation is local and has not been applied as production instrumentation.

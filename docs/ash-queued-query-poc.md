# Queued ASH query experiments and review history

This document preserves the earlier POC revisions and their measured limitations.
The production integration and standalone benchmark protocol are documented in
[ASH query integration](ash-query-integration.md) and the
[reproduction guide](ash-pq-benchmark.md). The historical tables below do not
claim final-candidate performance. No Cassandra changes are included.

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
4. The preparing worker executes the first search for every group size. Other
   prepared searches are published to a shared ready queue, where any execution
   worker may take them. Each traversal exclusively owns its searcher and view.
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

Set `queryExecution: queued` and `maxBatchSize: 8` in the parameters of
the ASH search compression entry. See the
[BenchYAML guide](benchyaml-guide.md#ash-query-parameters) for placement.

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

## Initial POC measurements and the latency issue

The following table records the initial POC v5 baseline. Focused follow-up
changes and measurements are documented below. Performance uses JDK 22,
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

The initial v5 source matched its recorded manifest and passed 61 JDK 22
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


## Focused tail-latency correction

The follow-up patch keeps the preparation and first-search execution path the
same for singleton and full-batch traffic. `QueuedQueryExecution` uses
`prepareBatch` for all group sizes. The latest review correction makes
`ASHQueryPreparer` select direct preparation for count 1 on every backend,
including native FP32. Larger groups use the backend's batch projection. `BatchingExecutor` publishes searches 2..N and runs the first
locally. Asynchronous submission remains asynchronous, and the CPU budget,
precision, admission policy, graph state and search parameters are unchanged.

Queued response-sample aggregation is now deferred until `QueryTester` has
finished its query benchmarks. Previously that reporting calculation could
trigger compilation that continued into the subsequent latency measurement.
This correction preserves every query, latency sample and reporting statistic.
It affects the new queued mode, not the existing direct-mode warmup policy.

Deferral removes an observed trigger for compilation interference; it does not
guarantee that compilation will stay out of timed work. Both direct and queued
execution can still experience late compilation. No compilation-quiescence gate,
different warmup queries, or new compiler flags are introduced by this patch.

The six-point request trace, JFR and CPU-0 scheduler recording showed that the
largest stalls were runnable worker CPU waits while C2 compiled code. One
13.137 ms request included 12.980 ms of worker run-queue delay; C2 occupied that
core for 12.813 ms. The unchanged published classes reproduced long waits
alongside compilation, so the timestamp overlay was not required for the issue.
The trace is diagnostic evidence, not replacement performance data.

Three fresh unprofiled JDK 22 JVMs per flow were compared on MPNet 1M, 2-bit,
one physical core. Both flows below use the corrected reporting schedule:

| Overquery | Original flow QPS, mean +/- SD | Unified flow QPS, mean +/- SD | Original unloaded p99.9, median [range] ms | Unified unloaded p99.9, median [range] ms |
| --- | ---: | ---: | ---: | ---: |
| 1x | 13585.66 +/- 182.00 | 13681.17 +/- 199.35 | 0.372607 [0.291989, 0.562090] | 0.211354 [0.144735, 0.230088] |
| 2x | 10245.88 +/- 104.74 | 10274.58 +/- 80.55 | 0.170740 [0.169270, 0.171266] | 0.169295 [0.167371, 0.170570] |

Percentiles are medians/ranges of per-process percentiles, not pooled percentiles.
Recall remained exactly 0.82012/0.93911. QPS differences are small relative to run
variation. The earlier direct synchronous 1x reference was 77.444 us mean and
0.137355 ms p99.9; this patch does not establish complete unloaded-latency parity.

The follow-up passed 34 relevant tests on each of JDK 22 and JDK 25, including
new reporting-order and failure-cleanup tests. Native kernels are unchanged.
The supplemental execution metrics are collected by the standard `QueryTester`
pipeline after the benchmarks finish; they are no longer calculated inline at
the end of the queued throughput benchmark.

Evidence is retained under the development server's `tail-profile` directory:

```text
/home/ted_willke/benchmark-runs/ash-queued-query-poc-20261103/tail-profile/
  focused-repeat-summary.json
  flow-report-candidate/checks-jdk22.log
  flow-report-candidate/checks-jdk25.log
  cases/mpnet-2bit-C1-trace-01/stage-analysis.json
  cases/mpnet-2bit-C1-original-profile-01/scheduler-latency.txt
```

The broader executor changes discussed in review, including removal of the slot
pool, short-lived drain tasks, callback failure policy and a CPU-budgeted
synchronous caller fast path, are not part of this focused patch.


## Review corrections and matched direct comparison

The review correction routes a one-row `ASHQueryPreparer.prepareBatch` call to
`prepare` after checking the array bounds. Validation still precedes state
invalidation. Every projection provider therefore uses its direct kernel without
batch row-reference staging; larger groups retain batched preparation. A dispatch
test wraps a backend without its own count-1 shortcut to verify this behavior.

Deferred reports now belong to a local list in one `QueryTester.run` invocation,
not to `ConfiguredSystem`. Standalone throughput calls return their supplementary
metrics immediately. Before an execution closes, it transfers ownership of its
completed response buffer and snapshots the histogram. The later report owns
only those samples, without retaining or reading the closed executor. Transfer
does not copy the large sample buffer; its existing sort/copy is deferred until
reporting. Failed measurements discard that invocation's pending reports.

The source compiled successfully and passed 38 focused tests in each of four
combinations: JDK 22 and JDK 25, each using native default query precision and
explicit FP32. The tests cover one-row dispatch and validation, prepared states,
graph searches, queue behavior, caller combining, worker lifetime, reporting
order, failure cleanup, standalone reporting, legacy benchmark compatibility,
and reporting a completed snapshot after executor closure. Native kernels,
training, encoding, graph layout and visited-set behavior are unchanged.

The fresh performance comparison uses ordinary BenchYAML on JDK 22, MPNet 1M,
2-bit ASH, one physical core (CPU 0), NUMA node 0, and the same cached index.
The existing benchmark warmups, queries, timers, latency endpoints and recall
checks are unchanged. Direct throughput uses four synchronous caller threads;
queued throughput uses one execution worker, maximum batch 8 and admission 8.
Both are restricted to the same CPU. Unloaded latency submits one query at a
time. Three fresh JVMs per arm ran sequentially in D/Q/Q/D/D/Q order, with idle
checks before each launch and without profiling. The historical single-JVM
direct reference is not used in this matched table.

| Overquery | Direct QPS, mean +/- SD | Queued QPS, mean +/- SD | Direct unloaded mean +/- SD (us) | Queued unloaded mean +/- SD (us) | Direct unloaded p99.9, median [range] ms | Queued unloaded p99.9, median [range] ms | Recall@10, both |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x | 12725.60 +/- 52.65 | 13469.67 +/- 23.27 | 78.34 +/- 0.20 | 83.91 +/- 0.24 | 0.140633 [0.134897, 0.141831] | 0.242624 [0.241883, 0.290886] | 0.82012 |
| 2x | 9664.89 +/- 31.14 | 10167.81 +/- 120.36 | 102.20 +/- 0.21 | 105.70 +/- 0.80 | 0.175723 [0.174020, 0.182396] | 0.172512 [0.170874, 0.174481] | 0.93911 |

Means and sample standard deviations describe the three per-JVM measurements.
Tail values are medians and ranges of per-JVM percentiles, not pooled percentiles.

Queued bounded-backlog response measurements are reported separately:

| Overquery | Loaded mean +/- SD (ms) | Loaded p95, median [range] ms | Loaded p99, median [range] ms | Mean batch size |
| --- | ---: | ---: | ---: | ---: |
| 1x | 0.664472 +/- 0.001166 | 0.732305 [0.731926, 0.736801] | 0.778456 [0.773166, 0.779190] | 8.0000 |
| 2x | 0.882424 +/- 0.010087 | 0.974059 [0.958689, 0.974725] | 1.019129 [1.002839, 1.020753] | 7.9996 |

At 1x, queued QPS is 5.85% higher and unloaded mean latency is 5.58 us higher.
At 2x, queued QPS is 5.20% higher and unloaded mean latency is 3.49 us higher.

This establishes the remaining unloaded cost under a matched protocol. It does
not identify the fraction due to thread handoffs, compilation or other work,
nor show that a synchronous caller fast path would eliminate the entire gap.
Loaded response samples end on the worker before completion delivery; unloaded
latency includes return to the caller. The loaded experiment measures bounded
backlog, not a controlled arrival-rate or Poisson latency curve.

Reproducibility: the measured Java changes are recorded in
`tail-profile/review-readiness/manifest.json` and `review-fixes.patch`, based on
`46dc0543`. The YAML templates retain a legacy descriptive `jvectorRef` label;
the checked source hashes, rather than that inherited label, identify this run.
The measurements use the normally compiled repository classes, with no overlay
classpath, diagnostic timestamps or custom query harness.

Evidence under the same development-server experiment root:

```text
tail-profile/review-readiness/
  validation-passed.json
  validation-commands.json
  tests-jdk22-auto.log
  tests-jdk22-fp32.log
  tests-jdk25-auto.log
  tests-jdk25-fp32.log
  benchmark-driver.log
  matched-1.log ... matched-6.log
  matched-summary.json
cases/*-rreview-matched-*/
  command.json
  placement.jsonl
  summary.json
  logging/*/experiments.csv
```

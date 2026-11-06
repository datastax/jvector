<!--
Copyright DataStax, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Integrating ASH queries into a database

The database selects the physical-core budget **C** for search. Queries may be
submitted independently without blocking a producer thread for every outstanding
request. JVector prepares currently waiting compatible queries together, then
executes each graph search independently. It never waits for a timer to fill a
batch. An isolated request starts as soon as its execution capacity is available.

The existing synchronous `GraphSearcher` API remains available. The prepared-query
and bounded-execution APIs are optional and do not change index formats, write
ordering, FP32 encoding/training, or the existing automatic FP16/FP32 precision policy.

See [Choosing ASH query-projection precision](ash-projection-precision.md) for
processor-family recommendations and projection settings.
The [ASH configuration reference](ash-configuration.md) lists model parameters,
construction modes, runtime flags and their defaults.
BenchYAML researchers can select `ash.queryProjection` in the run YAML; the
precision guide describes JVM overrides and the requested/effective precision
columns in `experiments.csv`.

## Recommended asynchronous allocation

Use these defaults for the search CPU budget assigned by the database:

* At most **C** execution workers, shared across the indexes competing for those
  cores. Producers do not need a worker thread for every submitted query.
* Maximum preparation batch **8** and admission **8C**. These are fixed defaults,
  not settings to tune for particular datasets. Admission includes pending,
  preparing and executing requests; completion callbacks are excluded.
* One immutable model representation per loaded model, with reusable prepared
  state and exclusive search state for each active request.

| Physical search cores C | Maximum execution workers | Request admission |
| --- | ---: | ---: |
| 1 | 1 | 8 |
| 2 | 2 | 16 |
| 4 | 4 | 32 |
| 8 | 8 | 64 |
| 16 | 16 | 128 |
| 32 | 32 | 256 |
| 48 | 48 | 384 |

The integration needs to select only C. Do not expose internal batching policy
as database configuration. Larger explicit batches remain supported by the
numerical API; this recommendation does not impose an eight-row API limit.
The defaults bound memory and waiting but cannot guarantee a time limit under
arbitrary overload. Reject or backpressure arrivals when admission is exhausted.

Choose physical cores, not both SMT siblings counted separately. Reserve a
separate budget for construction, compaction and other database work. Tell
JVector the allocation before initialization using `jvector.physical_core_count`;
use `-XX:ActiveProcessorCount` consistently when appropriate. Neither property
pins threads. Use the database's scheduler, OS affinity or container controls,
and place query execution and index memory together on NUMA systems.

For an integration retaining only synchronous searches, the earlier caller
recommendation is 4C bounded caller threads, with the same physical CPU budget.
That does not mean an asynchronous integration needs 4C execution workers.
Both execution modes are measured separately by the
[reproducible BenchYAML comparison](ash-pq-benchmark.md).

## Library responsibilities and host responsibilities

`ImmutableGraphIndex.queryPreparer(similarity)` returns an optional
`QueryPreparer`. Cache it with the loaded model. `newState()` creates reusable
`PreparedQuery` buffers. `prepare` and `prepareBatch` snapshot FP32 query inputs,
compute the projection and scoring data, and retain no traversal state. ASH
uses its selected native query representation; encoding and training remain
FP32. Unsupported providers retain their existing fallback behavior.

`BatchingExecutor<T>` is a reusable two-stage executor, independent of ASH and
graph types. Its operation supplies a compatibility key, preparation, execution
and completion. A key is an immutable identity for the numerical model, not a
dataset name or mutable query buffer. Use null when preparation cannot benefit
from batching, as with the current PQ query path. Distinct keys never share a
preparation call. Mixed-model requests retain FIFO pending order; grouping only
collects compatible requests currently adjacent at the front.

The host owns the executor and maps its request objects to these operations:

```java
int searchCores = databaseSearchBudget;
int maximumBatch = 8;
int admission = Math.multiplyExact(searchCores, maximumBatch);
var searches = new BatchingExecutor<Request>(searchCores, admission, maximumBatch,
                                            databaseSearchExecutor, operation);
if (!searches.trySubmit(request)) {
    // The host still owns this request: reject or apply bounded backpressure.
}
```

The supplied executor must schedule asynchronously and explicitly reject work
it cannot accept. Do not use direct execution, caller-runs fallback or silent
discard. The service validates known unsuitable thread-pool policies. It submits
at most C reusable drain tasks and releases their execution capacity when idle;
it does not permanently park C tasks in the database's pool. A dedicated host
pool may use an array-backed work queue to avoid a queue-node allocation for
each task submission. Do not submit synchronous waiters to a pool whose entire
capacity they would occupy while waiting for this same pool to run searches.

The execution operation borrows an exclusive `GraphSearcher`/view and binds the
prepared state through `ScoringView.scoreProviderFor(prepared, true)`. It then
uses the existing search, filtering and reranking API. Models without a preparer
use ordinary scoring. Numerical batches call the native batch kernel directly,
without a second scheduling layer. The preparing worker executes the first
search and publishes the others, allowing parallel traversal without waiting
for the whole batch to finish.

## Input, result and resource ownership

An accepted request owns stable input data until preparation completes; keeping
it stable through completion is the simplest contract. Source APIs that reuse
a scratch vector must copy into request-owned reusable storage before fetching
the next vector. A batch must not consist of several references to one source
buffer that is overwritten on each fetch. The database may queue many independent
requests from one producer, provided each request's inputs have this lifetime.

Each active traversal exclusively owns its searcher, view and mutable scoring
state. `PreparedQuery` owns the query snapshot and numerical buffers; its
`ramBytesUsed()` excludes shared model weights. Do not prepare it again until
every search and resumable iterator borrowing it has finished. Publishing it
to another thread requires a normal queue/executor handoff. Generation checks
detect reuse of a prepared state that is still bound to a scoring provider.

Completion returns each result or error through that request's callback. Use
the database's request identifier/promise/response destination to correlate it;
results need not return on the original worker or finish in submission order.
Completion callbacks must return promptly and must not wait for other work that
needs the same execution budget. Admission is released before completion so
callbacks can recycle a bounded request record safely.

For a completed top-K `SearchResult`, request state can be recycled once the
callback has copied or transferred its result. A resumable iterator instead
retains an exclusive search session: keep its searcher, view and prepared state
leased until iteration finishes or closes. Do not return a worker's reusable
searcher to the pool while such an iterator is live. Budget retained sessions
and application-held results separately from the executor's admission count.

This executor schedules read-only query work. It does not reorder additions,
deletions, flushes or publication of indexes. Retain the database's existing
ordering, visibility and view-consistency rules for those operations.

A request timeout can stop the client waiting; it does not cancel a native call
or release borrowed buffers. Drain accepted work before closing model/index
resources. Interruption is preserved by close, which waits for callbacks too.
Close the batching service before shutting down its host executor; forcibly
discarding already accepted host tasks violates the execution contract.

Preparation, search and explicit dispatch failures are delivered once to each
accepted request. A throwing completion callback is contained: inspect
`completionFailureCount()` and `firstCompletionFailure()` for monitoring. Later
submissions remain usable; graceful close reports retained callback failures
after draining. Call close from the lifecycle owner, not a service operation
or callback. Pool only bounded request/state storage, and include per-request
scratch, host task queues and retained sessions in the integration's memory budget.

## Deployment and measurement

Build/package native dependencies through the repository's
[oneDNN instructions](../jvector-native/src/main/native/README.md#recommended-onednn-build-and-package).
Enable Vector API modules and native access as described in the provider guide.
The preferred query implementation runs on the host workers without a nested
native thread pool; avoid adding native worker counts on top of that CPU budget.

Validate production-like CPU/NUMA placement and actual traffic. Isolated queries
must launch without deliberate batching delay; sustained backlog should create
larger batches; intermediate arrivals should neither require a different
configuration nor an unbounded queue. Include preparation, traversal, reranking,
admission and dispatch in the metrics where they occur. Compare distributions,
serious stalls and achieved QPS, with unchanged recall and explicit overload
behavior. Use the checked-in BenchYAML recipe for reproducible library comparisons;
its unloaded latency and bounded-backlog response samples are different metrics.

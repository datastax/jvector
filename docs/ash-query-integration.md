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

# Integrating ASH search into a database

ASH search uses the application's calling threads. A search is synchronous:
its result or exception is returned to the caller when the work finishes.
JVector does not create an ASH query executor. Concurrent searches sharing an ASH
model can automatically combine projection work; isolated requests start
immediately without waiting for more queries. Applications submit ordinary,
independent searches and do not construct batches.

## Recommended caller configuration

Let **C** be the physical cores allocated to search, after reserving capacity for
other database work. Count the cores available to this service or partition,
not every core in the host and not both SMT siblings as two physical cores.
Use one shared admission budget across indexes that compete for those cores.
A request that fans out to several indexes consumes several search slots.

Use **4C caller workers** and keep the default maximum projection batch size of
**8**. Apply the same rule across models and compression levels. The integration
selects its existing search CPU budget C; worker capacity is derived from it.
No model-specific concurrency setting or separate projection executor is needed.

| Search CPU budget C | Caller worker capacity 4C |
| --- | --- |
| 1 core | 4 workers |
| 2 cores | 8 workers |
| 4 cores | 16 workers |
| 8 cores | 32 workers |
| 16 cores | 64 workers |
| 32 cores | 128 workers |
| 48 cores | 192 workers |

This is the recommended configuration, not a requirement to maintain 4C requests
in flight. Under light traffic, one request starts immediately and the remaining
workers stay idle. Under load, additional callers allow searches to progress
while others wait for shared computation or I/O. They do not add CPU capacity.
Keep the worker capacity fixed as arrival rate changes; let bounded admission
control overload instead of creating more workers or waiting to fill a batch.

Validate end-to-end latency against the database's service objectives. More
runnable callers than cores can introduce CPU scheduling tails. Record p50, p95,
p99 and deadline misses as well as QPS; the recommendation does not guarantee an
improvement in every latency percentile.

Use a bounded platform-thread pool or an equivalent database scheduler for
these synchronous calls. A large number of tasks or virtual threads is not by
itself evidence that enough independent searches can execute concurrently.
Blocking native computation occupies the executing thread; test the scheduler
that will actually run in production.

## Tell JVector about the allocation

Before initializing JVector, set the existing JVM property
`-Djvector.physical_core_count=C` to the physical-core budget assigned to search.
ASH also respects the JVM's `availableProcessors()` limit, using the smaller of
these two counts. The property is a sizing hint: it does **not** pin threads or
reserve cores, and it also sizes other JVector physical-core executors. Use the
database's CPU controls, OS affinity, or container allocation to enforce the
budget. If necessary, use the JVM's `-XX:ActiveProcessorCount=C` consistently
with that allocation.

On NUMA machines, place search callers and index memory together where practical.
Validate placement rather than assuming a thread-count property controls it.
Budget concurrent index construction, compaction and training separately from
query traffic so they cannot silently consume the search allocation.

Keep ASH's default maximum projection batch size of **8**. No other ASH query
coordination setting is needed. The optional JVM property
`-Djvector.ash.queryBatchSize=1` disables query combining for comparison or
operational rollback. This property is read when each ASH model is constructed
or loaded; set it before doing so. It does not change compression, the index
format, or FP32 base-vector encoding/training. Batch size does not set the
application's queue depth or concurrent search count.

For native deployment, follow the repository's
[oneDNN build and packaging instructions](../jvector-native/src/main/native/README.md#recommended-onednn-build-and-package).
JDK 22+ native access and the Vector API must be enabled as described in the
[provider instructions](../README.md#simd-provider-selection). The packaged
oneDNN query implementation runs on the calling threads without a nested native
thread pool. Retain that model; adding native query threads on top of the
application's callers oversubscribes the CPU budget. When optimized projection
is unavailable, JVector retains its existing projection fallback.

## Admission, state ownership and completion

The database owns arrival scheduling, overload handling and completion delivery:

1. Admit a request to a bounded search executor or reject/backpressure it when
   that executor and its bounded queue are full. Size the admission queue from
   the request latency budget; an unbounded queue turns overload into long
   delays and retained request memory.
2. Reuse the loaded index and its immutable ASH model across searches. Give
   each active search exclusive use of a `GraphSearcher`, its view and mutable
   scoring/scratch state. A pool can reuse that state after a call completes;
   concurrent calls must not share one searcher's mutable state. Creating a
   separate model per request prevents sharing and repeatedly allocates matrices.
3. Obtain the query vector from the database's normal request/source API before
   invoking the synchronous search. Keep it and any borrowed buffers valid and
   unmodified until the call returns. ASH does not fetch queries from an input
   source or take ownership of the database's request queue.
4. Return the result or exception through the database's existing response path.
   An asynchronous integration may complete an application-owned promise after
   the synchronous call finishes. JVector does not create a future per query.

A request timeout may stop the database waiting for a result, but must not cause
its buffers, searcher, or index/model resources to be reused or closed while
native work is still running. Release them only after the actual call finishes.
Interrupts are preserved; they do not abandon borrowed buffers in another
caller's computation. Apply deadlines and cancellation at admission and at safe
boundaries supported by the integration. Include waiting for admission in
end-to-end latency.

## Validate the integration

Use the database's normal query mix and scheduler, with native dependencies and
CPU/NUMA placement matching deployment. Compare the default against batch size 1
under the same placement and caller budget:

- **Isolated requests:** submit one query at a time to the normal 4C-worker
  executor with the default batch size. There should be no deliberate delay to
  collect more queries. One active request is a traffic scenario, not a different
  worker configuration.
- **Sustained backlog:** keep the recommended 4C callers supplied with work.
  Record achieved QPS, CPU utilization, memory and p50/p95/p99
  acceptance-to-completion latency. Keep the 4:1 worker/core ratio when comparing
  different CPU budgets.
- **Intermediate and bursty traffic:** test several offered rates below
  saturation, including a Poisson arrival stream and realistic bursts. Report
  offered and achieved QPS separately and keep late requests in the latency
  measurement instead of dropping them.

Measure the entire real search, including query preparation and reranking.
Check recall against the same queries and ground truth, and require no regression
when changing execution policy. Repeat matched runs to distinguish scheduling
variance from a persistent regression. A projection-only speedup is useful
analysis, but does not establish an end-to-end search improvement.

Warm each loaded model through the real query path before serving latency-sensitive
traffic if deployment permits. Include representative concurrency as well as
isolated searches during warmup. Query-specific matrices, native kernels and caller
scratch are created lazily. Include model-retained memory in the index budget and
per-active-caller scratch in the caller budget; increasing the caller count has a
memory cost even when the model is shared.

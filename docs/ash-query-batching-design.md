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

# ASH query batching: maintainer design

Database integrators should use [the caller integration guide](ash-query-integration.md).
They do not need to know or configure the coordination described here.

## Projection and policy are separate

The selected `AshProjection` receives independent FP32 query rows and caller-owned
FP32 output rows. FP16 uses converted FP16 operands with FP32 accumulation/output;
explicit INT8 uses quantized operands with integer accumulation and FP32 output.
Both native query backends use oneDNN, accept arbitrary batch counts and execute
bounded tiles of up to eight rows. Training and base-vector encoding continue
through the separate FP32 projection object. Precision selection belongs to the
[precision guide](ash-projection-precision.md), not to batching policy.

`CallerBatchProcessor` is a generic synchronous computation combiner, independent
of ASH, FFM and oneDNN. Its `Operation` accepts borrowed input/output objects and
supports direct and batch operations. It creates no executor, worker, timer,
future or per-call request object after each thread's first use. It can be reused
for other computations that benefit from combining independent caller work.

## Fixed capacity and a pressure bypass

For ASH, C is the smaller of the JVM processor limit and the existing physical
core budget, with a minimum of one. The processor has C direct slots and
L = max(1, floor(C / 4)) combining contexts. Capacity follows the CPU budget,
not a caller-thread count that the library does not own. Neither value is learned
or matrix-specific. The sole optional ASH policy property is
`jvector.ash.queryBatchSize` (default 8, 1 disables coordination).

The recommended caller-side worker capacity is 4C for every model. With eight
allocated physical cores this gives 32 caller workers and two combining contexts.
The library derives its internal capacity; database integrations do not select a
leader count. Keep the worker/core ratio fixed for throughput comparisons. A
single active query is the light-traffic scenario within that configuration.

When no work is queued or executing as a multi-query batch, a caller claims a direct slot and runs
the original singleton operation immediately. No queue publication or batch-list
mutation is involved. Only actual overlap exhausting that direct budget sends
work through the combining path. This avoids throttling lightly overlapping,
short projections to L operations.

A combining caller claims a free context and immediately takes existing FIFO
requests, up to the configured capacity. It does not wait for a full batch.
While requests are pending or a genuine multi-query batch is executing, new
arrivals join that path, preserving FIFO progress and the shared matrix
computation. Once the queue and real batches drain, free direct slots are usable
immediately. An overflowing singleton alone cannot suppress free direct capacity:
it must not create a queue phase after a brief overlap spike. The distinction
uses the actual batch width, not time measurements or prediction.

Direct calls already executing are outside the combining limit. There can be at
most C + L executing projection operations, all on existing caller threads. This
is an operation bound, not an extra thread or CPU reservation. The policy does
not promise a full batch under heavy overall search traffic: backlog must occur
at projection, not merely elsewhere in search.

## Publication, completion and failure

Producers publish reusable thread-local request records to a CAS stack. A caller
with free combining capacity detaches and reverses a chain under a short
nonblocking drain gate, preserving FIFO publication order. Context capacity is
never reserved for a parked caller. No queue monitor is held during native work.

Request state publishes assignment and completion. A parked-flag handshake avoids
unnecessary wakeups while preventing a completion/park race from losing a signal.
A caller may help drain work while waiting. Inputs, outputs and failure references
are cleared before reuse. A failed batch completes every member with the same
failure and releases capacity, allowing subsequent calls to proceed. Interrupts
are restored after completion; borrowed data is never abandoned mid-computation.
Reentry into the same processor is rejected. Each participating caller has at
most one outstanding request in this processor. The application's admission
budget bounds total pending work; maximum batch size is not a queue-depth limit.

Input/output row-reference arrays used to bridge a batch to FFM are reused per
executing thread, grow only when needed, and are cleared in a finally block.
They do not retain the model. Native per-thread workspaces similarly do not own
the projection or its packed weights. Native batch kernels/weights are lazily
cached on the model and shared safely across callers. Exceptions cannot cross
the C ABI; status handling remains at the Java/native boundary.

## Capability and memory ownership

A projection explicitly advertises bulk support. Providers without a real batch
kernel bypass coordination. A missing batch ABI preserves compatibility with
older native libraries. A native UNSUPPORTED response disables subsequent batch
attempts on that FP16 projection and uses the existing singleton fallback.
Explicit INT8 requires its batch ABI and a qualified exact-integer primitive;
unsupported INT8 calls remain errors rather than changing precision. Other
failures remain errors rather than being silently hidden.

The query-specific projection and coordinator are initialized together on first
query use and published through the volatile query projection. Encoding/training
alone allocates neither. ASH accounts for the coordinator, its operation object,
owned slot/context arrays, and query projection storage once per model. A query
view that aliases the primary FP32 projection is not charged twice. Native packed
weight growth updates the existing projection retained-byte estimate. Caller TLS
requests and scratch belong to the caller budget, matching existing query-scratch
ownership; they are not multiplied into each model's retained total.

Batch histograms are diagnostic snapshots, exact when the processor is idle and
weakly consistent during concurrent writes. Updates belong to exclusive slots,
without a globally contended per-query counter. Snapshot allocation is outside
the search hot path. Statistics must not influence dispatch decisions.

## Regression gates

Tests cover singleton execution at batch capacity eight, pressure-triggered
combining, FIFO progress, bounded operation counts, repeated record reuse,
interruption, reentry, batch/direct failures, completion/park races, and steady
state allocation. Native tests compare singleton and batch results with an
independent FP16-rounded reference, including tails and counts above eight,
aliasing, validation, fallback and retained-memory growth. INT8 uses an independent
integer oracle and tests concurrent first use, model switching, numeric range
checks and missing/unsupported add-ons.

Use BenchYAML for graph measurements. Compare batch size 1 and the default under
matched affinity, CPU budget, callers, index/query data and warmup. Check isolated,
saturated and intermediate arrival scenarios, actual recall and acceptance-to-return
latency. Short projections must not pay persistent queue latency when overlap
fits the direct budget; projection-heavy searches must demonstrate end-to-end
benefit. Keep compute-only kernel audits separate from graph performance claims.

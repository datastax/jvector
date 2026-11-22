# ASH query batching: maintainer design

Database integrators should use [the caller integration guide](ash-query-integration.md).
They do not need to know or configure the coordination described here.

## What it does

Projecting several queries in one matrix multiply costs less per query than
projecting them one at a time. When synchronous searches on different threads
need a projection at the same time, ASH can combine them into one batch on one
of those caller threads. It creates no executor, worker, timer or thread pool.

The combiner, `CallerBatchProcessor`, is generic: it knows nothing about ASH,
FFM or oneDNN, and can serve other computations that benefit from combining
independent caller work.

## Setting

The only setting is the maximum batch: `AsymmetricHashing.setQueryOptions` per
model, or the `jvector.ash.queryBatchSize` property as the default (8; 1 turns
batching off).

## Capacity and the light-traffic path

Capacity follows the CPU budget. With C the smaller of the JVM processor limit
and the physical-core budget, the combiner has C direct slots and
max(1, floor(C / 4)) combining contexts. The recommended caller worker count is
4C.

While nothing is queued, a caller takes a direct slot and projects its own query
immediately, so light traffic pays no batching cost. Only when overlap uses up
the direct slots do callers queue. A combining caller takes the requests already
waiting, up to the maximum batch, without waiting for a full batch, and requests
are served in arrival order.

## Failures and fallback

A failed batch fails every query in it and frees its capacity, so later calls
proceed. Reentry into the same processor is rejected. A projection without a
batch kernel, or an older native library without the batch entry point, uses
single-query projection. FP16 falls back to single-query projection if the
native batch reports the operation unsupported; explicit INT8 reports an error
instead of changing precision.

Each model counts its combiner and query projection once in `ramBytesUsed()`.
Per-thread request records and scratch belong to the caller's budget.

## Measuring

Use BenchYAML for graph measurements. Compare batch size 1 with the default
under the same affinity, CPU budget, caller count, data and warmup, at isolated,
saturated and intermediate arrival rates, reporting recall and latency.

# Batched input vector access

`RandomAccessVectorValues` remains the baseline input interface. Sources may also
implement `BatchedVectorValues` to supply bounded range and selection cursors.
This concerns input vectors, independently of the stored graph's reader or writer.

## Responsibilities

- The algorithm chooses its sample, scan, or output-to-input ordinal mapping.
- `VectorAccess` chooses the optional cursor path or the existing point-access fallback.
- The source owns file format, scheduling, I/O workers, buffering, and read-ahead.
- Callers own vectors retained for training or other computation.

No quantization policy, benchmark phase name, global source registry, or disk graph
layout belongs in the file source. Adopting the utilities is optional for each
quantizer or index; implementations supplied by existing integrations remain valid.

## Consumer examples

PQ selects its training ordinals exactly as before, then materializes them:

```java
List<VectorFloat<?>> training = VectorAccess.copySelected(source, ordinals, executor);
```

For a capable source, the utility schedules selection reads in ordinal order and
restores the original sample order. The sample itself, duplicates, initialization,
and centroid learning are unchanged. The fallback preserves the existing parallel
point-access path and avoids copying non-shared resident vectors.

PQ encoding consumes bounded partitions through the same utility:

```java
VectorAccess.forEach(source, count, ordinalMapping, executor,
        (position, vector) -> pq.encodeTo(vector, outputSlice(position)));
```

NVQ's mean scan preserves its original accumulation order:

```java
try (var cursor = VectorAccess.openRange(source, 0, source.size())) {
    while (cursor.next()) VectorUtil.addInPlace(mean, cursor.vector());
}
```

NVQ bulk encoding also uses `VectorAccess.forEach`. Other quantizers or indexes
can adopt either utility without adding file-specific code.

## Cursor and source lifetime

A cursor is single-consumer. Independent cursors may run concurrently. A shared
cursor vector is borrowed until the next advance or close; copy it to retain it.
Non-shared resident sources allow retaining vectors without copying. Selections
preserve order and duplicates. Keep the selection array unchanged until close.

Close cursors with try-with-resources, stop consumers, and then close the source.
`FvecFileVectorValues` copies share one storage owner; closing a copy only closes
that view, while closing the root releases the I/O executor, channel, and cache.
I/O errors propagate as `UncheckedIOException`. Files must remain immutable.

## Buffered fvec source

`FvecFileVectorValues.open(path)` validates fixed-width fvecs and offers both the
point API and optional cursors. Range reads use contiguous bounded payloads.
Selected reads place individual requested records into reusable batches, with
one asynchronous task per batch instead of one future per vector. It does not
read gaps between random samples.

The default payload budget is `min(1% of input bytes, 64 MiB)`, with one record as
the minimum. All copies and cursors share this limit, including queued, in-flight,
and ready buffers. Twelve I/O workers and three batches of look-ahead are defaults;
limits are configurable. Demand leases prevent recycling while data is decoded.
Speculative reads may be evicted under pressure; consumed payloads are preferred
for reuse. Closing a cursor waits for reads that still reference its selection.

Input payload buffers, algorithm-owned training vectors, cursor scratch vectors,
JVM/channel internals, and the OS page cache are distinct memory categories. The
payload limit is not a total-process memory limit. The source neither maps files
nor evicts OS pages; cold-cache controls belong in benchmark setup.

`statistics()` exposes cumulative logical read counts/bytes and payload buffer
allocation for diagnostics; these are not physical device I/O counters.

## Validation and comparison

Focused tests cover borrowed-vector ownership, order/duplicates, resident fallback,
48 concurrent consumers under a one-record budget, invalid files, closed sources,
and PQ/NVQ equivalence using the same quantizer. The MPNet 1M comparison uses the
existing accepted writer identically across in-memory, mmap, and buffered loaders,
fresh training for each arm, and verified cold input before each run. This is a
cold-start comparison on a dataset that fits in RAM, not a memory-limited study.

## MPNet 1M preliminary result

| Metric | In-memory preload | mmap PR source | Buffered source |
|---|---:|---:|---:|
| Input preparation (s) | 12.57 | 0.17 | 0.22 |
| PQ training (s) | 5.88 | 6.36 | 11.02 |
| PQ encoding (s) | 3.13 | 3.10 | 3.27 |
| NVQ setup (s) | 0.31 | 0.32 | 0.49 |
| Graph insertion (s) | 22.10 | 46.83 | 27.86 |
| Graph cleanup (s) | 2.39 | 2.08 | 2.28 |
| Serialization (s) | 6.22 | 6.88 | 6.85 |
| Library pipeline (s) | 40.04 | 65.58 | 51.77 |
| Total process (s) | 61.56 | 72.33 | 59.07 |
| Process CPU (s) | 1606.22 | 2743.79 | 1893.42 |
| Peak heap (GiB) | 8.83 | 4.97 | 6.85 |
| Peak buffer pool (GiB) | 0.01 | 2.87 | 0.01 |
| Peak total RSS (GiB) | 10.47 | 9.45 | 7.99 |
| Cumulative GC reclaimed (GiB) | 58.73 | 59.27 | 59.59 |
| Physical input reads (GiB) | 3.03 | 3.02 | 3.02 |
| Physical output writes (GiB) | 3.94 | 3.94 | 3.94 |

MPNet 999,812 × 768, DOT_PRODUCT, fresh PQ training per arm, 48 build/writer workers, native avx3_spr. Input on boot Persistent Disk; output on RAID. Same accepted deferred NVQ/complete-record writer and common quantization access code across arms. This compares loaders, not stock-main versus the optional-disk branch. Cold input verified before each arm; the dataset fits RAM and warms during the run.

Heap/buffer pool are sampled peaks. RSS is a separate total-process peak. The file source used 29.29 MiB of reusable heap payload buffers against a 29.33 MiB limit; these are included in heap, not buffer-pool reporting. GC reclaimed is cumulative, not resident memory. Fresh independent codebooks and graphs were retained. No query benchmark was part of this build-only loader study.

The refactored source retains the serialization improvement (6.85s versus the first refactor's 11.54s), with a shared 29.29 MiB payload pool. Total process time was 59.07s versus 72.33s for the mmap source and 61.56s with resident preload. The resident library pipeline remains faster (40.04s versus 51.77s), especially training (5.88s versus 11.02s). The next performance target is selected-read sampling; this result does not establish behavior for a dataset larger than available memory.

Six focused native-provider tests passed, including the exact 128,000-vector Floyd sample/order and 48 concurrent readers with a one-record payload budget. The base module compiles for Java 11. Tests reused the frozen x86 native library to isolate this Java/API work; the ordinary whole-reactor native build requires initializing the Highway submodule in the new checkout.

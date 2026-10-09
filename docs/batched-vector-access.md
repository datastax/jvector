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
that view, while closing the root shuts down the I/O executor, closes the channel,
and drops cached payload references. Pooled direct buffers are reclaimed by the JVM
after their references are released.
I/O errors propagate as `UncheckedIOException`. Files must remain immutable.

## Buffered fvec source

`FvecFileVectorValues.open(path)` validates fixed-width fvecs and offers both the
point API and optional cursors. Range reads use contiguous bounded payloads.
Selected reads place individual requested records into reusable batches, with
one asynchronous task per batch instead of one future per vector. It does not
read gaps between random samples.

The default payload budget is `min(1% of input bytes, 64 MiB)`, with one record as
the minimum. All copies and cursors share this limit, including queued, in-flight,
and ready buffers. Forty-eight I/O workers and three batches of look-ahead are defaults;
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

## Loader regression audit

The initial refactor unintentionally reduced sample-read concurrency from 48 to
12 workers and replaced the earlier direct read buffers with heap buffers.
Positional reads into heap buffers require a temporary native buffer and a copy.
The correction restores 48 I/O workers and uses the same bounded pool for direct
payload buffers. Batch sizing preserves the 52-record full-scan batches and
29.29 MiB allocation on this MPNet input; increasing workers does not multiply
this payload budget.

A loader-only diagnostic used the unchanged production 128,000-vector selection,
without centroid learning, starting with verified zero resident input pages before
each case. Sample hashes matched across all cases and both passes:

| Loader | Cold selected-vector loading (s) | Subsequent loading (s) |
|---|---:|---:|
| Refactor, heap buffers, 12 workers | 4.60 | 0.25 |
| Earlier synchronous direct-buffer loader, 48 workers | 1.49 | 0.10 |
| Direct buffers, 12 workers | 3.81 | 0.22 |
| Heap buffers, 48 workers | 1.32 | 0.25 |
| Corrected direct buffers, 48 workers, preserved 52-record batches | 1.14 | 0.23 |

The worker reduction caused most of the cold selected-read slowdown in these
measurements. Direct buffers also avoid an unnecessary copy. Subsequent loading
is still slower than the older synchronous loader; the difference was about
0.13s for this sample. This diagnostic does not establish performance for data
larger than RAM.

The audit also corrected three edge cases: empty input is again rejected by NVQ
mean computation, overflowing fvec dimensions/buffer sizes are rejected or bounded,
and closed point-access views drop their current payload reference. Source copies,
cursor borrowing, sample order, duplicate handling, ordinal mapping, and fallback
ownership were checked against main. PQ's sample policy, centroid algorithm and
native kernels are unchanged. The main ByteBufferReader and graph reader/writer
were not modified on this branch.

## MPNet 1M corrected-source validation

| Metric | In-memory preload | mmap PR source | Buffered source |
|---|---:|---:|---:|
| Input preparation (s) | 12.57 | 0.17 | 0.23 |
| PQ training (s) | 5.88 | 6.36 | 7.61 |
| PQ encoding (s) | 3.13 | 3.10 | 3.82 |
| NVQ setup (s) | 0.31 | 0.32 | 0.52 |
| Graph insertion (s) | 22.10 | 46.83 | 47.81 |
| Graph cleanup (s) | 2.39 | 2.08 | 2.31 |
| Serialization (s) | 6.22 | 6.88 | 6.27 |
| Library pipeline (s) | 40.04 | 65.58 | 68.32 |
| Total process (s) | 61.56 | 72.33 | 72.81 |
| Process CPU (s) | 1606.22 | 2743.79 | 2830.54 |
| Peak heap (GiB) | 8.83 | 4.97 | 4.74 |
| Peak buffer pool (GiB) | 0.01 | 2.87 | 0.04 |
| Peak total RSS (GiB) | 10.47 | 9.45 | 5.92 |
| Cumulative GC reclaimed (GiB) | 58.73 | 59.27 | 69.88 |
| Physical input reads (GiB) | 3.03 | 3.02 | 3.03 |
| Physical output writes (GiB) | 3.94 | 3.94 | 3.94 |

MPNet 999,812 × 768, DOT_PRODUCT, fresh PQ training per arm, 48 build/writer workers, native avx3_spr. Input on boot Persistent Disk; output on RAID. Same accepted deferred NVQ/complete-record writer and common quantization access code across arms. This compares loaders, not stock-main versus the optional-disk branch. Cold input verified before each arm; the dataset fits RAM and warms during the run.

Heap/buffer pool are sampled peaks. RSS is a separate total-process peak. The file source used 29.29 MiB of reusable direct payload buffers against a 29.33 MiB limit; these contribute to buffer-pool reporting. GC reclaimed is cumulative, not resident memory. Fresh independent codebooks and graphs were retained. No query benchmark was part of this build-only loader study.

The first corrected-source build recorded PQ training of 8.42s and serialization
of 6.61s. The subsequent audited-source build above recorded training of 7.61s;
harness-only timing measured 1.30s of selected-vector loading within that phase.
The initial refactor had training of 11.02s. The earlier synchronous buffered
cold-start control recorded 7.56s; it used an earlier harness, so it is supporting
context rather than an exact end-to-end control.

Both corrected-source runs are preserved, including graph insertion times of
34.24s and 47.81s, versus the initial refactor's 27.86s. These fresh runs have
independent codebooks and parallel graph topology. The loader correction recovers
sample-loading speed and retains the serialization gain; these runs do not
establish an overall build improvement or explain the insertion variation.

The corrected-source evidence is retained at
`batched-vector-access-loader-fix-mpnet-1m-20261009` and
`batched-vector-access-audited-mpnet-1m-20261009` under the remote benchmark root.
The latter includes a harness-only timer around VectorAccess.copySelected;
production VectorAccess and centroid learning contain no diagnostic changes.

Six focused native-provider tests passed, including the exact 128,000-vector Floyd
sample/order, 48 concurrent readers with a one-record payload budget, and the audit
edge cases. The base module compiles for Java 11. Tests reused the frozen x86 native
library to isolate this Java/API work; the ordinary whole-reactor native build
requires initializing the Highway submodule in the new checkout.

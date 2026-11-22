# Batched input vector access

File-backed input vectors can be supplied without loading the entire dataset into
Java heap. `RandomAccessVectorValues` remains the basic contract. Its optional
`BatchedVectorValues` extension lets algorithms request ranges or arbitrary
selections while the source schedules bounded asynchronous I/O.

This API concerns the **input vector source**. It does not select the stored graph's
reader, writer, layout, or direct-I/O backend.

## Runnable hello-world examples

From the repository root, using the JDK required by the examples module (currently
JDK 23), run. The `jdk20` profile enables the Java vector provider and avoids
requiring a native-library build for these examples:

```bash
mvn -Pjdk20 -pl jvector-examples -am compile exec:exec@hello-batched-vectors
mvn -Pjdk20 -pl jvector-examples -am compile exec:exec@hello-batched-quantization
```

By default, both examples use **e5-small-v2-100k** from JVector's existing
[public dataset catalog](https://jvector-datasets-public.s3.us-east-1.amazonaws.com/datasets-clean/catalog_entries.yaml).
This is a real embedding dataset; its cleaned base contains **99,443 vectors with
384 dimensions**. The catalog loader downloads only the base file on first use and
reuses its cache thereafter. It honors catalog cache settings, including
`DATASET_CACHE_DIR`; there is no eager heap preload and no synthetic data generation.
The first invocation needs network access to the catalog and uncached base file.

- [HelloBatchedVectors](../jvector-examples/src/main/java/io/github/jbellis/jvector/example/HelloBatchedVectors.java)
  shows a full range scan, arbitrary selection with duplicates, and copying a borrowed vector.
- [HelloBatchedQuantization](../jvector-examples/src/main/java/io/github/jbellis/jvector/example/HelloBatchedQuantization.java)
  trains a PQ codebook and encodes the full base file with PQ and NVQ through their usual APIs.

Pass your own fixed-dimension `.fvecs` file using `-DvectorFile`:

```bash
mvn -Pjdk20 -pl jvector-examples -am compile exec:exec@hello-batched-vectors \
    -DvectorFile=/path/to/ada_002_100k_base.fvecs
mvn -Pjdk20 -pl jvector-examples -am compile exec:exec@hello-batched-quantization \
    -DvectorFile=/path/to/ada_002_100k_base.fvecs
```

Alternatively, run either class's `main` in your IDE with the file path as its single
argument. The quantization example needs at least 256 vectors. Caller-supplied files
and downloaded catalog files are never deleted or rewritten by these examples.
Values are read as stored; the source does not
normalize, deduplicate, or scrub them.

## Open a file

`FvecFileVectorValues` associates a file with the capability interface:

```java
try (var source = FvecFileVectorValues.open(Path.of("vectors.fvecs"))) {
    System.out.println(source.size());
    System.out.println(source.dimension());
}
```

The file must remain immutable while open. The source uses positional channel reads,
not memory mapping, and uses the OS page cache. Each fvecs record contains a
little-endian dimension integer followed by that many float32 coordinates; every
record must have the same dimension. I/O failures during consumption propagate as
`UncheckedIOException`.

## Dataset streaming and caches

Prepare the input through the existing dataset catalog/download cache, then open
its local `.fvecs` path with `FvecFileVectorValues`. The examples use
`loadBaseVectorFile`, which honors catalog cache settings and `DATASET_CACHE_DIR`
without materializing the base vectors. Downloads must finish before the source
is opened; cursors stream a stable local file, not an in-progress remote download.

## Purpose and benchmark opt-in

The input file need not be fully preloaded for quantizer training (samples or full
scans for statistics), batch encoding, or streaming FP/NVQ serialization. Training
samples are temporary working memory; returned models do not retain those samples.
Encoded vectors and graph structures keep their existing memory requirements.

BenchYAML and AutoBenchYAML preserve resident loading by default. To opt in, set this
root field in the index-parameter YAML used by either runner:

```yaml
useDatasetPrefetching: true
```

Omitted or `false` preserves the resident path. The existing VM option
`-Djvector.dataset_loader=buffered` explicitly overrides the YAML flag; `preload`
overrides the flag in the other direction. Neither runner changes a user's YAML files.

The harness uses file input for bulk consumption in quantized construction and
existing disk storage. Uncompressed construction, or a search configuration that
requires raw input rather than stored vectors/codes, still uses resident input.
The VM option does not bypass this resident-input routing. If an explicit
`buffered` request requires resident input, the harness logs the reason for using preload.
A mixed configuration requiring resident input remains resident; split those cases
into separate configurations to measure dataset prefetching independently. This is
source routing, not a restriction on the vector-access API.

Grid consumes each input batch through training/encoding and inline serialization;
the original insertion vector is supplied from that batch, while peer scoring uses
resident quantized codes. Query scoring is unchanged from main. Raw-input exact-query configurations retain
resident input; quantized query paths continue to use their existing codes and
stored reranking vectors. Compaction algorithms and query scoring are unchanged. AutoBenchYAML's separate
legacy compaction-query validation still uses resident original vectors; that pass
retains its existing memory requirement. The current compactor processes saved
partition indexes, not the original input file. Any future original-input training,
encoding or serialization step can use the same bulk hooks.

The source retains bounded reusable read buffers during consumption. It does not
change persistent dataset, index or quantizer caches. File preparation remains
outside construction timing; input consumption is included in the corresponding
library calls. Queries and ground truth remain resident. Files must be complete and
immutable, and buffered loading requires `NO_SCRUB` metadata. Existing direct calls
to `DataSets.loadDataSet(name)` still preload. Application-provided RAVVs need no
changes; future quantizers opt into `VectorAccess` for their bulk input needs.

Use a distinct AutoBenchYAML output/checkpoint file for each loading mode so earlier
completed results are not reused as a different mode. The harness closes its input
source after the configuration finishes.

## Random ordinals: fetch an arbitrary selection

Use a selection when the algorithm knows several requested ordinals ahead of time,
for example a random training sample. The ordinal list can be unsorted and can
contain duplicates. In this small example, every ordinal exists in the default
100k dataset:

```java
int[] ordinals = {42, 7, 90_000, 42};
List<VectorFloat<?>> retained = new ArrayList<>();
try (var cursor = VectorAccess.openSelection(source, ordinals, 0, ordinals.length)) {
    while (cursor.next()) {
        // File cursor storage is reused: copy vectors kept for later computation.
        retained.add(cursor.vector().copy());
        System.out.println("Loaded ordinal " + cursor.ordinal());
    }
}
// retained is in order 42, 7, 90000, 42 and remains valid after cursor closure.
```

For a training sample, `VectorAccess.copySelected(source, ordinals, executor)`
performs this materialization in parallel, with source-aware scheduling and the
same order/duplicate guarantees. The algorithm selects the ordinals; the source
only retrieves them. For a single unpredictable point request, the existing
`getVector(ordinal)` API remains available; copy a shared result before retaining it.

## Scan ordinals: process the file in order

Use a range for a full pass, such as computing statistics or encoding vectors.
Consume each borrowed vector before advancing; no copy is needed in this example:

```java
int scanned = 0;
double firstCoordinateSum = 0;
try (var cursor = VectorAccess.openRange(source, 0, source.size())) {
    while (cursor.next()) {
        firstCoordinateSum += cursor.vector().get(0);
        scanned++;
    }
}
System.out.printf("Scanned %d vectors; sum = %f%n", scanned, firstCoordinateSum);
```

For a partition, `openRange(source, 10, 20)` reads ordinals 10 through 19.
`openRange(startInclusive, endExclusive)` uses an exclusive end; equal endpoints
are an empty range, including `openRange(size, size)`. Selection arguments instead
use an array slice: `openSelection(ordinals, offset, count)`. Keep the selection
array unchanged until its cursor closes.

`VectorAccess` selects the optional batching capability or falls back to ordinary
point access. Algorithms should normally use these utilities rather than inspecting
the concrete file-source type. The runnable `HelloBatchedVectors` example demonstrates
both patterns, including a full scan of the real catalog dataset.

## Use with quantization and indexes

The algorithm chooses its vectors and performs computation. `VectorAccess` handles
capability selection and fallback. The source owns file format, scheduling, I/O
workers, buffering, and read-ahead.

PQ chooses its training sample as before and materializes it with:

```java
List<VectorFloat<?>> training = VectorAccess.copySelected(source, ordinals, executor);
```

For capable sources, the utility schedules selected reads in storage order, then
restores the original sample order and duplicates. PQ's sample policy and centroid
learning are unchanged. The point-access fallback retains the existing parallel
path and avoids copying non-shared resident vectors.

PQ, NVQ, and binary quantization bulk encoding use `VectorAccess.forEach`; NVQ's mean computation uses a
range cursor and preserves its accumulation order. Existing calls to
`ProductQuantization.compute`, `encodeAll`, and `NVQuantization.compute` accept the
file source directly, as shown in the runnable quantization example.

For a new quantizer, use the full-scan overload without generating all ordinals:

```java
VectorAccess.forEach(source, executor, (ordinal, vector) -> {
    // Encode or consume the borrowed vector here; copy it only if retaining it.
    encoded[ordinal] = compressor.encode(vector);
});
```

Use `copySelected` for retained training samples, `openRange` for an ordered
sequential reduction, and the mapped `forEach` overload when output positions
refer to different input ordinals. These utilities work with existing sources
without buffered hooks. Full-precision graph construction retains its existing
resident source in the harness. The optional bulk hooks are not a substitute for
resident raw vectors in comparison-heavy construction. A caller can still use the
ordinary point-access API; no library-wide policy is imposed.
The source does not change distance metrics, quantizer parameters, graph pruning,
or hierarchy. Algorithm restrictions still apply, such as FusedPQ requiring
256 clusters and anisotropic PQ requiring normalized training vectors.

Other quantizers and indexes can adopt the same utilities independently. Existing
integrations can continue supplying ordinary `RandomAccessVectorValues`; they do
not have to implement batching. No algorithm-specific policy belongs in the file
source, and alternative sources can implement the same interface.

## Configure bounded read-ahead

Most callers use `open(path)`. Immutable named options customize individual limits:

```java
var options = FvecFileVectorValues.Options.defaults()
        .withMaxBufferBytes(32L << 20)
        .withBatchVectors(64)
        .withReadAhead(3);
try (var source = FvecFileVectorValues.open(path, options)) {
    // Use VectorAccess or ordinary point access.
}
```

The default I/O worker count follows `PhysicalCoreExecutor`, the compute sizing
used by Grid and BenchYAML. It adapts to the machine and honors
`-Djvector.physical_core_count`; there is no fixed core count. Blocking file reads
for range and selection cursors run on a separate source-owned pool. Unpredictable
point reads execute on the requesting thread using exclusively leased channels;
point and asynchronous reads share the I/O concurrency limit. `withIoThreads(count)` remains available for explicit benchmark tuning. The
quantization example uses the shared compute pool and does not shut it down.

Other defaults are at most 64 vectors per batch and three batches of
read-ahead. Payload capacity is capped at `min(1% of file bytes, 64 MiB)`, with one
record as the minimum. A configured maximum is also capped at 1% of the file.
Copies and cursors share the budget, including queued, in-flight and ready payloads,
point-read buffers.
Read-ahead zero disables speculative batches. Selected reads fetch requested records
without reading the gaps between sparse samples.

This is an **I/O payload budget**, not a total-process memory limit. Retained training
vectors, encoded outputs, graph data, scratch vectors, JVM/channel internals and the
OS page cache consume additional memory. `statistics()` reports logical source
reads and payload allocation, not physical device I/O.

## Ownership and closure

- A cursor has one consumer. Independent cursors can run concurrently.
- A failed advance invalidates the cursor. Its value and ordinal are unavailable;
  open a new cursor to retry. Other views remain usable if the source is still open.
- A shared cursor vector is borrowed until the next advance or close. Call `.copy()`
  before retaining it. Non-shared resident sources allow retention without copying.
- Point-access views have their own scratch vector and are single-consumer. Use
  `copy()` or `threadLocalSupplier()` for concurrent point access.
- Close cursors with try-with-resources, stop consumers, then close the root source.
- Copies share the root's storage and I/O executor. Closing a copy affects that view
  only; closing the root invalidates all copies and releases the shared resources.
- If point reads are still running when the root closes, close waits for each read
  in progress to finish; later point reads fail with `IllegalStateException`.
- Direct payload buffers become eligible for JVM reclamation after references are
  released; closure does not promise immediate native-memory reclamation.

## Measure input loading

The [vector prefetching benchmark](vector-prefetch-benchmark.md) compares the exact
buffered source with full preload and demand-only reads, timing sample retrieval,
training and full-set encoding separately. It runs from the command line or IntelliJ.

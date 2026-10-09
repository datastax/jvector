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

The source retains bounded reusable read buffers during consumption. It does not
change persistent dataset, index, or quantizer caches. File preparation stays
outside construction timing; demand reads during training and encoding remain
part of those library calls. Existing preload paths are still available; merely
adding this source does not switch existing Grid/BenchYAML configurations to it.

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

PQ and NVQ bulk encoding use `VectorAccess.forEach`; NVQ's mean computation uses a
range cursor and preserves its accumulation order. Existing calls to
`ProductQuantization.compute`, `encodeAll`, and `NVQuantization.compute` accept the
file source directly, as shown in the runnable quantization example.

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
run on a separate source-owned pool so they do not occupy compute workers.
`withIoThreads(count)` remains available for explicit benchmark tuning. The
quantization example uses the shared compute pool and does not shut it down.

Other defaults are at most 64 vectors per batch and three batches of read-ahead. Payload capacity is capped at `min(1% of file bytes, 64 MiB)`, with one
record as the minimum. A configured maximum is also capped at 1% of the file.
Copies and cursors share the budget, including queued, in-flight and ready payloads.
Read-ahead zero disables speculative batches. Selected reads fetch requested records
without reading the gaps between sparse samples.

This is an **I/O payload budget**, not a total-process memory limit. Retained training
vectors, encoded outputs, graph data, scratch vectors, JVM/channel internals and the
OS page cache consume additional memory. `statistics()` reports logical source
reads and payload allocation, not physical device I/O.

## Ownership and closure

- A cursor has one consumer. Independent cursors can run concurrently.
- A shared cursor vector is borrowed until the next advance or close. Call `.copy()`
  before retaining it. Non-shared resident sources allow retention without copying.
- Point-access views have their own scratch vector and are single-consumer. Use
  `copy()` or `threadLocalSupplier()` for concurrent point access.
- Close cursors with try-with-resources, stop consumers, then close the root source.
- Copies share the root's storage and I/O executor. Closing a copy affects that view
  only; closing the root invalidates all copies and releases the shared resources.
- Direct payload buffers become eligible for JVM reclamation after references are
  released; closure does not promise immediate native-memory reclamation.

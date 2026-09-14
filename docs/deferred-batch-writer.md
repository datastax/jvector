# Deferred batch serialization

The parallel graph writer can serialize complete layer-zero records into bounded
worker buffers. Each record contains its ordinal, inline features, neighbor count,
and adjacency list. By default, a worker writes up to 64 complete records together.
The final partial batch is written without adding records or padding to the format.
This preserves the V6 layout and existing readers.

## Production sources

The library writer does not depend on Grid, BenchYAML, dataset preloading, or
#744's buffered-access API. It accepts the existing
`Map<FeatureId, IntFunction<Feature.State>>`: callers supply feature states for the
requested original ordinals from their own source. That source can be resident
vectors, a database's flush/compaction source, or another source; it must support
concurrent supplier calls and remain available through final serialization.
The writer does not require the original dataset to be fully resident.

Complete-record batching also applies when all input vectors are preloaded. Its
benefits come from deferring per-insertion feature writes and grouping final writes,
independently of how input vectors are loaded. Buffered range access is an optional
Grid integration, not a prerequisite for the core writer.

## Caller migration: early writes versus deferred batches

Use the parallel writer when deferring NVQ/FP encoding to final serialization.
A sequential final writer performs that encoding on one thread and can regress
construction time. Sequential callers should retain encoding during insertion
and `writeFeaturesInline`.

Both examples assume the same trained `nvq`, graph `builder`, node `count`,
`writerWorkers`, and output path. The builder's construction scoring is unchanged;
its peer vectors are supplied separately. `vectorForOrdinal` is the application's
lookup of an original input vector, not a harness loader.

### Old: write each inline feature during insertion

```java
try (var writer = new OnDiskParallelGraphIndexWriter.Builder(builder.getGraph(), outputPath)
        .withParallelWorkerThreads(writerWorkers)
        .withMapper(new OrdinalMapper.IdentityMapper(count - 1))
        .with(new NVQ(nvq))
        .build()) {
    for (int ordinal = 0; ordinal < count; ordinal++) {
        var vector = vectorForOrdinal.apply(ordinal);
        builder.addGraphNode(ordinal, vector);
        writer.writeFeaturesInline(ordinal, Feature.singleState(
                FeatureId.NVQ_VECTORS, new NVQ.State(nvq.encode(vector))));
    }
    builder.cleanup();
    writer.write(Map.of()); // Preserve the earlier feature writes; add adjacency.
}
```

### New: supply the features when writing the completed graph

```java
for (int ordinal = 0; ordinal < count; ordinal++) {
    builder.addGraphNode(ordinal, vectorForOrdinal.apply(ordinal));
}
builder.cleanup();

try (var writer = new OnDiskParallelGraphIndexWriter.Builder(builder.getGraph(), outputPath)
        .withParallelWorkerThreads(writerWorkers)
        .withMapper(new OrdinalMapper.IdentityMapper(count - 1))
        .with(new NVQ(nvq))
        .build()) {
    writer.write(Feature.singleStateFactory(FeatureId.NVQ_VECTORS,
            ordinal -> new NVQ.State(nvq.encode(vectorForOrdinal.apply(ordinal)))));
}
```

The second call supplies NVQ payloads during final serialization, together with
adjacency, in batches of up to 64 complete records. No `writeFeaturesInline` pass
is needed. The same pattern works for FP using `InlineVectors` and
`InlineVectors.State`, or another supported inline feature. Supply every configured
inline feature to obtain complete-record batching, including fused PQ if configured.

Final suppliers run concurrently. `vectorForOrdinal` must remain available through
`write` and provide independent borrowed vectors per worker, or stable vectors.
For an existing `RandomAccessVectorValues`, `threadLocalSupplier()` supplies those
independent views; it does not require buffered hooks. A caller can instead return
already encoded feature states from its own storage, without retaining original FP
vectors for this write.

Integrations that already call `write` with every inline feature supplier receive
batching when using the parallel writer. Integrations that write features early
must move those calls to final suppliers to use the deferred path. The incremental
API remains compatible: a one-pass source whose values cannot be obtained again
must retain encoded states for final writing or continue using incremental writes.

## Batch size and memory

The default is 64 records. To override it, call `withParallelBatchSize(records)`
before `with(...)` on either `OnDiskParallelGraphIndexWriter.Builder` or
`RandomAccessOnDiskGraphIndexWriter.Builder`. The value must be positive. Each active
worker holds one buffer of `recordSize * records` bytes; feature encoders and vector
sources have their own memory costs. No dataset-sized output buffer is allocated.

The unified `RandomAccessOnDiskGraphIndexWriter.Builder` retains its existing
parallel/sequential selection. Setting the batch size does not enable parallel
writing, and is ignored by the sequential writer. A caller-supplied executor
remains caller-owned and must outlive the write.

## Dataset prefetching in Grid

When the parallel writer is selected, Grid defers its inline NVQ or FP feature serialization
until after cleanup. Its final serialization suppliers consume worker-local ranges through
`VectorAccess.openRange`. With the sequential writer, which calls suppliers on one
thread, Grid keeps encoding these features on the insertion threads and writing them inline.
Sources implementing the buffered-access capability use that implementation;
ordinary sources retain the ordinary-access fallback. Borrowed vectors are consumed
before advancing a cursor, and ranges are released after the workers finish.

This integrates with #744's optional `useDatasetPrefetching` setting. It does not
change construction scoring, quantization parameters, or the query path. Production
integrations can opt into deferred serialization by supplying their features at the
final write; their existing vector-source and incremental-write APIs remain valid.

## Running the benchmark writer paths

BenchYAML and AutoBenchYAML select the parallel writer by default. To use the
sequential writer instead, add `-Djvector.parallel_writer=false` to the Java VM
options. In that mode, Grid encodes FP/NVQ features during insertion and preserves
the earlier inline-write path. No YAML setting is needed.

Grid sizes its writer pool from `PhysicalCoreExecutor`; no machine-specific worker
count is embedded in this change. Dataset prefetching is a separate optional setting
in the index YAML. An ordinary preloaded dataset can use the same deferred writer.

This default applies only to the benchmark entry points. The library's unified
writer builder retains its existing sequential default, and application callers
still explicitly choose parallel writing. Code calling Grid directly can select
its writer through `GraphIndexBuilderConfig.getInstance().setParallelBuild(...)`.

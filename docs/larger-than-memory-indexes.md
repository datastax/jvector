# Building and serving larger-than-memory indexes

An index is larger than memory when its on-disk graph does not fit in the memory left for the operating
system's page cache. Construction then has to avoid holding the source vectors, and each query reads graph
nodes from storage. This guide covers sizing, construction, the record layout on disk, and query settings
for that case. It applies to any on-disk graph; the examples use fused ASH traversal codes with NVQ
reranking, the layout that keeps the most of a search on one node read.

The [larger-than-memory BenchYAML example](../jvector-examples/yaml-configs/index-parameters/larger-than-memory)
runs the same pipeline.

## Sizing

**Disk.** Each base-layer node is one record:

| Part | Bytes |
| --- | --- |
| Node id | 4 |
| Fused ASH codes | degree × code bytes, where code bytes = `encodedBits` / 8 |
| NVQ vector | 4 + the sum over subvectors of (subvector dimensions + 28); dimensions + 60 with two subvectors |
| Full-precision vector, if stored inline | 4 × dimensions |
| Neighbor count and neighbor list | 4 × (degree + 1) |

Page alignment can add padding (see below). Upper graph layers and the header add little. Multiply the
record size by the vector count for the index size.

**Memory for construction.** Symmetric ASH construction keeps the graph's neighbor lists and every node's
ASH code in memory, not the source vectors. At degree 32 with neighbor overflow 1.2, the neighbor lists
take about 430 bytes per node on the Java heap; add the code bytes per node. Size `-Xmx` with headroom
above that, and leave the rest of memory to the page cache.

**Memory for queries.** Searches read node records through the page cache. The more of the index the
cache holds, the fewer reads reach the device, so query throughput rises with memory even when the
index does not fit.

## Construction

Construction streams the source vectors in order, trains ASH, encodes every vector, builds the graph by
comparing codes, and writes the index in one parallel pass:

```java
var pool = PhysicalCoreExecutor.pool();
int b = 2, d = 368;                                   // choose b and d with the record size in mind
var ash = AsymmetricHashing.initialize(vectors, AsymmetricHashing.ITQ,
        AsymmetricHashing.HEADER_BITS + b * d, 1, b, pool);
var mean = new VectorMean(vectors.size(), vectors.dimension());
var codes = ash.encodeAll(mean.duringScan(vectors), pool);
var nvq = NVQuantization.create(mean.mean(), 2);

var scores = BuildScoreProvider.ashBuildScoreProvider(VectorSimilarityFunction.DOT_PRODUCT, codes);
try (var builder = new GraphIndexBuilder(scores, vectors.dimension(), 32, 100, 1.2f, 1.2f, true, true)) {
    pool.submit(() -> IntStream.range(0, vectors.size()).parallel()
            .forEach(node -> builder.addGraphNode(node, scores.searchProviderFor(node)))).join();
    builder.cleanup();

    var source = vectors.threadLocalSupplier();
    try (var views = ExplicitThreadLocal.withInitial(() -> builder.getGraph().getView());
         var writer = new OnDiskParallelGraphIndexWriter.Builder(builder.getGraph(), path)
                 .withParallelWorkerThreads(PhysicalCoreExecutor.getPhysicalCoreCount())
                 .withMapper(new OrdinalMapper.IdentityMapper(vectors.size() - 1))
                 .with(new FusedASH(32, ash)).with(new NVQ(nvq))
                 .withPageAlignment(4096)
                 .build()) {
        System.out.println(writer.describeRecordLayout());
        writer.write(Map.of(
                FeatureId.FUSED_ASH, node -> new FusedASH.State(views.get(), codes, node),
                FeatureId.NVQ_VECTORS, node -> new NVQ.State(nvq.encode(source.get().getVector(node)))));
    }
}
```

- **Streaming input.** Open the base vectors as a file-backed source (`FvecFileVectorValues`) so they are
  read on demand. Scans such as encoding use contiguous ranges with bounded read-ahead; see
  [Batched input vector access](batched-vector-access.md).
- **Symmetric scoring.** `ashBuildScoreProvider` scores stored codes against each other, so insertion needs
  neither source vectors nor query projection.
- **Deferred features.** The parallel writer asks for the NVQ vectors while it writes each record, so they
  are encoded during the write instead of being held for every node; see
  [Deferred batch serialization](deferred-batch-writer.md).
- **Threads.** Graph insertion mostly waits on memory and gains from a second hardware thread per core;
  see [Thread budgets](ash-configuration.md#thread-budgets).

## Page-aligned records

A query reads one record per node it expands. When a record crosses a page boundary, that read costs two
page reads from the device instead of one. `withPageAlignment(pageBytes)` on any graph writer builder
places records so none crosses a page boundary:

- Records that fit in a page are packed as many per page as fit, with the remainder of each page left empty.
- A record larger than a page starts on a page and is padded to whole pages, but only when its last page
  would be at least half full. Otherwise the padding would cost most of a page per record while saving a
  small fraction of a page read, and the index is written unaligned.
- The header is padded to whole pages, so the first record starts on a page boundary.

`describeRecordLayout()` on the writer reports the record size, the placement and the padding, or why a
requested alignment was not applied. Alignment needs format version 7, changes no search results, and is
opt-in; without it records are packed back to back as before. Use the page size of the storage stack,
normally 4096.

Alignment pays most when a record nearly fills a whole number of pages. A record just over a page boundary
gains little and costs nearly a page of padding; reduce it below the boundary instead, with fewer ASH bits
or a lower degree. For example, at 768 dimensions with degree 32 and NVQ with two subvectors, 776 encoded
bits (368 dimensions at 2 bits) give a 4,068-byte record that fits one 4 KB page, while 808 bits
(384 dimensions) give 4,196 bytes, just over it.

Compaction does not yet accept page-aligned indexes.

## Serving queries

A query from disk makes one dependent read after another: it reads a node, chooses the next, then reads
again. Its latency is roughly the page reads per query times the device's read latency, and the CPU is
mostly idle. Throughput is then limited by the device's random-read rate:

    queries per second ≈ device random reads per second ÷ page reads per query

For example, a device rated at 268,000 random 4 KiB reads per second serving queries that make 100 page
reads each tops out near 2,700 queries per second, however many cores are available. Page alignment,
memory for the page cache, a lower degree, and less overquery all reduce page reads per query.

Settings for search that waits on storage:

- **Searches in flight.** Run more searches at once than there are cores, so others proceed while some
  wait on reads. Throughput rises with searches in flight until the device reaches its read limit; past
  that, more searches only add latency. Measure a range such as 1, 2, 4 and 8 searches per core.
- **Admission.** Admit one request per worker. A request waiting behind a busy worker adds that wait to
  its latency and adds no throughput.
- **Batching.** Use a maximum batch of 1. Preparing several ASH queries together saves microseconds of
  CPU; holding queries for a batch costs more.
- **Precision and batch size in code.** Set them on the model right after opening the index, before the
  first query:

  ```java
  var ash = ((FusedASH) index.getFeatures().get(FeatureId.FUSED_ASH)).getASH();
  ash.setQueryOptions(ASHQueryProjection.AUTO, 1);
  ```

[Integrating ASH queries into a database](ash-query-integration.md) covers bounded admission and executor
ownership; [ASH configuration and defaults](ash-configuration.md) lists the JVM settings.

## Measuring

Two independent conditions decide what a measurement shows:

- **Cold queries:** held-out queries, each presented once to a JVM whose JIT compilation has already settled
  on other inputs. A query set that repeats finds the pages its earlier passes cached and runs at memory
  speed, which a service does not see for new queries.
- **Cold index:** the index starts outside the page cache, so search reads it from storage. Warmup then
  caches the pages it touches, so it should use inputs that favor no measured query, such as a sample of
  base vectors.

For an index served from disk, measure cold queries on a cold index: evict the index, warm the JVM with base
vectors until compilation settles, and time one pass over the held-out queries. Report latency percentiles
and page reads from that same pass.

In BenchYAML, the index configuration sets `useDatasetPrefetching`, `construction.pageAlignment`,
`search.workers`, `search.queryExecution: queued`, `search.requestsPerWorker` and the ASH
`maxBatchSize`, and the
[larger-than-memory run configuration](../jvector-examples/yaml-configs/run-config-larger-than-memory.yml)
sets the measurement: `evictIndex`, base-vector warmup until compiled, one timed pass, and the response-time
and page-read metrics. Only the first search case in a JVM gives cold queries; later cases repeat them on a
warmer index, so for a cold measurement of each worker count, run each in its own JVM. See [Run configuration](benchyaml-guide.md#run-configuration).

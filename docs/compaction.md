# Graph Index Compaction

`OnDiskGraphIndexCompactor` merges multiple on-disk HNSW graph indexes into a single compacted index. This is useful in write-heavy workloads where data is continuously ingested into small segment indexes that accumulate over time; periodically compacting those segments into one larger index improves search throughput and recall without rebuilding from scratch.

## Overview

```
source[0].index  ─┐
source[1].index  ─┤──► OnDiskGraphIndexCompactor ──► compacted.index
source[N].index  ─┘
```

Each source is an `OnDiskGraphIndex` with an associated `FixedBitSet` marking which of its nodes are live (not deleted). The compactor merges all live nodes into a single graph, remaps ordinals so the output is contiguously numbered, and optionally retrains the Product Quantization codebook for the combined dataset.

## Usage

```java
List<OnDiskGraphIndex> sources = List.of(index0, index1, index2);

// Mark all nodes live (no deletions)
List<FixedBitSet> liveNodes = sources.stream()
    .map(s -> { var bs = new FixedBitSet(s.size()); bs.set(0, s.size()); return bs; })
    .collect(toList());

// Sequential ordinal remapping: source[s] node i → global offset[s] + i
int offset = 0;
List<OrdinalMapper> remappers = new ArrayList<>();
for (var src : sources) {
    remappers.add(new OrdinalMapper.OffsetMapper(offset, src.size()));
    offset += src.size();
}

var compactor = new OnDiskGraphIndexCompactor(
    sources, liveNodes, remappers,
    VectorSimilarityFunction.COSINE,
    /* executor= */ null,                 // null = jvector's shared physical-core pool
    /* taskWindowSize= */ -1              // <= 0 derives the window from the pool's parallelism
);

compactor.compact(Path.of("compacted.index"));
```

### Handling Deleted Nodes

Deleted nodes are excluded from the output by marking them as `false` in the corresponding `FixedBitSet`.

```java
// Example: every 5th node is deleted
FixedBitSet live = new FixedBitSet(source.size());
Map<Integer, Integer> oldToNew = new HashMap<>();
int newOrd = 0;
for (int i = 0; i < source.size(); i++) {
    if (i % 5 != 0) {
        live.set(i);
        oldToNew.put(i, newOrd++);
    }
}
remappers.add(new OrdinalMapper.MapMapper(oldToNew));
```

## Embedding API

For embedding the compactor into a host system (for example, a database's compaction pipeline), this branch adds `@Experimental` extension points that let the host supply its own threads, observe and throttle the merge, and place the output inside its own container file. All are additive: with none of them used, behavior and output are unchanged (a jvector-owned pool, no throttling, a standalone output file).

### Supplying an executor

The compactor dispatches its batch work through an internal `ExecutorCompletionService`, so it runs on any `Executor` the host supplies, with an explicit in-flight window:

```java
new OnDiskGraphIndexCompactor(sources, liveNodes, remappers, sim, executor, taskWindowSize);
```

- Pass a **`ForkJoinPool`** (work-stealing makes submit-and-block safe), or a **caller-runs** executor (`Runnable::run`) to run the entire merge on the *calling* thread — no jvector-owned pool. A host then gets parallelism by running multiple compactions concurrently rather than from a per-compaction pool.
- **Do not** pass a bounded `ThreadPoolExecutor` that is *also* running the calling thread: the caller blocks in `take()` while its sub-tasks queue behind it, which can thread-starvation-deadlock.
- `taskWindowSize` bounds in-flight batches (concurrency and peak write-side memory); `<= 0` derives it from a `ForkJoinPool`'s parallelism, else defaults to `1`. Read it back with `getTaskWindowSize()`.

A `ForkJoinPool` is passed as the `executor` (with `taskWindowSize <= 0` to derive its window) — there is no separate `ForkJoinPool` constructor.

### Progress and throttling

`setProgressLimiter(ProgressLimiter)` installs a control surface (package `io.github.jbellis.jvector.util.work`) that melds two independently-optional facets:

- **`onProgress(WorkStage stage, long completed, long total)`** — progress observation. Stages are `OnDiskGraphIndexCompactor.Phase.{MERGE_LEVELS, REFINE}`; `total` may be `-1` until known.
- **`acquire(long amount) → Grant`** — blocking work admission. For the compactor the unit is **bytes about to be written**. `acquire` is called on the orchestrating thread (never a pool worker), so a blocking limiter back-pressures dispatch without a `ForkJoinPool.ManagedBlocker` — exactly like ordinary code blocking on a rate limiter.

The default is `ProgressLimiter.UNLIMITED` (both facets no-op), so output and timing are unchanged when unset.

Two ready-made implementations compose as decorators:

```java
// Cap merge write bandwidth to 100 MB/s, logging throttled writes and progress.
compactor.setProgressLimiter(
    ProgressLimiter.logging(
        ProgressLimiter.rateLimited(100.0 * 1024 * 1024),   // leaky-bucket meter, bytes/sec
        msg -> log.info("compaction: {}", msg)));
```

- `rateLimited(unitsPerSecond)` — a leaky-bucket rate meter (drains when idle, interruptible; grant is a no-op).
- `logging(delegate, sink)` / `logging(sink)` — logs each `onProgress` and each *blocked* `acquire`, delegating both facets; a `sink` is any `Consumer<String>`.

A host that draws from its own shared budget implements `acquire` directly:

```java
compactor.setProgressLimiter(new ProgressLimiter() {
    @Override public void onProgress(WorkStage stage, long completed, long total) {
        metrics.report(stage, completed, total);
    }
    @Override public Grant acquire(long bytes) throws InterruptedException {
        hostIoBudget.acquire(bytes);   // block against a host-wide IO limiter
        return Grant.NOOP;             // rate-limiter model: nothing to release
    }
});
```

The returned `Grant` is closed when the admitted work completes. A **rate-limiter** realization pays its cost at `acquire` and returns `Grant.NOOP`; a **semaphore** realization (permits = in-flight bytes) releases on `Grant.close()`.

> **Caveat (semaphore realization).** During `REFINE`, batches are submitted and drained through a sliding window of `taskWindowSize`, so a permit-releasing semaphore must admit at least `taskWindowSize` batches' worth of bytes or the window can't fill. `MERGE_LEVELS` has no such constraint, and the rate-limiter realization has none in either phase.

### Output destination (no temp-file copy)

By default `compact(Path)` writes a standalone file. To place the graph body directly inside a host container after a reserved header — eliminating a temp-file-and-copy — use either overload:

```java
// (1) write the body into an existing file at a reserved offset; bytes in [0, startOffset) are preserved.
compactor.compact(componentPath, /* startOffset= */ headerSize);

// (2) resource-scoped destination with a commit/abort lifecycle; returns the body length.
long bodyLength = compactor.compact(CompactionDestination.toFile(outputPath));
```

`compact(CompactionDestination)` opens one `Target` per call and drives its lifecycle:

```java
CompactionDestination dest = () -> {
    FileChannel ch = FileChannel.open(componentPath, CREATE, WRITE, READ);
    writeHostHeader(ch);                                        // reserve [0, headerSize)
    return new CompactionDestination.Target() {
        public Path file()        { return componentPath; }
        public long startOffset() { return headerSize; }
        public void commit(long bodyLength) throws IOException { // success: finalize the container
            writeHostFooter(ch, bodyLength);
            ch.force(true);
        }
        public void close() throws IOException { ch.close(); }   // always runs; no commit ⇒ host discards the file
    };
};
long bodyLength = compactor.compact(dest);
```

- `commit(bodyLength)` fires exactly once, only on success, after the body is written and forced. `close()` always runs (try-with-resources); reaching it without a prior `commit` is an unambiguous abort — discard the partial output.
- The compactor writes into `file()` at `startOffset()` using its own random-access and memory-mapped IO — the destination expresses *where*, not *how*.
- To read the committed body back (e.g. for a checksum), address the region with the `SeekableSink` primitive (package `io.github.jbellis.jvector.disk`): `SeekableSink.over(channel, target.startOffset())` gives region-relative `writeAt`/`readAt`.

## Algorithm

### Ordinal Remapping

Each source assigns its own local ordinals. The compactor maps them to a new global ordinal space using user-provided `OrdinalMapper`.


### PQ Retraining

If the source indexes use FusedPQ, the compactor retrains the Product Quantization codebook on the combined dataset before writing the output. This is done by `PQRetrainer`, which
performs **balanced proportional sampling** across all sources (up to `ProductQuantization.MAX_PQ_TRAINING_SET_SIZE` vectors total, at least 1000 per source).


### Neighbor Selection (per node)

For each live node at each graph level, the compactor gathers a candidate neighbor pool and then applies diversity selection:

**1. Gather from same source** (`gatherFromSameSource`)\
Iterate the node's existing neighbors in its source index. Filter out deleted nodes. Score each with the similarity function. No graph search — neighbors are already precomputed.

**2. Gather from other sources** (`gatherFromOtherSource`)\
Run a graph search in every other source index starting from that source's entry point. If FusedPQ is available, approximate PQ scoring is used during the search and top results are rescored exactly.

- *Level 0*: a full hierarchical graph search is used (`GraphSearcher.search()`), descending from the entry node down to level 0.
- *Level L > 0*: the compactor first descends greedily from the source's entry node through each level above L (one `searchOneLayer` call with topK=1 per level, feeding the result into the next via `setEntryPointsFromPreviousLayer()`), then performs the full beam search at level L. This mirrors standard HNSW construction and gives a much better starting point than jumping directly to level L from the global entry node.

```
searchTopK  = max(2,  ceil(degree / numSources) * 4)
beamWidth   = max(degree, searchTopK) * 2
```

**3. Diversity selection** (Vamana-style)\
Candidates are sorted by score (descending). The compactor selects up to `maxDegree` diverse neighbors using an adaptive alpha:

```
for alpha in [1.0, 1.2]:
    for each candidate c (highest score first):
        if c is already selected: skip
        if ∀ selected neighbor j: similarity(c, j) ≤ score(c) × alpha:
            select c
    if |selected| == maxDegree: stop
```

### Hierarchical Levels

Level 0 (base layer) stores inline vectors, FusedPQ codes, and the neighbor list. Upper levels store only the neighbor list (plus PQ codes at level 1 for cross-level searching).

Processing is batched per source and run in parallel across sources on the supplied executor (a `ForkJoinPool` by default; see [Embedding API](#embedding-api)). A backpressure window keeps at most `taskWindowSize` batches in-flight at once, bounding memory use.

### Entry Node

The entry node of the compacted graph is:
1. The designated entry node of the first source that reaches the graph's max level, if it is live.
2. Otherwise, the first live node found at the max level, scanning those sources in order.

## Benchmarking

Use `CompactorBenchmark` (in `benchmarks-jmh`) to measure compaction performance. See `benchmarks-jmh/src/main/java/io/github/jbellis/jvector/bench/CompactorBenchmark.md` for full instructions.

### Default: partition and compact in one run

Adjust `-Xmx` to fit the dataset in memory (e.g., 220g for large datasets).

```bash
java -Xmx220g --add-modules jdk.incubator.vector \
  -cp benchmarks-jmh/target/compactor-benchmark.jar \
  io.github.jbellis.jvector.bench.CompactorBenchmark \
  -p workloadMode=PARTITION_AND_COMPACT \
  -p datasetNames=<dataset> \
  -p numPartitions=4 \
  -p splitDistribution=UNIFORM \
  -p indexPrecision=FUSEDPQ \
  -wi 0 -i 1 -f 1
```

### Measuring the heap memory footprint

To measure compaction's true heap footprint — without the dataset occupying heap — run the two steps separately.

> **Note:** `measureRecall` defaults to `true`. With the default, even `COMPACT` mode loads the dataset's query vectors and ground truth and runs a search after compacting, which adds heap usage. Set `-p measureRecall=false` so the heap reflects only the compactor itself.

**Step 1: build partitions** (dataset in memory, large heap required)

```bash
java -Xmx220g --add-modules jdk.incubator.vector \
  -cp benchmarks-jmh/target/compactor-benchmark.jar \
  io.github.jbellis.jvector.bench.CompactorBenchmark \
  -p workloadMode=PARTITION \
  -p datasetNames=<dataset> \
  -p numPartitions=4 \
  -p splitDistribution=UNIFORM \
  -p indexPrecision=FUSEDPQ \
  -wi 0 -i 1 -f 1
```

**Step 2: compact only** (dataset not loaded; use a small heap to measure the compactor's footprint)

```bash
java -Xmx5g --add-modules jdk.incubator.vector \
  -cp benchmarks-jmh/target/compactor-benchmark.jar \
  io.github.jbellis.jvector.bench.CompactorBenchmark \
  -p workloadMode=COMPACT \
  -p measureRecall=false \
  -p datasetNames=<dataset> \
  -p numPartitions=4 \
  -p splitDistribution=UNIFORM \
  -p indexPrecision=FUSEDPQ \
  -wi 0 -i 1 -f 1
```

`COMPACT` with `measureRecall=false` skips dataset loading entirely, so `-Xmx5g` is sufficient even for large datasets. This confirms that the compactor itself — not the dataset — drives heap usage.

Key `workloadMode` values:

| Mode | Description |
|---|---|
| `PARTITION_AND_COMPACT` | **(default)** Build partitions, compact them |
| `PARTITION` | Build N partition indexes and exit; use before `COMPACT` |
| `COMPACT` | Compact existing partitions |
| `BUILD` | Build one index over the full dataset |

Results are written as JSONL to `target/benchmark-results/compactor-*/compactor-results.jsonl`. The `durationMs` field records only the compaction time (not dataset loading or JVM startup). When `measureRecall=true`, each result also includes `recall`, `avgSearchLatencyMs`, and `p99SearchLatencyMs` from searching the compacted graph.

Comparison against build-from-scratch (results averaged over three runs).

- Build from scratch: build with PQ, search using FusedPQ with FP reranking.
- Compaction: build source partitions with PQ, compact using FusedPQ with FP rescoring, search using FusedPQ with FP reranking. Source partitions are based on a Fibonacci distribution with 4 partitions.

| Dataset               | Dim  | Build from Scratch | Compaction | Delta  |                                                                                
  |-----------------------|-----:|-------------------:|-----------:|-------:|
| cap-6M                |  768 |              0.626 |      0.619 | -0.008 |                                                                                
| cap-1M                |  768 |              0.656 |      0.656 |  0.000 |                                                                                
| gecko-100k            |  768 |              0.690 |      0.701 | +0.011 |                                                                                
| e5-small-v2-100k      |  384 |              0.572 |      0.586 | +0.014 |                                                                                
| ada002-1M             | 1536 |              0.687 |      0.703 | +0.016 |                                                                                
| e5-base-v2-100k       |  768 |              0.676 |      0.692 | +0.016 |                                                                                
| cohere-english-v3-10M | 1024 |              0.544 |      0.561 | +0.017 |                                                                                
| e5-large-v2-100k      | 1024 |              0.686 |      0.703 | +0.017 |                                                                                
| ada002-100k           | 1536 |              0.751 |      0.769 | +0.018 |                                                                                
| cohere-english-v3-1M  | 1024 |              0.593 |      0.612 | +0.019 |    

# Heap memory footprint

All datasets above can be compacted under `COMPACT` with `measureRecall=false` and `-Xmx5g`. In addition, compaction successfully scales to a dataset with 2560 dimensions and 10M vectors under the same heap constraint.


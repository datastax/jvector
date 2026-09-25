# Graph Index Compaction

`OnDiskGraphIndexCompactor` merges multiple on-disk HNSW graph indexes into a single compacted index. This is useful in write-heavy workloads where data is continuously ingested into small segment indexes that accumulate over time; periodically compacting those segments into one larger index improves search throughput and recall without rebuilding from scratch.

## Overview

```
source[0].index  ─┐
source[1].index  ─┤──► OnDiskGraphIndexCompactor ──► compacted.index
source[N].index  ─┘
```

Each source is an `OnDiskGraphIndex` with an associated `FixedBitSet` marking which of its nodes are live (not deleted). The compactor merges all live nodes into a single graph, assigns the output ordinals (dense over the live nodes, grouped by region), and retrains the Product Quantization codebook for outputs that carry one.

## Usage

```java
List<OnDiskGraphIndex> sources = List.of(index0, index1, index2);

// Mark all nodes live (no deletions)
List<FixedBitSet> liveNodes = sources.stream()
    .map(s -> { var bs = new FixedBitSet(s.size()); bs.set(0, s.size()); return bs; })
    .collect(toList());

var compactor = new OnDiskGraphIndexCompactor(
    sources, liveNodes,
    VectorSimilarityFunction.COSINE,
    /* executor= */ null                  // null = use by default shared threadpool in compactor
);

compactor.compact(Path.of("compacted.index"));

// Where every source node landed, one mapper per source: oldToNew(i) is the output ordinal of
// source node i, or OrdinalMapper.OMITTED for a deleted node.
List<OrdinalMapper> mapping = compactor.ordinalMappers();
```

### Output ordinals

The compactor assigns the output ordinals; the caller does not choose them. Live nodes are numbered `0 .. live - 1`, sources in ascending-size order, and within a source by the region of the largest source's hierarchy each node descends to, so nodes that are close in vector space get adjacent ordinals: records are written sequentially, every cell's members are contiguous (the cell join below relies on this), and consecutive nodes explore the same neighbourhood of every target. Callers that keep external references to nodes (a row id to ordinal table, for instance) translate them through `ordinalMappers()` after `compact` returns.

### Handling Deleted Nodes

Deleted nodes are excluded from the output by marking them as `false` in the corresponding `FixedBitSet`; they get no output ordinal (`ordinalMappers()` reports `OrdinalMapper.OMITTED` for them).

```java
// Example: every 5th node is deleted
FixedBitSet live = new FixedBitSet(source.size());
for (int i = 0; i < source.size(); i++) {
    if (i % 5 != 0) {
        live.set(i);
    }
}
```

## Algorithm

### Ordinal assignment

Each source assigns its own local ordinals. The compactor builds the output mapping at the start of `compact`: sources in ascending-size order, and within a source by the level-1 node of the largest source's hierarchy each node descends to (an exact greedy descent through a resident copy of that source's upper layers, refined by a width-8 beam over level 1, keyed by a breadth-first walk position over the level-1 graph). When the largest source has no hierarchy there are no cells, and level 0 falls back to the graph search.


### PQ Retraining

If the sources carry PQ (FusedPQ, or a compressed sidecar), the compactor retrains the Product Quantization codebook on the combined dataset and re-encodes every live node for the output. This is done by `PQRetrainer`, which
performs **balanced proportional sampling** across all sources (up to `ProductQuantization.MAX_PQ_TRAINING_SET_SIZE` vectors total, at least 1000 per source). The merge itself does not run on these codes: whatever the sources carry, it runs on the vectors (see "The merge store" below), so the input type determines the output format and nothing else.


### Neighbor Selection (per node)

For each live node at each graph level, the compactor gathers a candidate neighbor pool and then applies diversity selection:

**1. Gather from same source** (`gatherFromSameSource`)\
Iterate the node's existing neighbors in its source index. Filter out deleted nodes. Score each with the similarity function. No graph search — neighbors are already precomputed.

**2. Gather from other sources** (`gatherFromOtherSource`)\
Keep the top `searchTopK` candidates per larger target, scored exactly: at level 0 through the cell join described below, at the upper levels through a graph search of the target. The graph search also stands in at level 0 for a node without a cell; when the target carries PQ codes the traversal scores through them and the top candidates are rescored exactly, otherwise the traversal is exact.

```
searchTopK  = max(2,  ceil(degree / numSources) * 4)
```

- *Level 0* uses pair-asymmetric cross-linking. Sources are processed smallest first, one source at a time. A node gathers candidates only from the sources **larger** than its own, through the **cell join**. Every candidate is also *offered back* to the node it found, with its exact score: since similarity is symmetric, the offer is exactly the candidate that node's own scan of the smaller source would have produced. Each node holds up to 16 offer slots, kept in a banded, spill-to-disk buffer so peak memory is independent of node count. When a source's turn comes, its nodes union the offers they received with their retained same-source edges and their own candidates before diversity selection. The largest source gathers nothing itself; a node of it that received no offers keeps its retained edges unchanged and skips selection entirely.
- *Level L > 0*: the compactor first descends greedily from the source's entry point through each level above L (one `searchOneLayer` call with topK=1 per level, feeding the result into the next via `setEntryPointsFromPreviousLayer()`), then performs the full beam search at level L. This mirrors standard HNSW construction and gives a much better starting point than jumping directly to level L from the global entry node.


**3. Diversity selection** (Vamana-style)\
Candidates are sorted by exact score (descending). The compactor selects up to `maxDegree` diverse neighbors using an adaptive alpha. At level 0 the pairwise comparisons between a candidate and the already-selected neighbours run on vectors decoded from the wide code (below), and the candidate's pruning threshold is recomputed against the node from the same decoded vector, so both sides of the inequality carry the same quantization bias; the ordering itself always uses the exact scores.

```
for alpha in [1.0, 1.2]:
    for each candidate c (highest score first):
        if c is already selected: skip
        if ∀ selected neighbor j: similarity(c, j) ≤ score(c) × alpha:
            select c
    if |selected| == maxDegree: stop
```

### Cell join

With a hierarchy in the largest source, level-0 candidates come from a scan rather than a search. The output ordinals group every source's nodes by the level-1 node of the largest source they descend to, their *cell*; each source's nodes of a cell form a contiguous ordinal range, so their vectors are contiguous in the merge store (the inverted lists of an IVF index whose centroids are the largest source's level-1 nodes). Level 0 is processed in ranges of consecutive ordinals: for each node of a range a width-16 beam over the level-1 graph, seeded at the node's own cell, picks its probe cells (at most 4,096 members per larger source, best cells first); the range's (cell, node) pairs are then inverted so that each probed cell of each larger source is streamed once and every vector in it is scored against all the range's nodes that probe it (eight queries per pass, the queries kept in L1), keeping each node's best `searchTopK` by exact score. Survivors enter the unchanged pipeline (reverse offers, diversity, write). Nodes without a cell, and merges where the largest source has no hierarchy, use the graph search.

### The merge store

Before level 0 is written, every live vector is copied once into a scratch region of the output file at its output ordinal (`4 × dimension` bytes per node; truncated away when the merge finishes). The cell scan streams it, and survivors, retained edges and offered candidates take their vectors from it, so no source record is read for a candidate during level 0 and every score the output is written from is exact. Outputs that carry PQ (FusedPQ, or a sidecar) get a second cache from the same pass holding the retrained codes by output ordinal: record writes copy neighbour codes from it instead of re-encoding per edge, and for sidecar sources (`compact(graphPath, compressedPath)`) it becomes the merged compressed vectors file.

### Wide code

The compactor also trains a product quantization with four dimensions per subspace (`max(8, dimension / 4)` subspaces of 256 centroids: 192 bytes per node at 768 dimensions) and encodes every live node into a third cache in the same pass. It serves one purpose: the pairwise diversity comparisons at level 0, where a candidate's vector is decoded from its code rather than read from the merge store, so a node's selection touches one row per candidate instead of one per (candidate, neighbour) pair. Candidate scores against the node stay exact. The coarser code also collapses near-duplicates onto one code, which the diversity rule then prunes, so the degree is spent on distinct neighbours; a finer two-dimensional code kept them apart and merged to lower recall.

### Hierarchical Levels

Level 0 (base layer) stores inline vectors, FusedPQ codes, and the neighbor list. Upper levels store only the neighbor list (plus PQ codes at level 1 for cross-level searching).

Processing is batched per source and run in parallel across sources using a `ForkJoinPool`. A backpressure window keeps at most `taskWindowSize` batches in-flight at once, bounding memory use.

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


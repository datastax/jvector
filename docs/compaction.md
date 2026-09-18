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
    /* executor= */ null                  // null = use by default shared threadpool in compactor
);

compactor.compact(Path.of("compacted.index"));
```

### Letting the compactor choose ordinals

By default the output ordinals are exactly the ones the caller's `OrdinalMapper`s assign. Callers that do not need a particular numbering can hand that choice to the compactor:

```java
compactor.setReassignOrdinals(true);
compactor.compact(Path.of("compacted.index"));

// The mapping actually used, per source: translate through it, not through the mappers passed in.
List<OrdinalMapper> effective = compactor.effectiveRemappers();
```

The compactor then numbers each source's live nodes by locality (nodes that are close in vector space get adjacent ordinals). Records are written sequentially, similar vectors land in adjacent records, and consecutive cross-source searches walk the same neighbourhood of every target, which is where most of the merge time goes on large inputs. The caller's mappers are still used to enumerate nodes; only their ordinal values are replaced. Sources with PQ codes (fused, or a compressed sidecar) use the retrained codebook; sources without any PQ get a merge-time codebook trained on a balanced sample (`dimension / 8` subspaces, at least 8), used only during the merge: nothing quantized is written to the output.

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

## Algorithm

### Ordinal Remapping

Each source assigns its own local ordinals. The compactor maps them to a new global ordinal space using the user-provided `OrdinalMapper`s, or, with `setReassignOrdinals(true)`, a mapping of its own: sources in ascending-size order, and within a source by the region of the largest source's hierarchy each node descends to (a code-scored greedy descent through a resident copy of that source's upper layers, keyed by a breadth-first walk position over its level-1 graph). When the largest source has no hierarchy the nodes are ordered by PQ-code prefix instead.


### PQ Retraining

If the source indexes use FusedPQ, the compactor retrains the Product Quantization codebook on the combined dataset before writing the output. This is done by `PQRetrainer`, which
performs **balanced proportional sampling** across all sources (up to `ProductQuantization.MAX_PQ_TRAINING_SET_SIZE` vectors total, at least 1000 per source).

Sources without any quantization (full-precision graphs) are merged without any: when the compactor assigns the ordinals, the scratch region holds the vectors themselves in merged order, the resident copy of the largest source's upper layers holds their vectors, and every comparison of the merge — ordinal assignment, probe selection, the cell scan, candidate scoring, offers and diversity — is exact. Nothing is trained and nothing quantized is written. See "Full-precision sources" below.


### Neighbor Selection (per node)

For each live node at each graph level, the compactor gathers a candidate neighbor pool and then applies diversity selection:

**1. Gather from same source** (`gatherFromSameSource`)\
Iterate the node's existing neighbors in its source index. Filter out deleted nodes. Score each with the similarity function. No graph search — neighbors are already precomputed.

**2. Gather from other sources** (`gatherFromOtherSource`)\
Run a graph search in other source indexes and keep the top `searchTopK` hits per target. When PQ codes are available (fused, or a compressed sidecar) the traversal scores through a per-query lookup table over the target's codes and the top candidates are rescored exactly; otherwise the traversal is exact.

```
searchTopK  = max(2,  ceil(degree / numSources) * 4)
```

- *Level 0* uses pair-asymmetric cross-linking, and, when the compactor assigns the ordinals, finds each node's candidates in a larger source by the **cell join** described below instead of a graph search. Sources are processed smallest first, one source at a time. A node searches only the sources **larger** than its own (a full hierarchical `GraphSearcher.search()` from the target's entry point). Every hit is also *offered back* to the node it found, with the exact score the searcher computed: since similarity is symmetric, the offer is exactly the candidate that node's own search of the smaller source would have produced. Each node holds up to 16 offer slots, kept in a banded, spill-to-disk buffer so peak memory is independent of node count. When a source's turn comes, its nodes union the offers they received with their retained same-source edges and their own forward-search results before diversity selection. The largest source runs no searches at all; a node of it that received no offers keeps its retained edges unchanged and skips selection entirely.
- *Level L > 0*: the compactor first descends greedily from the source's entry point through each level above L (one `searchOneLayer` call with topK=1 per level, feeding the result into the next via `setEntryPointsFromPreviousLayer()`), then performs the full beam search at level L. This mirrors standard HNSW construction and gives a much better starting point than jumping directly to level L from the global entry node.


**3. Diversity selection** (Vamana-style)\
Candidates are sorted by score (descending). The compactor selects up to `maxDegree` diverse neighbors using an adaptive alpha. Candidates that arrived as offers are compared with already-selected neighbours through their PQ codes (a symmetric code-to-code similarity built once per compaction from the retrained codebook, for dot product, Euclidean and cosine), so no offerer's vector is read during selection.

```
for alpha in [1.0, 1.2]:
    for each candidate c (highest score first):
        if c is already selected: skip
        if ∀ selected neighbor j: similarity(c, j) ≤ score(c) × alpha:
            select c
    if |selected| == maxDegree: stop
```

### Cell join

With `setReassignOrdinals(true)` and a hierarchy in the largest source, the level-0 cross-source search is replaced by a scan. The reassigned ordinals group every source's nodes by the level-1 node of the largest source they descend to, their *cell* (the best-scoring level-1 node found by a small beam from the greedy descent's landing); each source's nodes of a cell form a contiguous ordinal range, and their codes are contiguous in the pre-encoded cache, which is stored in 64-code, subspace-major blocks for this purpose. For a node, a beam over the level-1 graph seeded at its own cell picks the 16 best cells; for each larger source the node's 8-bit lookup table is applied to that source's codes in those cells (in score order, at most 4,096 codes), the top `searchTopK` by table score are rescored on vectors decoded from the wide code (below; on full-precision sources the scan itself is exact, see "Full-precision sources"), and they enter the unchanged pipeline (reverse offers, diversity, write). The scan runs through a Google Highway kernel (`pq_scan_blocked_u8`) when the native library is available, otherwise through a plain Java loop. Nodes without a cell, and merges where the largest source has no hierarchy, use the graph search.

### Pre-encoded codes

Before level 0 is written, every live node is encoded once against the retrained codebook into a memory-mapped code cache indexed by new ordinal. Record writes copy neighbour codes from the cache instead of re-encoding them per edge, the cross-source searches score through it, and the offer diversity checks read it. For sidecar sources (`compact(graphPath, compressedPath)`) the same cache also becomes the merged compressed vectors file.

### Wide code

When the cell join is active the compactor also trains a second, finer product quantization — two dimensions per subspace, 256 centroids, i.e. 192 bytes per node at 384 dimensions — and encodes every live node into a second scratch cache in the same pass as the pre-encoded codes. At level 0 the vector of a candidate (a scanned candidate that survived the table ranking, or one of the node's retained edges) is decoded from this code instead of being read from a record: each two-dimensional centroid is one packed `long`, so a decode is one table read per subspace plus the global centroid. Scoring against the node's exact vector and the diversity checks run on the decoded vectors, which are near-exact for this purpose (merged recall within 0.2 pt of exact reranks at 3×8M). No record other than the node's own is read during level 0. The second cache is truncated together with the code cache.

### Full-precision sources

When no source carries codes, the cell join runs on vectors end to end. The pre-encode pass writes each live vector into the scratch region at its merged ordinal, so every cell's vectors are contiguous (the inverted lists of an IVF index). Level 0 is processed in batches of 1,024 consecutive ordinals: for each node of a batch the probe beam picks its cells exactly, then, per larger source, each probed cell is streamed once and every vector in it is scored against all the batch's nodes that probe it (eight queries per pass, the queries kept in L1), keeping each node's best `searchTopK` by exact score. Survivors, retained edges and offered candidates take their vectors from the scratch, so records are never read for candidates and the output is written from exact scores throughout. The wide code is not built in this mode.

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


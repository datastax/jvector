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
    /* executor= */ null                  // null = create internal ForkJoinPool
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
searchTopK  = max(2,  ceil(degree / numSources) * 2)
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

Processing is batched per source and run in parallel across sources using a `ForkJoinPool`. A backpressure window keeps at most `taskWindowSize` batches in-flight at once, bounding memory use.

### Entry Node

The entry node of the compacted graph is:
1. The original entry node of `sources[0]`, if it is live.
2. Otherwise, the first live node found by scanning all sources in order.

## Benchmarking

Use `CompactorBenchmark` (in `benchmarks-jmh`) to measure compaction performance. See `benchmarks-jmh/src/main/java/io/github/jbellis/jvector/bench/CompactorBenchmark.md` for full instructions.

### Default: partition and compact in one run

```bash
java -Xmx220g --add-modules jdk.incubator.vector \
  -jar benchmarks-jmh/target/benchmarks-jmh-*.jar CompactorBenchmark \
  -p workloadMode=PARTITION_AND_COMPACT \
  -p datasetNames=<dataset> \
  -p numPartitions=4 \
  -p splitDistribution=FIBONACCI \
  -p indexPrecision=FUSEDPQ \
  -wi 0 -i 1 -f 1
```

### Measuring peak heap during compaction

To measure how little RAM compaction actually needs — without the dataset occupying heap — run the two steps separately.

**Step 1: build partitions** (dataset in memory, large heap required)

```bash
java -Xmx220g --add-modules jdk.incubator.vector \
  -jar benchmarks-jmh/target/benchmarks-jmh-*.jar CompactorBenchmark \
  -p workloadMode=PARTITION_ONLY \
  -p datasetNames=<dataset> \
  -p numPartitions=4 \
  -p splitDistribution=FIBONACCI \
  -p indexPrecision=FUSEDPQ \
  -wi 0 -i 1 -f 1
```

**Step 2: compact only** (dataset not loaded; use a small heap to prove low-memory operation)

```bash
java -Xmx5g --add-modules jdk.incubator.vector \
  -jar benchmarks-jmh/target/benchmarks-jmh-*.jar CompactorBenchmark \
  -p workloadMode=COMPACT_ONLY \
  -p datasetNames=<dataset> \
  -p numPartitions=4 \
  -p splitDistribution=FIBONACCI \
  -p indexPrecision=FUSEDPQ \
  -wi 0 -i 1 -f 1
```

`COMPACT_ONLY` skips dataset loading entirely, so `-Xmx5g` is sufficient even for large datasets. This lets you confirm that the compactor itself — not the dataset — is the memory bottleneck.

Key `workloadMode` values:

| Mode | Description |
|---|---|
| `PARTITION_AND_COMPACT` | **(default)** Build partitions, compact them, then measure recall |
| `PARTITION_ONLY` | Build N partition indexes and exit; use before `COMPACT_ONLY` |
| `COMPACT_ONLY` | Compact existing partitions without loading the dataset; `durationMs` = `compact()` time |
| `BUILD_FROM_SCRATCH` | Build one index over the full dataset; `durationMs` = `build()` time |

Results are written as JSONL to `target/benchmark-results/compactor-*/compactor-results.jsonl`. The `durationMs` field records only the target function time (not dataset loading or JVM startup).

## Recall 


The table compares building from scratch with compaction under the following configurations (results averaged over three runs):

- Build from scratch: build with PQ; search using FusedPQ with FP reranking.
- Compaction: build source partitions with PQ; compact using FusedPQ with FP rescoring; search using FusedPQ with FP reranking.

| Dataset              | Dim  | Build from Scratch | Compaction |  Delta |
|----------------------|-----:|-------------------:|-----------:|-------:|
| cap-6M               |  768 |              0.626 |      0.619 | -0.008 |
| cap-1M               |  768 |              0.656 |      0.656 |  0.000 |
| gecko-100k           |  768 |              0.690 |      0.701 | +0.011 |
| e5-small-v2-100k     |  384 |              0.572 |      0.586 | +0.014 |
| ada002-1M            | 1536 |              0.687 |      0.703 | +0.016 |
| e5-base-v2-100k      |  768 |              0.676 |      0.692 | +0.016 |
| cohere-english-v3-10M | 1024 |              0.544 |      0.561 | +0.017 |
| e5-large-v2-100k     | 1024 |              0.686 |      0.703 | +0.017 |
| ada002-100k          | 1536 |              0.751 |      0.769 | +0.018 |
| cohere-english-v3-1M | 1024 |              0.593 |      0.612 | +0.019 |

# Memory footprint

All datasets above can be compacted under `COMPACT_ONLY` with `-Xmx5g`. In addition, compaction successfully scales to a dataset with 384 dimensions and 50M vectors under the same memory constraint.


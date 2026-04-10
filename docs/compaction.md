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

Each source assigns its own local ordinals starting from 0. The compactor maps them to a new global ordinal space using `OrdinalMapper`:

| Class | Behavior |
|---|---|
| `IdentityMapper` | Ordinals unchanged |
| `OffsetMapper` | `newOrdinal = oldOrdinal + offset` |
| `MapMapper` | Arbitrary `oldOrdinal → newOrdinal` map |

### PQ Retraining

If the source indexes use FusedPQ, the compactor retrains the Product Quantization codebook on the combined dataset before writing the output. This is done by `PQRetrainer`, which:

1. Performs **balanced proportional sampling** across all sources (up to `ProductQuantization.MAX_PQ_TRAINING_SET_SIZE` vectors total, at least 1000 per source).
2. Sorts samples by `(source, node)` so disk reads are sequential, enabling OS read-ahead.
3. Uses a `ThreadLocal<View[]>` in `SampledRAVV` so each worker thread reuses its file handles instead of opening one per sample read.
4. Calls `ProductQuantization.compute()` on the samples to produce a new codebook.

Chunked sampling reduces page faults: instead of picking random individual nodes, nodes are grouped into 32-node chunks and the chunk order is shuffled (Fisher-Yates). This cuts random jump frequency by ~32× while preserving statistical randomness.

### Neighbor Selection (per node)

For each live node at each graph level, the compactor gathers a candidate neighbor pool and then applies diversity selection:

**1. Gather from same source** (`gatherFromSameSource`)\
Iterate the node's existing neighbors in its source index. Filter out deleted nodes. Score each with the similarity function. No graph search — neighbors are already precomputed.

**2. Gather from other sources** (`gatherFromOtherSource`)\
Run a graph search in every other source index starting from that source's entry point. At level 0, a full KNN search is used. At upper levels, a single-layer search is run. If FusedPQ is available, the search uses approximate PQ scoring and rescores the top results exactly.

```
searchTopK  = max(2,  ceil(degree / numSources) * 2)
beamWidth   = max(degree, searchTopK) * 2
```

**3. Diversity selection** (Vamana-style)\
Candidates are sorted by score (descending). The compactor selects up to `maxDegree` diverse neighbors using an adaptive alpha:

```
alpha ← 1.0
for each candidate c (highest score first):
    if ∀ selected neighbor j: similarity(c, j) ≤ score(c) × alpha:
        select c
    if |selected| < maxDegree and more candidates remain:
        alpha += 0.2  (up to 1.2)
```

Starting at alpha=1.0 ensures the nearest neighbors are preferred; the gradual increase admits diverse candidates when necessary to fill the degree budget.

### Hierarchical Levels

Level 0 (base layer) stores inline vectors, FusedPQ codes, and the neighbor list. Upper levels store only the neighbor list (plus PQ codes at level 1 for cross-level searching).

Processing is batched per source and run in parallel across sources using a `ForkJoinPool`. A backpressure window keeps at most `taskWindowSize` batches in-flight at once, bounding memory use.

### Entry Node

The entry node of the compacted graph is:
1. The original entry node of `sources[0]`, if it is live.
2. Otherwise, the first live node found by scanning all sources in order.

## Design Notes

**Beam width scales with degree, not a fixed floor.** An earlier implementation enforced `MIN_BEAM_WIDTH = 100`, which was wasteful for small graphs and could be insufficient for large ones. The current formula ties beam width to the graph's actual degree configuration.

**Scratch space is per-thread.** Each worker thread owns a `Scratch` object with pre-allocated candidate arrays, vector buffers, and `GraphSearcher` instances (one per source). This avoids allocations in the hot path and keeps thread contention low.

**Sorted samples enable sequential I/O.** Sorting `SampleRef` entries by `(source, node)` before PQ training means `extractTrainingVectors` (which runs in a parallel stream) accesses each source's mmap'd file in ascending node order, letting the OS prefetcher cover the data.

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

- Build from scratch: Build with PQ; search using FusedPQ with FP reranking.
- Compaction: Build source indices with PQ; compact using FusedPQ with FP rescoring; search using FusedPQ with FP reranking.

| Dataset                     | Build from Scratch | Compaction | Delta  |
|----------------------------|------------------:|-----------:|-------:|
| openai-v3-small-100k       | 0.781             | 0.778      | -0.003 |
| openai-v3-large-3072-100k  | 0.824             | 0.825      | +0.001 |
| openai-v3-large-1536-100k  | 0.796             | 0.797      | +0.001 |
| e5-small-v2-100k           | 0.472             | 0.504      | +0.032 |
| e5-base-v2-100k            | 0.622             | 0.631      | +0.009 |
| e5-large-v2-100k           | 0.560             | 0.618      | +0.058 |
| glove-25-angular           | 0.069             | 0.060      | -0.009 |
| glove-50-angular           | 0.153             | 0.121      | -0.032 |
| glove-100-angular          | 0.179             | 0.143      | -0.036 |
| glove-200-angular          | 0.178             | 0.154      | -0.024 |
| lastfm-64-dot              | 0.189             | 0.151      | -0.038 |
| ada002-100k                | 0.687             | 0.714      | +0.027 |
| colbert-1M                 | 0.385             | 0.318      | -0.067 |
| gecko-100k                 | 0.610             | 0.635      | +0.025 |
| nytimes-256-angular        | 0.342             | 0.328      | -0.014 |
| sift-128-euclidean         | 0.479             | 0.447      | -0.032 |
| cap-1M                     | 0.658             | 0.645      | -0.013 |
| cap-6M                     | 0.630             | 0.607      | -0.023 |
| cohere-english-v3-100k     | 0.743             | 0.738      | -0.005 |
| cohere-english-v3-1M       | 0.596             | 0.593      | -0.003 |
| cohere-english-v3-10M      | 0.545             | 0.539      | -0.006 |
| dpr-1M                     | 0.238             | 0.219      | -0.019 |
| dpr-10M                    | 0.165             | 0.147      | -0.018 |

One noticeable drop is on colbert-1M, which shows a ~6% decrease in recall. The issue is still under investigation.

# Memory footprint

All datasets above can be executed with ```-Xmx=5G```. In addition, compaction successfully scales to a dataset with 384 dimensions and 50M vectors (e.g., IBM DataPile) under the same memory constraint.


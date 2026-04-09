<!--
  ~ Copyright DataStax, Inc.
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~ http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
  -->

# CompactorBenchmark

`CompactorBenchmark` evaluates the **performance, memory usage, and recall quality** of graph index compaction using `OnDiskGraphIndexCompactor`.

The benchmark compares compaction against building the index from scratch and supports multiple configurations of compaction precision and compression resources.

---

# 1. Benchmark Goals

The benchmark focuses on validating that compaction can:

- Run with **significantly lower memory footprint**
- Produce results comparable to **building from scratch**
- Support **compressed compaction workflows**
- Generate optional **PQ artifacts**

Key metrics include:

- **Peak RSS memory**
- **Compaction runtime**
- **Recall@K**
- **I/O throughput**

---

# 2. Workload Modes

The benchmark supports three workload modes.

## BUILD_FROM_SCRATCH

Build the index directly from the dataset.

```
dataset → build graph → output index
```

Purpose:

- Baseline runtime
- Baseline memory usage
- Reference recall

Example:

```
-p workloadMode=BUILD_FROM_SCRATCH
```

---

## SEGMENTS_AND_COMPACT

Simulates production ingestion by building multiple segments first.

```
dataset
 ├─ segment1
 ├─ segment2
 ├─ ...
 └─ segmentN

segments → compaction → final index
```

Purpose:

- Evaluate compaction behavior
- Measure memory savings

Example:

```
-p workloadMode=SEGMENTS_AND_COMPACT
```

---

## COMPACT_ONLY

Compacts already-built segments.

```
existing segments → compaction → final index
```

Purpose:

- Isolate compaction performance
- Debug compaction algorithms

Example:

```
-p workloadMode=COMPACT_ONLY
```

---

# 3. Compaction Modes

`CompactMode` defines how compaction behaves.

```java
enum CompactMode {
    INLINE,
    PQ_VECTORS_OUTPUT,
    FUSEDPQ_FROM_SOURCES,
    FUSEDPQ_FROM_PQVECTORS
}
```

Each mode corresponds to a specific `CompactOptions` configuration.

---

# 4. Mode Descriptions

## INLINE

Exact compaction using full vectors.

Configuration:

```java
CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS))
    .precision(EXACT)
    .compressionConfig(CompressionConfig.none())
    .build();
```

Properties:

- Highest recall
- Highest memory usage
- No compression involved

---

## PQ_VECTORS_OUTPUT

Exact compaction while generating PQVectors.

Configuration:

```java
CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS))
    .precision(EXACT)
    .compressionConfig(
        CompressionConfig.withPQCodebook(pq, pqVectorsOutputPath)
    )
    .build();
```

Behavior:

- Compaction uses full vectors
- PQVectors are encoded during compaction
- PQVectors are written to a **separate file**

Example output:

```
compacted_index
compacted_index.pq
```

---

## FUSEDPQ_FROM_SOURCES

Compressed compaction using PQ resources stored in source segments.

Configuration:

```java
CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS, FUSED_PQ))
    .precision(COMPRESSED)
    .compressionConfig(
        CompressionConfig.withSourcePQ(PQSourcePolicy.AUTO)
    )
    .build();
```

Requirements:

- Source segments must contain `FUSED_PQ`

Behavior:

- Compaction uses compressed scoring
- PQ information is reused from source indexes
- Output index contains `FUSED_PQ`

Advantages:

- Much lower memory footprint
- Faster cross-segment search

---

## FUSEDPQ_FROM_PQVECTORS

Compressed compaction using externally provided PQVectors.

Configuration:

```java
CompactOptions.builder()
    .writeFeatures(EnumSet.of(INLINE_VECTORS, FUSED_PQ))
    .precision(COMPRESSED)
    .compressionConfig(
        CompressionConfig.withPQVectors(pqVectors)
    )
    .build();
```

Requirements:

```
pqVectorsInputPath
```

Behavior:

- Compressed search uses caller-provided PQVectors
- Output index contains `FUSED_PQ`
- Does not depend on source PQ features

Advantages:

- Enables compressed compaction even if sources contain only inline vectors
- Allows global PQ models shared across indexes

---

# 5. PQ Parameters

The benchmark distinguishes three PQ-related parameters.

## PQ Codebook

```
pqPath
```

Used to load a trained PQ model.

Alternatively PQ can be trained during the benchmark:

```
-p trainPQ=true
```

---

## PQVectors Input

```
pqVectorsInputPath
```

Used only for:

```
FUSEDPQ_FROM_PQVECTORS
```

This file contains precomputed PQ vectors used during compressed compaction.

---

## PQVectors Output

```
pqVectorsOutputPath
```

Used only for:

```
PQ_VECTORS_OUTPUT
```

This file stores PQ vectors generated during compaction.

If not provided, the benchmark writes:

```
<compactOutput>.pq
```

---

# 6. Index Precision

Segment indexes can be built with two precision modes.

```
enum IndexPrecision {
    FULLPRECISION,
    FUSEDPQ
}
```

### FULLPRECISION

Segments contain:

```
INLINE_VECTORS
```

### FUSEDPQ

Segments contain:

```
INLINE_VECTORS + FUSED_PQ
```

This enables compressed compaction from sources.

---

# 7. Recall Measurement

Recall evaluation can be enabled with:

```
-p enableRecall=true
```

During recall evaluation:

- INLINE mode uses **exact scoring**
- PQ modes use **approximate scoring**

Search compares results against the dataset ground truth.

Metrics reported:

- Recall@10
- Query latency

---

# 8. Important Benchmark Parameters

| Parameter | Description |
|-----------|-------------|
| datasetNames | Dataset name |
| datasetPortion | Fraction of dataset used |
| numSegments | Number of segments |
| graphDegree | Graph connectivity |
| beamWidth | Graph construction beam |
| splitDistribution | Data partitioning strategy |
| parallelWriteThreads | Compaction parallelism |
| vectorizationProvider | SIMD backend |
| enableRecall | Enable recall evaluation |

---

# 9. Memory Measurement

The benchmark tracks:

- **Peak RSS**
- **Heap usage**
- **I/O throughput**

Peak RSS is the primary metric because compaction targets **machines with limited memory**.

---

# 10. Typical Workflows

## Baseline Build

```
BUILD_FROM_SCRATCH
indexPrecision=FULLPRECISION
```

Purpose:

- Establish baseline recall and runtime.

---

## Segment Compaction

```
SEGMENTS_AND_COMPACT
compactMode=INLINE
```

Purpose:

- Compare compaction vs build-from-scratch.

---

## Low-Memory Compaction

```
SEGMENTS_AND_COMPACT
indexPrecision=FUSEDPQ
compactMode=FUSEDPQ_FROM_SOURCES
```

Purpose:

- Measure compressed compaction performance.

---

## External PQVectors Compaction

```
SEGMENTS_AND_COMPACT
compactMode=FUSEDPQ_FROM_PQVECTORS
pqVectorsInputPath=<path>
```

Purpose:

- Test compressed compaction using externally provided PQ vectors.

---

# Summary

`CompactorBenchmark` evaluates multiple compaction strategies:

| Mode | Precision | PQ Source |
|-----|-----|-----|
| INLINE | Exact | None |
| PQ_VECTORS_OUTPUT | Exact | Provided PQ codebook |
| FUSEDPQ_FROM_SOURCES | Compressed | Source indexes |
| FUSEDPQ_FROM_PQVECTORS | Compressed | Caller-provided PQVectors |

This framework enables comprehensive evaluation of compaction behavior across different memory, performance, and compression scenarios.

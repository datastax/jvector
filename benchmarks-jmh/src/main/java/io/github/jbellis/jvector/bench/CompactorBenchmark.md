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

# Compactor Benchmark

The `CompactorBenchmark` measures the performance and search quality of different **index construction pipelines** for `OnDiskGraphIndex`.

It supports benchmarking:

- Building an index **from scratch**
- Building **segments and compacting them**
- **Compaction only** on prebuilt segments

It also evaluates different **vector storage strategies** used during compaction.

---

# TLDR

Run with defaults:

```bash
mvn compile exec:exec@compactor -pl benchmarks-jmh -am
```

---

# Overview

The benchmark can run several indexing pipelines.

## 1. Build From Scratch

Build the index directly from the dataset.

```
dataset
   │
   ▼
graph build
```

This acts as the **baseline** for comparison.

---

## 2. Segment Build + Compaction

Split the dataset into segments and then merge them.

```
dataset
   │
   ▼
segment build (N segments)
   │
   ▼
compaction
   │
   ▼
final index
```

This simulates **production ingestion pipelines** where data arrives incrementally.

---

## 3. Compaction Only

Use prebuilt segments and benchmark only the compaction phase.

```
segments
   │
   ▼
compaction
   │
   ▼
final index
```

This isolates the **compaction algorithm performance**.

---

# Compaction Modes

Compaction supports three vector storage strategies.

---

## INLINE

Vectors are stored inline in the graph.

```
Graph Node
 ├── neighbors
 └── full vector
```

---

## PQ_SEPARATE

Vectors are stored in a separate **PQVectors file**.

```
Graph Node
 └── neighbors

PQVectors
 └── compressed vectors
```

Search uses PQ decoding during scoring.

PQ can either be:

- loaded from `pqPath`
- trained from the dataset (`trainPQ=true`)

---

## FUSED_PQ

Neighbor PQ codes are fused directly into the graph.

```
Graph Node
 ├── neighbors
 └── PQ codes
```

Graph building uses **ADC scoring directly from the graph**.

- requires FusedPQ in source segments

---

# Recall Evaluation

Recall evaluation depends on the compaction mode.

| Mode | Vector source used during search |
|-----|--------------------------------|
| INLINE | Full precision vectors |
| PQ_SEPARATE | `PQVectors` |
| FUSED_PQ | fused PQ codes inside graph |

For `PQ_SEPARATE`, the benchmark loads the compressed vectors and computes scores using:

```
PQVectors.scoreFunctionFor(query)
```

This uses PQ codebooks to compute approximate similarity.

---

# Parameters

Parameters can be supplied via:

```
-Dargs="-p <param>=<value>"
```

Example:

```
-Dargs="-p datasetNames=glove-100-angular"
```

---

# Core Parameters

| Parameter | Default | Description |
|---|---|---|
| `datasetNames` | `glove-100-angular` | Dataset to load |
| `datasetPortion` | `1.0` | Fraction of dataset to use |
| `workloadMode` | `SEGMENTS_AND_COMPACT` | Benchmark pipeline mode |
| `numSegments` | `2` | Number of segments before compaction |

---

# Graph Construction Parameters

| Parameter | Default | Description |
|---|---|---|
| `graphDegree` | `32` | Max neighbors per node |
| `beamWidth` | `100` | Beam search width during graph construction |

---

# Compaction Parameters

| Parameter | Default | Description |
|---|---|---|
| `compactMode` | `INLINE` | Vector storage strategy |
| `pqPath` | *(empty)* | PQ codebook path |
| `pqVecPath` | *(auto)* | Output path for PQVectors |
| `trainPQ` | `false` | Train PQ from dataset |

---

# Storage Parameters

These allow benchmarking across different storage tiers.

| Parameter | Description |
|---|---|
| `storageDirectories` | Comma-separated list of directories for storing segments |
| `storageClasses` | Expected storage class for each directory |

If provided, the benchmark verifies mount points using `CloudStorageLayoutUtil`.

---

# Parameter Validation

The benchmark performs strict validation before execution.

### General validation

- `graphDegree` must be positive
- `beamWidth` must be positive
- `numSegments` must be positive

---

### Mode-specific validation

**BUILD_FROM_SCRATCH**

Ignored parameters:

```
compactMode
pqPath
pqVecPath
trainPQ
numSegments
```

The benchmark logs a warning if these are provided.

---

**PQ_SEPARATE**

If `trainPQ=false`, then:

```
pqPath must be provided
```

If `trainPQ=true`, then:
```
train PQ from the entire dataset and saved the PQ to pqPath if provided. The PQ is saved to default if not provided.
During compaction, the compactor uses that trained PQ to encode each vectors and output to pqVecPath.
```

---

# PQ / Compaction Options in CompactorBenchmark

This section explains the interaction between:

- `compactMode`
- `pqPath`
- `pqVecPath`
- `trainPQ`

---

# Quick Mental Model

There are **two independent choices**:

1. **How vectors are represented in the output index**
   - `INLINE`       → full precision vectors stored inline in the graph
   - `PQ_SEPARATE`  → compressed vectors stored in a separate `.pq` file (PQVectors)
   - `FUSED_PQ`     → compressed neighbor codes stored inside the graph (fused PQ)

2. **Where the PQ codebook comes from (only relevant to PQ_SEPARATE)**
   - load from `pqPath`  (`trainPQ=false`)
   - train from full dataset (`trainPQ=true`)
   - If `trainPQ=ture`, save the trained PQ to `pqPath`
`pqVecPath` is simply the **output path** of PQVectors, only relevant to `PQ_SEPARATE`.

---

# Summary Table

| compactMode | What gets written | Needs `pqPath`? | Produces `pqVecPath`? | Recall uses |
|---|---|---|---|---|
| `INLINE` | graph with inline full vectors | No | No | `ravv` (full precision vectors) |
| `PQ_SEPARATE` | graph + `PQVectors` file | Yes if `trainPQ=false` | Yes | `PQVectors.scoreFunctionFor(...)` |
| `FUSED_PQ` | graph with fused PQ codes | No | No | fused PQ feature |

---

# Execution Examples

## Run with Defaults

```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor
```


## Build From Scratch

```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor \
-Dargs="-p workloadMode=BUILD_FROM_SCRATCH"
```

---

## Segment Build + Compaction

```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor \
-Dargs="-p workloadMode=SEGMENTS_AND_COMPACT -p numSegments=4"
```

---

## PQ Separate Compaction

```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor \
-Dargs="-p compactMode=PQ_SEPARATE -p trainPQ=true"
```

---

## Fused PQ Compaction

```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor \
-Dargs="-p compactMode=FUSED_PQ"
```


## Full JMH Control

Standard JMH parameters can be passed.

Example:

```bash
./mvnw -pl benchmarks-jmh exec:exec@compactor \
-Dargs="-wi 2 -i 5 -p numSegments=1,4"
```

---

# Interpreting Results

### Score (ms/op)

Average time for the benchmark pipeline.

Meaning depends on mode.

| Mode | Meaning |
|---|---|
| BUILD_FROM_SCRATCH | graph build time |
| SEGMENTS_AND_COMPACT | segment build + compaction time |
| COMPACT_ONLY | compaction time |

---

### Recall (AuxCounter)

Measures search quality against ground truth.

Interpretation:

- Compare recall of `SEGMENTS_AND_COMPACT` against `BUILD_FROM_SCRATCH`
- Ensure compaction maintains graph quality

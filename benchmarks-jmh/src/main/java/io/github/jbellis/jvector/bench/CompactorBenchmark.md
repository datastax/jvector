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

---

# 1. Workload Modes

| Mode | Description |
|------|-------------|
| `PARTITION_AND_COMPACT` | **(default)** Build partitions, compact them, then measure recall — all in one run |
| `PARTITION_ONLY` | Build N partition indexes and exit; no compaction |
| `COMPACT_ONLY` | Compact existing partitions without loading the dataset |
| `BUILD_FROM_SCRATCH` | Build a single index over the full dataset |

---

# 2. Quick Start

## Default: partition and compact in one run

The default mode builds partitions and immediately compacts them. Use this when you want a single-command end-to-end result. Adjust `-Xmx` to fit the dataset in memory (e.g., 220g for large datasets).

```bash
java -Xmx220g --add-modules jdk.incubator.vector \
  -cp benchmarks-jmh/target/compactor-benchmark.jar \
  io.github.jbellis.jvector.bench.CompactorBenchmark \
  -p workloadMode=PARTITION_AND_COMPACT \
  -p datasetNames=ada002-100k \
  -p numPartitions=4 \
  -p splitDistribution=FIBONACCI \
  -p indexPrecision=FUSEDPQ \
  -wi 0 -i 1 -f 1
```

---

# 3. Measuring Peak Heap During Compaction

The two-step workflow (`PARTITION_ONLY` → `COMPACT_ONLY`) exists to isolate compaction's true memory footprint. In `PARTITION_AND_COMPACT` mode the dataset is still resident in heap during compaction, which inflates the apparent memory cost. `COMPACT_ONLY` skips dataset loading entirely, so the heap limit applies only to the compactor itself.

This lets you prove that compaction can run on machines with very little RAM — e.g., `-Xmx5g` is sufficient even for large datasets.

## Step 1: Build partitions

Run with a large heap since the full dataset must be loaded into memory.

```bash
java -Xmx220g --add-modules jdk.incubator.vector \
  -cp benchmarks-jmh/target/compactor-benchmark.jar \
  io.github.jbellis.jvector.bench.CompactorBenchmark \
  -p workloadMode=PARTITION_ONLY \
  -p datasetNames=ada002-100k \
  -p numPartitions=4 \
  -p splitDistribution=FIBONACCI \
  -p indexPrecision=FUSEDPQ \
  -wi 0 -i 1 -f 1
```

The partition files are written to disk and reused in the next step.

## Step 2: Compact only (low-memory run)

The dataset is **not** loaded in this mode. Use a small `-Xmx` to measure and prove the compactor's true peak heap.

```bash
java -Xmx5g --add-modules jdk.incubator.vector \
  -cp benchmarks-jmh/target/compactor-benchmark.jar \
  io.github.jbellis.jvector.bench.CompactorBenchmark \
  -p workloadMode=COMPACT_ONLY \
  -p datasetNames=ada002-100k \
  -p numPartitions=4 \
  -p splitDistribution=FIBONACCI \
  -p indexPrecision=FUSEDPQ \
  -wi 0 -i 1 -f 1
```

`durationMs` in the output records only the `compact()` call — not JVM startup or I/O setup.

---

# 4. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `datasetNames` | `ada002-100k` | Dataset name |
| `workloadMode` | `PARTITION_AND_COMPACT` | Which phase(s) to run |
| `numPartitions` | `4` | Number of source partition indexes |
| `splitDistribution` | — | Data partitioning strategy (`UNIFORM`, `FIBONACCI`, …) |
| `indexPrecision` | — | `FULLPRECISION` (inline vectors only) or `FUSEDPQ` (inline + FusedPQ) |
| `storageDirectories` | *(temp dir)* | Comma-separated list of directories where partition files are written; partitions are distributed round-robin across them. Defaults to a JVM temp directory if unset. |

---

# 5. Index Precision

`indexPrecision` controls what features are written into each partition index.

| Value | Written features |
|-------|-----------------|
| `FULLPRECISION` | `INLINE_VECTORS` only |
| `FUSEDPQ` | `INLINE_VECTORS` + `FUSED_PQ` — required for compressed compaction |

---

# 6. Results

Results are written as JSONL to:

```
target/benchmark-results/compactor-<timestamp>/compactor-results.jsonl
```

Key fields:

| Field | Description |
|-------|-------------|
| `durationMs` | Time spent in the measured phase only |
| `recall` | Recall@10 (present when workload mode includes recall, e.g. `PARTITION_AND_COMPACT`) |
| `peakHeapMb` | Peak JVM heap observed during the run |

---

# 7. Memory Footprint

All datasets in the recall table (see `docs/compaction.md`) can be run under `COMPACT_ONLY` with `-Xmx5g`. Compaction also successfully scales to a dataset with 2560 dimensions and 10M vectors under the same constraint.

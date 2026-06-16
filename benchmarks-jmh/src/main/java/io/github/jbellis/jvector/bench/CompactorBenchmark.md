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

`CompactorBenchmark` measures the **compaction time, search latency, recall, and heap
memory footprint** of graph index compaction (`OnDiskGraphIndexCompactor`).

A run does up to two things:

1. **Compact** several partition indexes into one graph, timing only the compaction.
2. **Search** the resulting graph against the dataset's queries to report recall and
   per-query search latency (only when `measureRecall=true`).

---

## Contents

1. [Quick start](#1-quick-start)
2. [Workload modes](#2-workload-modes)
3. [measureRecall](#3-measurerecall)
4. [Two-step workflow: partition, then compact](#4-two-step-workflow-partition-then-compact)
5. [Parameters](#5-parameters)
6. [Split distributions](#6-split-distributions)
7. [Index precision](#7-index-precision)
8. [Output](#8-output)

---

# 1. Quick start

**End-to-end (build partitions, then compact, then measure recall):**

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

Size `-Xmx` to fit the dataset in memory (e.g. `220g` for large datasets). The flags
`-wi 0 -i 1 -f 1` mean zero warmup iterations, one measurement iteration, one fork.

---

# 2. Workload modes

Set with `-p workloadMode=<MODE>`.

| Mode | What it does |
|------|--------------|
| `PARTITION_AND_COMPACT` *(default)* | Build N partitions, then compact them |
| `PARTITION` | Build N partition indexes and exit; no compaction |
| `COMPACT` | Compact already-built partitions from disk |
| `BUILD` | Build a single index over the full dataset (baseline) |

`COMPACT` requires partition files on disk — run `PARTITION` first (or use
`PARTITION_AND_COMPACT`, which does both in one process).

---

# 3. `measureRecall`

**`measureRecall` defaults to `true`.** After compacting (or building), the benchmark
loads the dataset's query vectors and ground truth, searches the resulting graph, and
reports `recall` plus `avgSearchLatencyMs` / `p99SearchLatencyMs`.

This matters for memory experiments: even though `COMPACT` mode never loads the base
vectors into heap, with the default `measureRecall=true` it **still loads query vectors +
ground truth and runs a search**, which adds heap usage and extra work.

**To see the true heap memory footprint of compaction, set `measureRecall=false` together
with `COMPACT` mode.** That skips dataset loading entirely, so the heap reflects only the
compactor itself:

```bash
-p workloadMode=COMPACT -p measureRecall=false
```

When comparing across modes, keep `datasetNames`, `indexPrecision`, `graphDegree`, and
`beamWidth` identical so the numbers are directly comparable.

---

# 4. Two-step workflow: partition, then compact

Splitting the run in two means you **partition once and compact many times**: step 1
writes the partition files to disk, and any number of subsequent `COMPACT` runs (different
heaps, `numPartitions`, repeated measurements, etc.) reuse them without re-partitioning.

Use `-p storageDirectories=<dir>[,<dir2>...]` to control where partition files are
written. By default they go to a JVM temp directory, which may be cleaned up between runs;
point `storageDirectories` at a persistent path so step 1's output survives for reuse in
step 2. The same value must be passed to both steps. Multiple comma-separated directories
distribute partitions round-robin (e.g. across disks).

The two-step workflow could also isolate compaction's real heap cost. In
`PARTITION_AND_COMPACT` the dataset is still resident in heap during compaction, which
inflates the apparent cost. Running `COMPACT` with `measureRecall=false` loads nothing but
the partitions being merged — proving compaction runs with a small heap (e.g. `-Xmx5g`,
even for large datasets).

**Step 1 — build the partitions** (needs a large heap to hold the dataset):

```bash
java -Xmx220g --add-modules jdk.incubator.vector \
  -cp benchmarks-jmh/target/compactor-benchmark.jar \
  io.github.jbellis.jvector.bench.CompactorBenchmark \
  -p workloadMode=PARTITION \
  -p datasetNames=ada002-100k \
  -p numPartitions=4 \
  -p splitDistribution=FIBONACCI \
  -p indexPrecision=FUSEDPQ \
  -p storageDirectories=/path/to/partitions \
  -wi 0 -i 1 -f 1
```

Partition files are written to `storageDirectories` and reused in step 2.

**Step 2 — compact the partitions.** Choose one of the two options below.

*Option A — recall and search latency* (`measureRecall=true`, the default): after
compacting, the benchmark searches the compacted graph and reports `recall`,
`avgSearchLatencyMs`, and `p99SearchLatencyMs`. This loads query vectors + ground truth, so
size `-Xmx` accordingly.

```bash
java -Xmx220g --add-modules jdk.incubator.vector \
  -cp benchmarks-jmh/target/compactor-benchmark.jar \
  io.github.jbellis.jvector.bench.CompactorBenchmark \
  -p workloadMode=COMPACT \
  -p measureRecall=true \
  -p datasetNames=ada002-100k \
  -p numPartitions=4 \
  -p splitDistribution=FIBONACCI \
  -p indexPrecision=FUSEDPQ \
  -p storageDirectories=/path/to/partitions \
  -wi 0 -i 1 -f 1
```

*Option B — heap memory footprint* (`measureRecall=false`): the dataset is *not* loaded,
so a small heap proves the compactor's true footprint.

```bash
java -Xmx5g --add-modules jdk.incubator.vector \
  -cp benchmarks-jmh/target/compactor-benchmark.jar \
  io.github.jbellis.jvector.bench.CompactorBenchmark \
  -p workloadMode=COMPACT \
  -p measureRecall=false \
  -p datasetNames=ada002-100k \
  -p numPartitions=4 \
  -p splitDistribution=FIBONACCI \
  -p indexPrecision=FUSEDPQ \
  -p storageDirectories=/path/to/partitions \
  -wi 0 -i 1 -f 1
```

All datasets in the recall table (see `docs/compaction.md`) run under Option B with
`-Xmx5g`. Compaction also scales to 2560 dimensions × 10M vectors under the same limit.

---

# 5. Parameters

Set any parameter with `-p <name>=<value>`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `workloadMode` | `PARTITION_AND_COMPACT` | Which phase(s) to run — see §2 |
| `measureRecall` | `true` | Search the result for recall + latency — see §3 |
| `datasetNames` | `ada002-100k` | Dataset name |
| `numPartitions` | `4` | Number of source partition indexes |
| `splitDistribution` | `FIBONACCI` | How vectors are divided across partitions — see §6 |
| `indexPrecision` | `FUSEDPQ` | `FULLPRECISION` or `FUSEDPQ` — see §7 |
| `graphDegree` | `32` | Max neighbors per node |
| `beamWidth` | `100` | Beam width during graph construction |
| `storageDirectories` | *(temp dir)* | Comma-separated dirs for partition files; distributed round-robin. Defaults to a JVM temp dir. |

---

# 6. Split distributions

`splitDistribution` controls how vectors are divided across partitions.

| Distribution | Weights (4 partitions) | Description |
|---|---|---|
| `UNIFORM` | [1, 1, 1, 1] (25% each) | Equal-sized partitions |
| `FIBONACCI` | [1, 2, 3, 5] (9/18/27/45%) | Fibonacci-weighted; partitions grow progressively |
| `LOG2N` | [1, 2, 4, 8] (7/13/27/53%) | Power-of-two weighted |
| `TIERED_10_90` | [1, 9] (10/90%) | Small + large; new segment compacted into a large index |
| `TIERED_1_99` | [1, 99] (1/99%) | Extreme tiered compaction scenario |

`TIERED_10_90` and `TIERED_1_99` are meant for 2-partition runs (`-p numPartitions=2`).

---

# 7. Index precision

`indexPrecision` controls what features are written into each partition index.

| Value | Written features |
|-------|-----------------|
| `FULLPRECISION` | `INLINE_VECTORS` only |
| `FUSEDPQ` | `INLINE_VECTORS` + `FUSED_PQ` — required for compressed compaction |

---

# 8. Output

## JMH summary table

Printed to the console when the run finishes, as secondary metrics on the `run` benchmark:

| Metric | Meaning |
|--------|---------|
| `run:compactionTimeMs` | Time inside `compact()` only |
| `run:avgSearchLatencyMs` | Mean per-query `search()` latency on the result graph |
| `run:p99SearchLatencyMs` | 99th-percentile per-query `search()` latency |
| `run:recall` | Recall@10 |

The latency and recall metrics are only present when `measureRecall=true`.

## Results JSONL

Full results are also written to:

```
target/benchmark-results/compactor-<timestamp>/compactor-results.jsonl
```

| Field | Description |
|-------|-------------|
| `durationMs` | Time in the measured phase only (`compact()` for `COMPACT`); excludes JVM startup and I/O setup |
| `recall` | Recall@10 (present when `measureRecall=true`) |
| `avgSearchLatencyMs` / `p99SearchLatencyMs` | Per-query `search()` latency (present when `measureRecall=true`) |

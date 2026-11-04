<!--
Copyright DataStax, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# JVector fastervector benchmarks

Benchmarks fastervector's IVF-ASH index ([`jvector-fastervector`](../jvector-fastervector/README.md))
with jvector's benchmark tooling: the same datasets, ground truth, `datasets.yml`, `run-config.yml`
and reporting as the graph index. That makes IVF results directly comparable with the graph's.

| Runner | Graph counterpart | Use |
|---|---|---|
| `BenchIvfYAML` | `BenchYAML` | Interactive: a console table per dataset, optional `experiments.csv` logging |
| `AutoBenchIvfYAML` | `AutoBenchYAML` | Regression: `--output` JSON and CSV, checkpointing; the CSV lines up with `AutoBenchYAML`'s |

> **Status: experimental.** Linux x86_64 with AVX2 only, like `jvector-fastervector`. The module is
> opt-in (`-Pfastervector`) and is not built by default or in CI.

---

## Requirements

Everything `jvector-fastervector` needs (JDK 22+, staged native libraries, MKL at run time; see its
README), plus disk space for the indexes under `ssdPaths`.

Run every command from the **repository root**: configuration paths resolve from there, as in
`jvector-examples`. Run a benchmark in a JVM of its own, never in the same JVM as a graph benchmark:
fastervector's `libjvector.so.0` clashes with jvector-native's.

## Configuration

IVF parameters live in `yaml-configs/ivf-parameters/<dataset>.yml`, or `default.yml` when a
dataset has no file of its own. Every list is a grid, and every combination runs.

```yaml
yamlSchemaVersion: 1
dataset: default
construction:
  nlist: [1024]               # IVF coarse centroids; each value is a construction point
  storageFactor: [1.0]        # 1.0 = 32x compression, 2.0 = 16x
  buildThreads: 0             # 0 = all CPUs
  ssdPaths: [ivf_cache]       # one directory per SSD; relative paths resolve from the repo root
  useSavedIndexIfExists: No   # Yes reopens an index already built for the same construction point
search:
  topKNprobe:                 # topK -> nprobe values, like topKOverquery
    10: [8, 16, 32, 64]
    100: [16, 64]
  searchThreads: [1, 8]
```

Each construction point builds into `<ssdPaths[i]>/<dataset>/nlist<N>-sf<F>/`. Unknown fields and
invalid values are rejected with the file and field named.

`BenchIvfYAML` also reads `jvector-examples/yaml-configs/run-config.yml`, the same file `BenchYAML`
uses, for what to compute, show and log. `AutoBenchIvfYAML`, like `AutoBenchYAML`, always computes
average QPS, mean latency and recall.

## What is measured

| Metric | How |
|---|---|
| `QPS` (batched) | The whole query set as one `search` call on a pool of `searchThreads`; 3 warm-up and 3 measured runs. How fastervector is meant to be used. |
| `Per-query QPS` | One query per call over `searchThreads` Java threads, each with its own searcher. The shape of the graph benchmark's QPS. |
| `Mean Latency` | One thread, one query per call, over the query set twice. |
| `Recall@k`, `MAP@k` | Against the dataset's ground truth, with jvector-examples' `AccuracyMetrics`. |
| Build time, index size, file count | Per construction point. |
| Heap and off-heap | JVM figures, as for the graph. fastervector's native memory isn't included. |

Warm-up runs use random unit vectors, and each measured run follows a GC pause, as in the graph
benchmark. Graph node counts don't apply to IVF. Datasets with EUCLIDEAN similarity are skipped,
because fastervector scores by inner product only; COSINE datasets are normalized on load.

## Running

### As a Java program

Build the runnable jar. It bundles the native libraries if they're staged when you build. With
`-am`, Maven also builds `jvector-examples` and `jvector-native`, including jvector-native's meson
build on Linux, so skip the tests:

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
./mvnw -Pfastervector -pl jvector-fastervector-bench -am package -DskipTests
```

Regression run:

```bash
java --enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector \
     -Djvector.enable_native_vectorization=false \
     -Xmx<half-of-RAM>g \
     -cp jvector-fastervector-bench/target/jvector-fastervector-bench-*-jar-with-dependencies.jar \
     io.github.jbellis.jvector.fastervector.bench.AutoBenchIvfYAML \
     --output ivf-bench-results [--config <ivf-config.yml>] cap-1M
```

- `--output <prefix>` is required. It writes `<prefix>.json`, `<prefix>.csv` and
  `<prefix>.checkpoint.json`; an interrupted run resumes from the checkpoint.
- Dataset regexes are matched against the `regression-tests` section of `datasets.yml`; none
  matches them all.
- `--config` uses one IVF config file for every dataset.

Interactive run: the same command with the main class
`io.github.jbellis.jvector.fastervector.bench.BenchIvfYAML`. Its arguments are dataset regexes,
matched against every dataset in `datasets.yml`, and/or IVF config files (`*.yml`).

If the jar was built without staged libraries, add
`-Djvector.fastervector.native.dir=$PWD/jvector-fastervector/native/linux-x86_64`.

### Through Maven

Each runner is a profile that runs it after compiling. `bench.args` holds the runner's arguments
and `bench.jvm.args` any extra JVM options:

```bash
./mvnw -Pfastervector,bench-ivf -pl jvector-fastervector-bench -am process-classes \
    -Dbench.args="cap-1M"
./mvnw -Pfastervector,autobench-ivf -pl jvector-fastervector-bench -am process-classes \
    -Dbench.args="--output ivf-bench-results cap-1M" -Dbench.jvm.args="-Xmx64g"
```

These are profiles bound to a phase rather than `exec:exec@<id>` goals: with `-am`, an `exec`
goal would also run in `jvector-examples`, whose default profile turns `exec` on.

## Comparing with the graph index

Run `AutoBenchYAML` on the same machine and datasets, in its own JVM, as CI's `run-bench.yml` does:

```bash
./mvnw -pl jvector-examples -am package -DskipTests
java --enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector \
     -Djvector.enable_native_vectorization=true \
     -Xmx<half-of-RAM>g \
     -cp jvector-examples/target/jvector-examples-*-jar-with-dependencies.jar \
     io.github.jbellis.jvector.example.AutoBenchYAML \
     --output graph-bench-results cap-1M
```

Use different `--output` prefixes: each runner keeps its checkpoint in `<prefix>.checkpoint.json`.

`ivf-bench-results.csv` and `graph-bench-results.csv` share their first 11 columns:

```
dataset,QPS,QPS StdDev,Mean Latency,Recall@10,Index Construction Time,Avg Nodes Visited,
Build Heap Used (MB),Build Off-Heap (MB),Search Heap Used (MB),Search Off-Heap (MB)
```

They're computed by the same `BenchmarkSummarizer`, so each is an average over the dataset's
configurations, by the same rules. `Avg Nodes Visited` is blank for IVF. The IVF file adds
`Per-query QPS,Per-query QPS StdDev,Index Size (MB)`. Compare IVF's `QPS` (batched) and
`Per-query QPS` against the graph's `QPS` at similar `Recall@10`; the `.json` files have every
configuration's numbers, which is where to look for the recall-throughput tradeoff.

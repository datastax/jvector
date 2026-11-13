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

The commands below use the environment variables from the `jvector-fastervector` README
(`JVECTOR_HOME`, `ONEAPI_SETVARS`, and `FASTERVECTOR_PINNED` for staging), set to the paths on your
machine.

Run every command from the **repository root**, `cd "$JVECTOR_HOME"`: configuration paths
resolve from there, as in `jvector-examples`. Run a benchmark in a JVM of its own, never in the
same JVM as a graph benchmark: fastervector's `libjvector.so.0` clashes with jvector-native's.

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
  searchThreads: [1, 8]       # 0 = all CPUs
  queryThreads: [1]           # threads of a single-query searcher (latency, per-query QPS); 0 = all CPUs
```

`searchThreads` sets the batched searcher's threads and the number of concurrent callers in the
coalesced and per-query modes. `queryThreads` is optional (default `[1]`): the native thread count
of a searcher that takes one query per call, as in the latency benchmark and each per-query caller.
Each listed value is another search point.

Each construction point builds into `<ssdPaths[i]>/<dataset>/nlist<N>-sf<F>/`. Unknown fields and
invalid values are rejected with the file and field named.

The machine-specific settings can be overridden on the command line, so the same dataset files
serve every machine:

| Property | Replaces |
|---|---|
| `-Divf.ssdPaths=/data/ssd0/ivf_cache` | `construction.ssdPaths` (comma-separated) |
| `-Divf.buildThreads=16` | `construction.buildThreads` |
| `-Divf.searchThreads=1,16,48` | `search.searchThreads` (comma-separated) |
| `-Divf.queryThreads=1,4` | `search.queryThreads` (comma-separated) |
| `-Divf.perQuery=false` | Skips direct per-query throughput (see below) |

Each applied override is printed. Through Maven, pass them in `-Dbench.jvm.args`.

The dataset files for the six `regression-tests` datasets (`cap-1M.yml`, `cap-6M.yml`,
`cohere-english-v3-1M.yml`, `cohere-english-v3-10M.yml`, `dpr-gemma-1m.yml`, `dpr-gemma-10m.yml`)
are matched to the graph's regression configs; see [Comparing with the graph
index](#comparing-with-the-graph-index). Their names must match `datasets.yml` exactly, case
included, or the lookup falls back to `default.yml` on Linux.

`yaml-configs/ivf-experiments/` holds one-off experiment configs, such as
`cap-1M-calibration.yml`, the sweep the per-dataset files are set from, and
`cap-1M-query-threads.yml`, which sweeps `queryThreads`. Pass one to `BenchIvfYAML`
by path.

`BenchIvfYAML` also reads `jvector-examples/yaml-configs/run-config.yml`, the same file `BenchYAML`
uses, for what to compute, show and log. `AutoBenchIvfYAML`, like `AutoBenchYAML`, always computes
average QPS (batched, coalesced and per-query), mean latency and recall.

## What is measured

| Metric | How |
|---|---|
| `QPS` (batched) | The whole query set as one `search` call on a pool of `searchThreads`; 3 warm-up and 3 measured runs. How fastervector is meant to be used. |
| `Coalesced QPS` | One query per call from `searchThreads` Java threads, through an `IvfAshBatchingSearcher` that combines concurrent queries into batched calls on one searcher of `searchThreads` threads. The shape of the graph benchmark's QPS. |
| `Per-query QPS` | One query per call from `searchThreads` Java threads, each calling fastervector directly on its own searcher of `queryThreads` threads. Doesn't scale yet (below). Off with `-Divf.perQuery=false`. |
| `Mean Latency` | One thread and one searcher of `queryThreads` threads, one query per call, over the query set twice. |
| `Recall@k`, `MAP@k` | Against the dataset's ground truth, with jvector-examples' `AccuracyMetrics`. |
| Build time, index size (`Index Size (MB)`), file count | Per construction point. Index size is the total size of the index files on disk. |
| Heap and off-heap | JVM figures, as for the graph. fastervector's native memory isn't included. |
| `Index Memory (MB)` | Resident memory (RSS) the index open adds: the index's in-memory footprint. A built index is closed and reopened to measure it the same way. Linux only. |
| `Search Peak Memory (MB)` | Peak RSS during a search point, above the RSS before the index was opened: the index plus the search's working memory, and any JVM heap the search newly touched. Linux only. |
| `Build Peak Memory (MB)` | Peak RSS during a build, above the RSS before it. Linux only. |

**Batched, coalesced and per-query.** All three modes run the same queries over the same index.
*Batched* passes the whole query set (10,000 queries) to fastervector in one `search` call: one
crossing of the Java/native boundary in each direction. fastervector then spreads the queries over
its own threads, and reads each probed list once for every query in the batch that probes it.
*Coalesced* and *per-query* both make one call per query from `searchThreads` Java threads, as the
graph benchmark and a server answering one request at a time do; they differ in what's behind the
call.

*Per-query* calls fastervector directly, and at the pinned fastervector commit that barely scales.
A searcher uses its pool only for the thread count: each call starts that many new I/O threads on
io_uring rings `0..n-1`, which every searcher shares behind a lock, and scores on a process-wide
scan pool of that size, which every searcher with the same count shares. With the default
`queryThreads: 1`, 48 callers share one scoring thread and one ring; on `cap-1M` they gave about
376 QPS against 56,699 batched. No thread count gives callers their own resources.

*Coalesced* puts `IvfAshBatchingSearcher` (in `jvector-fastervector`) in front of one searcher:
callers each submit one query, and a dispatcher thread searches whatever has queued in one call,
so queries that arrive during a search form the next batch. With 48 callers, batches of up to 48
queries keep fastervector on its batched path. The mean batch size is logged. Compare the graph's
`QPS` with coalesced QPS for the same request shape, and with batched QPS for IVF's best case.
Turn direct per-query off (`-Divf.perQuery=false`) in regression runs, since it takes most of a
run's time.

Warm-up runs use random unit vectors, and each measured run follows a GC pause, as in the graph
benchmark. Graph node counts don't apply to IVF. Datasets with EUCLIDEAN similarity are skipped,
because fastervector scores by inner product only; COSINE datasets are normalized on load.

## Running

### As a Java program

Build the runnable jar, `target/jvector-fastervector-bench-<version>-jar-with-dependencies.jar`.
Any `package` that includes this module builds it; see the build table in the
[`jvector-fastervector` README](../jvector-fastervector/README.md#build-and-test). It bundles the
native libraries if they're staged when you build. Either command also builds `jvector-examples`
and `jvector-native`, including jvector-native's meson build on Linux, so skip the tests:

```bash
cd "$JVECTOR_HOME"
source "$ONEAPI_SETVARS" >/dev/null 2>&1 || true
./mvnw -Pfastervector clean package -DskipTests
# or only this module and what it depends on:
./mvnw -Pfastervector -pl jvector-fastervector-bench -am package -DskipTests
```

A build without `-Pfastervector` doesn't touch this module, so an older jar stays in `target/`. If
`java -cp` then reports `ClassNotFoundException` for a runner, rebuild with `-Pfastervector`.

Regression run:

```bash
cd "$JVECTOR_HOME"
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
`-Djvector.fastervector.native.dir=$JVECTOR_HOME/jvector-fastervector/native/linux-x86_64`.

### Progress output

Both runners log each search point as it starts and ends (`Search point cap-1M/nlist1024-sf1.0
[3/12] topK=10 nprobe=32 searchThreads=48 queryThreads=1`), each benchmark's duration, and every throughput run
and latency pass with its result. A heartbeat logs the current phase every 60 seconds, so a long
single-threaded phase isn't silent; `-Divf.heartbeatSeconds=<n>` changes the interval, and `0`
turns it off.

### Through Maven

Each runner is a profile that runs it after compiling. `bench.args` holds the runner's arguments
and `bench.jvm.args` any extra JVM options:

```bash
cd "$JVECTOR_HOME"
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
cd "$JVECTOR_HOME"
./mvnw -pl jvector-examples -am package -DskipTests
java --enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector \
     -Djvector.enable_native_vectorization=true \
     -Xmx<half-of-RAM>g \
     -cp jvector-examples/target/jvector-examples-*-jar-with-dependencies.jar \
     io.github.jbellis.jvector.example.AutoBenchYAML \
     --output graph-bench-results cap-1M
```

Use different `--output` prefixes: each runner keeps its checkpoint in `<prefix>.checkpoint.json`.

### How the regression configs are matched

Each IVF regression config mirrors the graph's config for the same dataset in
`jvector-examples/yaml-configs/index-parameters/`:

| Graph | IVF | Why |
|---|---|---|
| `topKOverquery: 10: [1.0]`, one search point | `topKNprobe: 10: [<nprobe>]`, one search point | `BenchmarkSummarizer` averages every configuration of a dataset, so both sides need the same number of points with the same topK. |
| PQ, `mFactor: 8` (32x) | `storageFactor: [1.0]` (32x) | The same compressed size per vector. |
| `useSavedIndexIfExists: No` | `useSavedIndexIfExists: No` | Both report a fresh build time. |
| QPS on all CPUs | `searchThreads: [0]` (all CPUs) | The same concurrency. Compare the graph's `QPS` with IVF's `Coalesced QPS` (the same single-query shape) and batched `QPS` (IVF's best case); see [What is measured](#what-is-measured). |

`nprobe` is chosen for **equal recall**: the smallest value whose `Recall@10` reaches the graph's
`Recall@10` on that dataset, so the comparison is QPS at the same recall. Until the calibration
run, `nlist` and `nprobe` in these files are provisional estimates, and the files say so.

These differences remain, and the comparison should state them:

- **Reranking.** The graph reranks its candidates with NVQ vectors; IVF-ASH scores from its 32x
  codes alone. IVF reaches the graph's recall through a larger `nprobe`.
- **Storage.** The graph's index is read through the page cache and is usually memory-resident at
  1M; IVF-ASH reads its lists from the SSD paths on every query. Use the same disk for both.
- **Build threads.** The graph builds on the physical core count; IVF's `buildThreads: 0` uses
  all CPUs. Set `-Divf.buildThreads=<physical cores>` to match.
- **Batching.** IVF's batched and coalesced QPS read each probed list once for all the queries in
  the batch that probe it; the graph answers each query on its own. Coalesced batches are only as
  large as the number of queries waiting, so they depend on the number of callers.
- **Native memory.** fastervector's native memory isn't in the off-heap figures.

`ivf-bench-results.csv` and `graph-bench-results.csv` share their first 11 columns:

```
dataset,QPS,QPS StdDev,Mean Latency,Recall@10,Index Construction Time,Avg Nodes Visited,
Build Heap Used (MB),Build Off-Heap (MB),Search Heap Used (MB),Search Off-Heap (MB)
```

They're computed by the same `BenchmarkSummarizer`, so each is an average over the dataset's
configurations, by the same rules. `Avg Nodes Visited` is blank for IVF. The IVF file adds
`Coalesced QPS,Coalesced QPS StdDev,Per-query QPS,Per-query QPS StdDev,Index Size (MB),Index Memory (MB),Search Peak Memory (MB),Build Peak Memory (MB)`;
the per-query columns are blank when it was turned off, and the memory columns off Linux.
The graph's CSV has no footprint columns: its `Disk Usage (MB)` and memory-mapped
`Off-Heap Mapped (MB)` are in `AutoBenchYAML`'s JSON, and the mapped figure is the whole mapping,
not what's resident. Compare the graph's `QPS` against IVF's `Coalesced QPS` and `QPS` (batched) at similar `Recall@10`;
the `.json` files have every configuration's numbers, which is where to look for the
recall-throughput tradeoff.

### Recall-vs-QPS curves

The single-point regression configs give one number per side. To see the whole tradeoff, sweep
the search knob instead: `cap-1M-recall-qps-sweep.yml` exists on both sides, in
`yaml-configs/ivf-experiments/` (nprobe) and in `jvector-examples/yaml-configs/index-parameters/`
(overquery). Run them with the interactive runners, `BenchIvfYAML` and `BenchYAML`, with logging
turned on in `run-config.yml`: the Auto runners average every point into one row, and a curve needs
one `experiments.csv` row per point. Then plot both files on one chart:

```bash
python3 jvector-fastervector-bench/scripts/plot_recall_qps.py \
    <graph experiments.csv> <ivf experiments.csv> recall_vs_qps.png
```

It needs `pandas` and `matplotlib`. It draws one curve per graph build and per IVF
`(nlist, storageFactor)`, with Recall@10 on x and QPS on a log scale, and labels points with their
overquery or nprobe. IVF's curve uses batched QPS; `--ivf-qps-column
search.throughput.coalesced.avg_qps` plots coalesced QPS instead.

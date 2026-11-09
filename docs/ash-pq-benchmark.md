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

# Reproducing ASH versus PQ

This study compares fused ASH and fused PQ graph indexes at matched code sizes:
both store the same bytes per vector, including ASH's five-byte header. It uses
ordinary packaged JVector code and `io.github.jbellis.jvector.example.BenchYAML`;
no separate query harness, source overlay or profiling agent is involved.

The [index configurations](../jvector-examples/yaml-configs/index-parameters/studies/ash-vs-pq)
define each case. The [run configurations and dataset catalogs](../jvector-examples/yaml-configs/ash-pq)
define what is measured and where the input files are. Two scripts run them:
`run-ash-pq-benchmark.sh` runs one case in one JVM, and `run-ash-pq-study.py`
steps through a whole study. The [BenchYAML Guide](benchyaml-guide.md) describes
the index configuration, and [ASH configuration and defaults](ash-configuration.md)
describes model and runtime settings.

## Results

[ASH performance](ash-performance.md) reports the 6M and 10M results: ASH wins
total construction in 6/6 configurations, recall in 12/12 cases and QPS in 22/24
cases. It also describes the machine, JVM, placement and protocol used, so you
can note any differences in your own runs. The 1M recipes (MPNet, Cohere, Ada002
and OpenAI-3072) use the same parameters and protocol.

## Build the checkout

Use JDK 22 for performance measurements. Follow the
[native prerequisites](../jvector-native/src/main/native/README.md), including the
compiler, Meson, Ninja and CMake. Native packaging fetches the pinned oneDNN
source and includes the statically linked ASH add-on. ASH training also uses
the installed BLAS/LAPACK provider; record it with the results, since a Java
fallback does not reproduce native training throughput.

```bash
export JAVA_HOME=/path/to/jdk-22
export PATH="$JAVA_HOME/bin:$PATH"
git submodule update --init --recursive
mvn -B -pl jvector-examples -am -DskipTests package
```

This writes `jvector-examples/target/jvector-examples-<version>-jar-with-dependencies.jar`.
Build once, before any benchmark, and keep the checkout and JAR fixed throughout
a comparison. The scripts require a clean checkout so each run records an exact
commit. The run command requires the native provider, so a failed native
installation cannot silently become a scalar measurement.

## Inputs and index parameters

Obtain the normalized fvecs/ivecs files named in
[`catalog-1m.yml`](../jvector-examples/yaml-configs/ash-pq/catalog-1m.yml) and
[`catalog-large.yml`](../jvector-examples/yaml-configs/ash-pq/catalog-large.yml).
Cohere 1M, Ada002 1M and OpenAI-3072 1M are in the public dataset catalog
(see [dataset catalogs](../jvector-examples/yaml-configs/dataset-catalogs/datasets.md));
the others must be obtained separately. Set `JVECTOR_BENCH_DATA_DIR` to the
absolute directory that holds them, keeping the relative file names in the
catalog. All use `DOT_PRODUCT` and `NO_SCRUB`. Use the exact base, query and
top-100 ground-truth files: a deduplicated base file without rebuilt ground
truth would change the experiment.

| Dataset | Recipe prefix | Base vectors | Dimensions | Queries | Code bytes, b=2 / b=4 |
| --- | --- | ---: | ---: | ---: | ---: |
| MPNet 1M | `mpnet` | 999,812 | 768 | 10,000 | 101 / 197 |
| Cohere 1M | `cohere` | 1,000,000 | 1024 | 10,000 | 133 / 261 |
| Ada002 1M | `ada002` | 982,790 | 1536 | 10,000 | 197 / 389 |
| OpenAI-3072 1M | `openai3072` | 999,000 | 3072 | 1,000 | 389 / 773 |
| MPNet 10M | `mpnet10m` | 9,997,697 | 768 | 10,000 | 101 / 197 |
| Cohere 10M | `cohere10m` | 10,000,000 | 1024 | 10,000 | 133 / 261 |
| CAP 6M | `cap6m` | 6,070,488 | 1536 | 10,000 | 197 / 389 |

ASH uses one landmark, an ITQ-trained transform to D/2 dimensions, and 2 or 4
bits per projected dimension. PQ uses as many 8-bit subspaces as ASH's code
bytes, 256 centroids, no centering, and non-anisotropic training. The 2-bit and
4-bit labels describe the ASH cases, not PQ centroid precision.

Both use an on-disk index (format 7) with fused graph features, NVQ reranking,
degree 32, `efConstruction=100`, neighbor overflow 1.2, hierarchy, final
refinement and the parallel writer, and return the top 10 at overquery 1 and 2.
ASH queries use queued execution with a maximum batch of 8 and automatic query
projection, which selects FP16 on supported CPUs. PQ queries use direct execution.

## Run a study

`run-ash-pq-study.py` runs every case of one dataset set in sequence, each in a
fresh JVM, and aggregates the CSV files. It contains no query loop or timer.

```bash
export JVECTOR_BENCH_DATA_DIR=/data/vectors
python3 jvector-examples/run-ash-pq-study.py --dataset-set large \
  --data "$JVECTOR_BENCH_DATA_DIR" --output /data/jvector-bench/ash-pq-large
```

`--dataset-set 1m` (the default) runs the four 1M datasets. By default, each
configuration is built twice, in separate case directories, and one query JVM per
core count reuses the first build's index; PQ and ASH alternate order. CPU and
memory placement come from `lscpu` on the host: construction uses one thread on
every physical core with memory interleaved across all NUMA nodes, and queries
use the first 1 and 8 physical cores of one node with memory on that node.
`--help` lists the options for other placements and repetition counts, and
`--plan-only` prints the job list without running it.

`plan.json` fixes the commit, JAR checksum and settings. `status.json`,
`results.json` and `summary.json` are updated after each invocation, and an
interrupted study resumes only with the same source and JAR. A process watchdog
(3 hours for large construction, 30 minutes otherwise) detects a hung
invocation; its wall clock is never reported as a measurement. Extra
construction repetitions release their caches after recording results.

## Run one case

`run-ash-pq-benchmark.sh` prepares a case directory, records the command and runs
standalone BenchYAML. It checks that the CPU list names the requested number of
distinct online physical cores, so two SMT siblings cannot count as two cores.

```bash
export JVECTOR_BENCH_DATA_DIR=/data/vectors
export JVECTOR_BENCH_HEAP=128g
BENCH_CASE=/data/jvector-bench/mpnet-2bit-ash-r1

# Construction: every physical core, memory on all nodes. Pick lists from lscpu.
JVECTOR_BENCH_NUMA_POLICY=interleave:0,1 \
  bash jvector-examples/run-ash-pq-benchmark.sh \
  mpnet-2bit-ash.yml construction 48 0-47 "$BENCH_CASE"

# Queries reuse that model and graph, each in a fresh JVM.
bash jvector-examples/run-ash-pq-benchmark.sh mpnet-2bit-ash.yml queued 1 0 "$BENCH_CASE"
bash jvector-examples/run-ash-pq-benchmark.sh mpnet-2bit-ash.yml queued 8 0-7 "$BENCH_CASE"
```

Query memory defaults to NUMA node 0 (`JVECTOR_BENCH_NUMA_POLICY=bind:0`). For the
matching PQ case, replace `ash` with `pq`, use `direct` instead of `queued`, and
use a separate case directory. For the larger datasets, set
`JVECTOR_BENCH_CATALOG=catalog-large.yml` for every invocation of the case.

Construction refuses nonempty model and index caches, and query runs require
them. The recipes set `useSavedIndexIfExists: Yes` so the graph outlives its
construction JVM; nothing is deleted automatically. Do not run cases at the same
time or share their CPUs with other work. Report any heap, CPU, NUMA or library
differences from the measurement environment in
[ASH performance](ash-performance.md).

Each run writes a complete `runs/<invocation>/command.sh`, including its
environment, so the Java command can be rerun directly. The case directory holds
copies of the catalog, dataset metadata and dataset list, because BenchYAML
resolves them relative to its working directory.

## Measurements and artifacts

* **Construction:** BenchYAML's training, encoding, NVQ and graph-write timers;
  dataset loading and setup are excluded. Each construction invocation reports
  two overquery rows from one build. Cached-index invocations report no
  construction time.
* **Throughput:** random-query warmups, then timed full query passes including
  preparation, traversal and reranking. Each sample lasts at least three seconds.
  Direct mode uses 4 × cores callers; queued mode uses at most one worker per
  core, admission 8 × cores and a maximum batch of 8.
* **Unloaded latency:** one query at a time through the same execution path,
  after three warmup passes of at least three seconds. It is not latency under a
  controlled arrival rate.
* **Queued response:** sampled during throughput, from admission to the worker's
  completion callback. It measures a bounded backlog, not an arrival-rate curve.
* **Recall:** the same inputs and ground truth for every mode. Concurrent results
  are checked against single-threaded searches with the same prepared query.

CSV results, system information and dataset information go to each case's
`logging` directory. The scripts also record the source commit, JAR checksum,
index and run YAML, Java version, CPU topology, console output and exit code
under `runs`. Keep these artifacts together with any reported numbers.

Hardware, OS scheduling and parallel graph construction cause run-to-run
variation; reproducing the study means the same protocol and comparable
distributions, not identical numbers.

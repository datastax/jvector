# Reproducing ASH versus PQ

This study compares fused ASH and fused PQ graph indexes at matched code sizes:
both store the same bytes per vector, including ASH's five-byte header. It runs
on five public 1M datasets, which BenchYAML downloads from the public JVector
dataset catalog, and uses only packaged JVector code and
`io.github.jbellis.jvector.example.BenchYAML`. Results are BenchYAML's own
output: CSV files and the tables it prints.

## Two ways to run it

Every case comes in two forms with the same index settings:

- **One JVM per measurement**
  ([`studies/ash-vs-pq`](../jvector-examples/yaml-configs/index-parameters/studies/ash-vs-pq)),
  run by `run-ash-pq-study.py`. One JVM builds and saves the index, then a fresh
  JVM measures each query core budget and overquery. Each JVM sees only its CPU
  budget (`-XX:ActiveProcessorCount` and a CPU list), so a 1-core measurement
  really has one core, including for garbage collection and JIT compilation, and
  no measurement inherits JIT, heap or cache state from construction or from an
  earlier measurement. Use this form for comparisons you report.
- **One JVM per case**
  ([`studies/ash-vs-pq-single-jvm`](../jvector-examples/yaml-configs/index-parameters/studies/ash-vs-pq-single-jvm)),
  run by `benchyaml-play.sh`. One JVM builds the index, then queries it once with 8
  search workers at overquery 1. It is quicker and closer to an application that
  builds and serves in one process, but its queries run in the JVM that just built
  the index, sharing its heap, JIT state and threads. Use it for quick end-to-end checks, and compare its results only with each other.

## Build the checkout

Use JDK 22 or later on Linux, with `numactl` and `lscpu`. Follow the
[native prerequisites](../jvector-native/src/main/native/README.md), including the
compiler, Meson, Ninja and CMake. Native packaging fetches the pinned oneDNN
source and includes the ASH add-on. ASH training also uses the installed
BLAS/LAPACK provider; note which one you have, since a Java fallback trains
more slowly.

```bash
export JAVA_HOME=/path/to/jdk-22
export PATH="$JAVA_HOME/bin:$PATH"
mvn -B -pl jvector-examples -am -DskipTests package -Dnative.onednn=true
```

This writes `jvector-examples/target/jvector-examples-<version>-jar-with-dependencies.jar`.
Build once and keep the checkout and JAR fixed throughout a comparison. The
scripts require a clean checkout so each run records an exact commit, and they
require the native vector provider, so a failed native build cannot silently
become a scalar measurement.

## Datasets and index settings

| Dataset | Recipe prefix | Base vectors | Dimensions | Queries | Code bytes per vector |
| --- | --- | ---: | ---: | ---: | ---: |
| Ada002 1M | `ada002` | 982,790 | 1536 | 10,000 | 197 |
| Cohere English v3 1M | `cohere` | 1,000,000 | 1024 | 10,000 | 133 |
| DPR-Gemma 1M | `dprgemma` | 999,762 | 768 | 10,000 | 101 |
| OpenAI-1536 1M | `openai1536` | 999,000 | 1536 | 1,000 | 197 |
| OpenAI-3072 1M | `openai3072` | 999,000 | 3072 | 1,000 | 389 |

All use dot-product similarity with the dataset's own top-100 ground truth. The
first run downloads each dataset into the data directory you name; later runs
reuse it.

ASH uses one landmark, an ITQ-trained transform to D/2 dimensions, and 2 bits per
projected dimension. PQ uses as many 8-bit subspaces as ASH's code bytes, 256
centroids, no centering and non-anisotropic training.

Both use an on-disk index (format 7) with fused graph features, NVQ reranking,
degree 32, `efConstruction=100`, neighbor overflow 1.2, hierarchy, final
refinement and the parallel writer, and return the top 10 at overquery 1 and 2.
ASH queries use queued execution with a maximum batch of 8, automatic query
projection and adaptive stopping. PQ queries use direct execution.

Each algorithm is built at its fastest thread setting. On the machines we
measured, ASH construction was fastest on both hardware threads of every core
(SMT) and PQ construction on one thread per core; check this on your own
hardware with `--build-smt`.

## Run the study (one JVM per measurement)

```bash
python3 jvector-examples/run-ash-pq-study.py \
  --data /data/vectors --output /data/jvector-bench/ash-vs-pq
```

For each dataset and algorithm, this builds the index in one JVM, then measures
overquery 1 and 2 with 1 and with 8 physical cores, each in a fresh JVM: 2
construction and 8 query JVMs per dataset. Placement comes from `lscpu`:
construction uses every physical core with memory interleaved across all NUMA
nodes, and queries use the first physical cores of one node, with memory on that
node. Options:

- `--datasets` and `--algorithms` select a subset.
- `--build-smt` lists the algorithms built with SMT (default `ash`).
- `--no-ash-adaptive` searches ASH without adaptive stopping.
- `--query-cores` and `--overqueries` change the query budgets and overqueries.
- `--repetitions` repeats each query JVM, for run-to-run variation.
- `--dry-run` prints the JVM launches without running them.

Rerunning the same command after an interruption reuses indexes that were
already built. Do not run anything else on the machine during a study.

### One JVM by hand

`run-ash-pq-benchmark.sh` runs one JVM of the study; the study script calls it
for every launch. It checks that the CPU list names exactly the requested
physical cores, so two hardware threads of one core cannot count as two cores.

```bash
export JVECTOR_BENCH_DATA_DIR=/data/vectors
CASE=/data/jvector-bench/cohere-2bit-ash

# Construction on 48 physical cores using both threads of each (pick CPUs from lscpu).
JVECTOR_BENCH_BUILD_SMT=1 JVECTOR_BENCH_NUMA_POLICY=interleave:0,1 \
  bash jvector-examples/run-ash-pq-benchmark.sh cohere-2bit-ash.yml construction 48 0-95 "$CASE"

# Queries reuse the saved model and graph, each in a fresh JVM.
JVECTOR_BENCH_ADAPTIVE=1 JVECTOR_BENCH_OVERQUERY=1.0 \
  bash jvector-examples/run-ash-pq-benchmark.sh cohere-2bit-ash.yml queued 1 0 "$CASE"
JVECTOR_BENCH_ADAPTIVE=1 JVECTOR_BENCH_OVERQUERY=1.0 \
  bash jvector-examples/run-ash-pq-benchmark.sh cohere-2bit-ash.yml queued 8 0-7 "$CASE"
```

For the matching PQ case, use `cohere-2bit-pq.yml`, `direct` instead of
`queued`, no SMT or adaptive settings, and its own case directory. The comments
at the top of the script list every option.

## Run a case in one JVM

`benchyaml-play.sh` rebuilds the checkout, then runs one recipe in a single JVM on
one hardware thread of each physical core. Copy
[`bench-profiles/example.env`](../jvector-examples/bench-profiles/example.env) to
`bench-profiles/local/profile.env` and set the JDK, a working directory and the
data directory (see the [benchmark launcher](../jvector-examples/bench-profiles/README.md)).

```bash
BENCH_RUN_CONFIG=jvector-examples/yaml-configs/ash-pq/run-single-jvm.yml BENCH_ADAPTIVE=true \
  jvector-examples/benchyaml-play.sh \
  jvector-examples/yaml-configs/index-parameters/studies/ash-vs-pq-single-jvm/cohere-2bit-ash.yml
```

For PQ, run `cohere-2bit-pq.yml` without `BENCH_ADAPTIVE`. These recipes differ
from the study's in four lines: `construction.workers: max`,
`search.workers: [8]`, overquery `[1.0]` and `useSavedIndexIfExists: No`. Queries are
measured as in the study. Keep one search case per JVM: a second case would repeat
queries the first already ran, so its queries would no longer be cold.

## What is measured

The run configurations in
[`yaml-configs/ash-pq`](../jvector-examples/yaml-configs/ash-pq) select what BenchYAML
measures and records.

- **Construction:** BenchYAML's training, encoding, NVQ and graph-write timers.
  Dataset loading and download are excluded.
- **Queries (warm index, cold queries):** the index stays in the page cache, the
  JVM warms up on base vectors until JIT compilation settles, then each held-out
  query runs once. That single pass is the throughput measurement; recall and
  visit counts come from later passes in the same JVM. Direct execution uses 4 ×
  cores caller threads; queued execution uses at most one worker per core and a
  maximum batch of 8.
- **Recall:** recall@10 against the dataset's ground truth, the same for every
  mode. Concurrent results are checked against single-threaded searches.

BenchYAML writes its CSV files under each case directory's `logging` folder and
prints its tables. The scripts also keep, under `runs`, the source commit, JAR
checksum, index and run YAML, Java version, CPU topology, console output, exit
code and a `command.sh` that reruns the exact Java command.

Hardware, OS scheduling and parallel graph construction cause run-to-run
variation, so reproducing the study means the same protocol and comparable
results, not identical numbers.

## Published results

[ASH performance](ash-performance.md) reports results on 6M and 10M datasets that
are not in the public catalog, measured with an earlier version of this protocol.

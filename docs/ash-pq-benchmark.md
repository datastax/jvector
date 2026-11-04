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

# Reproducing fused PQ versus fused ASH with BenchYAML

This comparison uses the branch's ordinary packaged JVector code and
`io.github.jbellis.jvector.example.BenchYAML`. The fixed index configurations,
run configurations and local dataset catalog are in
[`jvector-examples/yaml-configs/ash-pq`](../jvector-examples/yaml-configs/ash-pq).
No separate query harness, source overlay, profiling agent, manually linked
library, or dataset-specific projection setting is required.

## Build the checkout

Use JDK 22 for performance measurements. Follow the repository's
[native prerequisites](../jvector-native/src/main/native/README.md), including
the compiler, Meson, Ninja and CMake. Native packaging fetches the pinned oneDNN
source and includes the statically linked ASH add-on by default. Reviewers need
the normal build dependencies and access to the input datasets. ASH training
also uses the installed BLAS/LAPACK provider; record that native provider with
the results, since a Java fallback does not reproduce native training throughput.

From a clean checkout of the reported commit:

```bash
export JAVA_HOME=/path/to/jdk-22
export PATH="$JAVA_HOME/bin:$PATH"
git submodule update --init --recursive
mvn -B -pl jvector-examples -am -DskipTests package
```

The standalone artifact is
`jvector-examples/target/jvector-examples-<version>-jar-with-dependencies.jar`.
Build once, before any benchmark process. Keep the checkout and JAR fixed
throughout a comparison. Native is explicitly required by the run command so a
failed native installation cannot silently become a scalar measurement.
Optimized FP16 execution additionally depends on the CPU's capabilities; the
portable FP32 fallback remains available on other systems.

Relevant repository tests can be run independently:

```bash
mvn -B -pl jvector-tests,jvector-examples -am test \
  -DskipSIMD=false -DskipScalar=true \
  '-Dtest=TestASH*,NativeASH*,TestPreparedASHQuery,TestBatchingExecutor,TestCallerBatchProcessor,TestQueuedWorkerLifetime,TestDeferredQueryReporting,QueryExecutionConfigTest,StandaloneASHBenchmarkConfigTest' \
  -Dsurefire.failIfNoSpecifiedTests=false
```

The production change is also tested on JDK 25. Those correctness runs are not
combined with the JDK 22 performance results.

## Inputs and fixed index parameters

Obtain the same normalized fvecs/ivecs inputs listed in
[`catalog-1m.yml`](../jvector-examples/yaml-configs/ash-pq/catalog-1m.yml).
Some datasets require separate access permission; the checkout contains neither
credentials nor private machine paths. Set `JVECTOR_BENCH_DATA_DIR` to their
absolute parent directory, preserving the relative file names in the catalog.
The four "1M" names describe the dataset families; the reporting artifacts
record the actual base/query counts, including the 1,000 OpenAI3072 queries.
All four use `DOT_PRODUCT` and `NO_SCRUB` from `dataset-metadata.yml`.

| Dataset prefix | Input dimension | 2-bit ASH / matching PQ code bytes | 4-bit ASH / matching PQ code bytes |
| --- | ---: | ---: | ---: |
| `mpnet` | 768 | 101 | 197 |
| `cohere` | 1024 | 133 | 261 |
| `ada002` | 1536 | 197 | 389 |
| `openai3072` | 3072 | 389 | 773 |

ASH uses one landmark, an ITQ-trained transform with D/2 output dimensions, and
2 or 4 bits per projected dimension. Its five-byte header is included in the
matched code budget. PQ uses that many 8-bit subspaces, 256 centroids, no
centering, and the default non-anisotropic training setting. The 2-bit/4-bit
labels describe the ASH cases, not PQ centroid precision.

Both algorithms use fused graph features, NVQ reranking, degree 32,
`efConstruction=100`, neighbor overflow 1.2, hierarchy and final refinement,
and top 10 at overquery 1 and 2. Native/oneDNN selection uses production defaults:
FP16 for supported ASH query projection, FP32 for encoding and training, and
Highway scoring. The deferred complete-record writer is enabled for both.

## Run one case

The checked-in shell recipe only prepares a case directory, records the command
and invokes standalone BenchYAML. It contains no query loop or timing logic.
It checks that the CPU list contains the requested number of distinct online
physical cores, so two SMT siblings cannot accidentally count as two cores.
CPU counts are sizing hints; `numactl` enforces the actual allocation.

```bash
export JVECTOR_BENCH_DATA_DIR=/data/normalized-vector-datasets
export JVECTOR_BENCH_HEAP=128g
BENCH_CASE=/data/jvector-bench/mpnet-2bit-ash-r1

# Choose these CPU/node lists from lscpu on the actual host.
# The reference construction budget is 48 physical cores across two NUMA nodes.
JVECTOR_BENCH_NUMA_POLICY=interleave:0,1 \
  bash jvector-examples/run-ash-pq-benchmark.sh \
  mpnet-2bit-ash.yml construction 48 0-47 "$BENCH_CASE"

# Reuse that model and graph. Each invocation is a fresh JVM.
bash jvector-examples/run-ash-pq-benchmark.sh \
  mpnet-2bit-ash.yml direct 1 0 "$BENCH_CASE"
bash jvector-examples/run-ash-pq-benchmark.sh \
  mpnet-2bit-ash.yml queued 1 0 "$BENCH_CASE"
bash jvector-examples/run-ash-pq-benchmark.sh \
  mpnet-2bit-ash.yml direct 8 0-7 "$BENCH_CASE"
bash jvector-examples/run-ash-pq-benchmark.sh \
  mpnet-2bit-ash.yml queued 8 0-7 "$BENCH_CASE"
```

Query memory placement defaults to NUMA node 0. Adapt the physical CPU and
memory-node lists together on other hosts. Do not run cases concurrently or
share their CPU allocation with another experiment. The heap must fit the
machine; report any heap, CPU, NUMA or library differences from the reference.

Replace `ash` with `pq` and use a separate case directory for its matching PQ
case. Repeat with each prefix and bit count in the table. Construction refuses
nonempty model/index caches; query invocations require caches and reject a run
that rebuilt an index. No existing indexes are automatically deleted.

For the full report, collect five fresh construction JVMs per case in separate
directories. Retain the first graph for queries, without choosing the best build.
Run three fresh query JVMs per algorithm, core budget and execution mode against
that first graph. Alternate PQ/ASH and direct/queued order across repetitions.
That is 80 builds and 192 query JVMs; each query JVM reports both overquery levels.

The optional checked-in study driver launches exactly these standalone invocations
sequentially and aggregates their CSV files. It has no query loop or performance
timer. For a host with the reference CPU/node numbering:

```bash
python3 jvector-examples/run-ash-pq-study.py \
  --data "$JVECTOR_BENCH_DATA_DIR" --output /data/jvector-bench/full-study
```

Use `--plan-only` to inspect the full job list. CPU/node arguments are listed by
`--help` for hosts with different topology. `plan.json` fixes the commit, artifact
checksum and settings; `status.json`, `results.json` and `summary.json` are updated
between invocations. Completed invocations can be resumed only with the same
source and artifact. A 30-minute process watchdog detects a hung invocation;
its wall clock is never reported as a performance measurement. Extra construction
repetitions release their model/index caches after recording results. First-build
graphs and all raw reporting artifacts are retained.

## Standalone Java entry point

The shell recipe writes a complete copyable `runs/<invocation>/command.sh`,
including environment settings. Reviewers can execute that Java command directly.
For example, after the case-directory setup above, a one-core queued invocation
has this form (set `REPO_ROOT` and `BENCH_JAR` to the checkout and built artifact):

```bash
cd "$BENCH_CASE"
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
GOTO_NUM_THREADS=1 MKL_DYNAMIC=FALSE ONEDNN_VERBOSE=0 \
numactl --physcpubind=0 --membind=0 "$JAVA_HOME/bin/java" \
  --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -da -Xms2g -Xmx128g -XX:ActiveProcessorCount=1 \
  -Djvector.vectorization_provider=native -Djvector.parallel_writer=true \
  -Djvector.physical_core_count=1 -Djvector.bench.queryThreads=4 \
  -Djvector.bench.verifyConcurrentResults=true -Djvector.bench.throughputMinSeconds=3.0 \
  "-Djvector.bench.sourceRef=$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  "-Djvector.bench.runConfig=$REPO_ROOT/jvector-examples/yaml-configs/ash-pq/run-queued.yml" \
  -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML \
  "$REPO_ROOT/jvector-examples/yaml-configs/ash-pq/mpnet-2bit-ash.yml"
```

The case directory contains copies of the checked-in dataset catalog, dataset
metadata and dataset list because the standard loader resolves them relative to
the working directory. `jvector.bench.runConfig` selects a run YAML without
editing the default. Absolute index-YAML paths work directly. No benchmark
kernel switches or prototype classpaths are used.

## Measurements and artifacts

* **Construction:** use BenchYAML's own inclusive training, encoding, NVQ and
  graph/write timers. Do not add an external timer to these values. The two
  overquery rows in a construction invocation describe one build, not two
  repetitions. Cached-index invocations do not supply fresh construction times.
* **Throughput:** unchanged random-query warmups and measured full query passes,
  including preparation, traversal and reranking. Each QPS sample lasts at least
  three seconds and completes the final pass. Direct mode has 4C synchronous
  callers; queued mode has at most C execution workers and admission 8C, with
  default maximum batch 8. PQ uses the same queued policy without ASH preparation.
* **Unloaded latency:** BenchYAML submits and completes one query at a time,
  with the same configured worker capacity. The timer ends when the call returns,
  including preparation, execution and dispatch. This is not sustained latency
  under an externally controlled arrival rate.
* **Queued loaded response:** supplementary samples during throughput include
  admission and processing through the worker's completion callback. They end
  before the submitting caller resumes. They measure bounded backlog, not a
  Poisson arrival-rate curve; do not compare these percentiles as if they were
  the unloaded measurement. Reporting aggregates completed samples only after
  that query measurement group finishes.
* **Recall:** the same inputs, ground truth and request/result correlation are
  used for both execution modes. The concurrency check also compares completed
  queued results to serial searches. Require no execution-policy recall regression.

CSV results, system information and dataset information are written under each
case's `logging` directory. The helper additionally records the immutable source
reference, JAR checksum, exact index/run YAML, Java version, CPU topology, Java
PID, console output and exit code under `runs`. Retain those artifacts together.
Report QPS and mean-latency means and sample standard deviations across JVMs,
and identify whether tail summaries are per-JVM medians/ranges or pooled samples.
Construction summaries use all five builds and retain their variance.

Reporting deferral removes unnecessary compilation between measurement phases;
it does not guarantee that the JVM will never compile during a timed phase.
There is no compiler-quiescence gate or special warmup of measured query inputs.
Hardware, OS scheduling and parallel graph construction cause run variation;
reproduction means the same protocol and comparable distributions.

## Unloaded-latency warmup correction

Unloaded latency has its own warmup through the same single-request execution
path used by measurement. Throughput's batched warmup alone does not exercise
the synchronous queue handoff/completion path. Before timing, both direct and
queued modes run three complete random-query warmup passes, each for at least
three seconds. The query buffer is reused after each synchronous completion;
measured dataset queries are not used for warmup. The scoring operation and
timed latency interval are unchanged. No compiler-quietness gate is used.

An ASH JFR profile identified compilation of the single-request completion,
queue and FP16 projection paths during the original latency measurement.
Old query results from that protocol must be identified separately. Since the
first-overquery latency phase precedes the next overquery's throughput phase,
corrected comparisons repeat the full query JVM, rather than replacing only
a latency cell. Construction results remain applicable when all construction
classes and native libraries are verified unchanged. Retain the source and
artifact provenance for each construction and query invocation.

#!/usr/bin/env bash
# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Runs one JVM of the ASH versus PQ study (docs/ash-pq-benchmark.md): either the
# construction of one case, or one query measurement against the index that
# construction saved. run-ash-pq-study.py calls this once per JVM for a whole study.
#
#   run-ash-pq-benchmark.sh <case.yml> <construction|direct|queued> <physical-cores> <cpu-list> <case-directory>
#
#   case.yml        a file name in yaml-configs/index-parameters/studies/ash-vs-pq
#   mode            construction builds and saves the model and graph; direct or queued
#                   (whichever the YAML's search.queryExecution selects) queries them
#   physical-cores  the CPU budget: the JVM sees this many processors
#   cpu-list        the CPUs it may run on, one per physical core (both threads of each
#                   core with JVECTOR_BENCH_BUILD_SMT=1), for example 0-7
#   case-directory  holds the case's saved model, graph, logs and per-run records;
#                   use a new directory for each construction
#
# Required environment:
#   JVECTOR_BENCH_DATA_DIR  absolute directory for the datasets. They are fetched from the
#                           public JVector dataset catalog on first use and reused afterwards.
# Optional environment:
#   JVECTOR_BENCH_BUILD_SMT=1      construction uses both hardware threads of each core
#   JVECTOR_BENCH_ADAPTIVE=1       searches use adaptive stopping
#   JVECTOR_BENCH_OVERQUERY=<x>    query runs measure only this overquery (for example 2.0),
#                                  so the timed pass is the first search the JVM makes
#   JVECTOR_BENCH_NUMA_POLICY      bind:<nodes> (default bind:0) or interleave:<nodes>
#   JVECTOR_BENCH_HEAP             maximum Java heap (default 128g)
#
# Requires Linux with numactl and lscpu, a clean git checkout, and one examples JAR built
# with Maven (see the doc). BenchYAML writes its CSV under the case directory's logging/
# folder and prints its tables; this script adds only a record of how each JVM was run.
set -euo pipefail

if [[ $# != 5 ]]; then
  echo "Usage: $0 <case.yml> <construction|direct|queued> <physical-cores> <cpu-list> <case-directory>" >&2
  echo "Example: $0 cohere-2bit-ash.yml queued 8 0-7 /data/bench/cohere-2bit-ash" >&2
  exit 2
fi
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
CONFIG_NAME="$1"
MODE="$2"
CORES="$3"
CPU_LIST="$4"
CASE_DIRECTORY="$5"
case "$MODE" in construction|direct|queued) ;; *) echo "Invalid mode: $MODE" >&2; exit 2;; esac
if ! [[ "$CORES" =~ ^[1-9][0-9]*$ ]]; then echo "Physical core count must be positive" >&2; exit 2; fi
for command in java numactl lscpu sha256sum awk; do
  command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 2; }
done
if ! [[ "$CPU_LIST" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
  echo "CPU list must contain CPU numbers or ranges, such as 0 or 0-7." >&2
  exit 2
fi
# JVECTOR_BENCH_BUILD_SMT=1 builds on both hardware threads of each budgeted physical
# core (2 x cores threads). Queries always use one thread per physical core.
THREADS="$CORES"
if [[ "${JVECTOR_BENCH_BUILD_SMT:-0}" == 1 ]]; then
  [[ "$MODE" == construction ]] || { echo "JVECTOR_BENCH_BUILD_SMT applies to construction only" >&2; exit 2; }
  THREADS=$((2 * CORES))
fi
lscpu --parse=CPU,CORE,SOCKET,ONLINE | awk -F, -v selection="$CPU_LIST" -v budget="$CORES" -v threads="$THREADS" '
  BEGIN {
    count = split(selection, pieces, ",")
    for (i = 1; i <= count; i++) {
      n = split(pieces[i], bounds, "-")
      first = bounds[1] + 0; last = n == 1 ? first : bounds[2] + 0
      if (last < first || last - first + 1 > threads) { bad = 1; exit 2 }
      for (cpu = first; cpu <= last; cpu++) chosen[cpu] = 1
    }
  }
  !/^#/ && ($1 in chosen) {
    if ($4 == "N") { bad = 1; exit 2 }
    seen[$1] = 1; physical[$3 ":" $2] = 1
  }
  END {
    for (cpu in chosen) wanted++
    for (cpu in seen) found++
    for (core in physical) cores++
    if (bad || wanted != threads || found != threads || cores != budget) {
      print "CPU selection must contain exactly the budgeted physical cores (and, with SMT builds, both threads of each)." > "/dev/stderr"
      exit 2
    }
  }'
DATA_ROOT="${JVECTOR_BENCH_DATA_DIR:-}"
if [[ "$DATA_ROOT" != /* ]]; then
  echo "Set JVECTOR_BENCH_DATA_DIR to an absolute directory for the datasets." >&2
  exit 2
fi
mkdir -p "$DATA_ROOT"
if [[ -n "${JAVA_TOOL_OPTIONS:-}" || -n "${JDK_JAVA_OPTIONS:-}" ]]; then
  echo "Clear JAVA_TOOL_OPTIONS/JDK_JAVA_OPTIONS so the recorded Java command is complete." >&2
  exit 2
fi
git -C "$REPO_ROOT" diff --quiet
git -C "$REPO_ROOT" diff --cached --quiet
SOURCE_REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"
JAVA_BIN="$(command -v java)"
CONFIG="$REPO_ROOT/jvector-examples/yaml-configs/index-parameters/studies/ash-vs-pq/$CONFIG_NAME"
RUN_CONFIG="$REPO_ROOT/jvector-examples/yaml-configs/ash-pq/run-$MODE.yml"
CATALOG="$REPO_ROOT/jvector-examples/yaml-configs/dataset-catalogs/public-catalog.yaml"
[[ -f "$CONFIG" && "$CONFIG_NAME" != */* ]] || {
  echo "Unknown index config: $CONFIG_NAME" >&2; exit 2;
}

if [[ "$MODE" != construction ]]; then
  # The index YAML selects query execution at the search level.
  EXECUTION="$(sed -nE 's/^[[:space:]]+queryExecution:[[:space:]]*(direct|queued)[[:space:]]*(#.*)?$/\1/p' "$CONFIG")"
  if [[ "${EXECUTION:-direct}" != "$MODE" ]]; then
    echo "The index YAML selects ${EXECUTION:-direct}. Use its matching query report selection." >&2
    exit 2
  fi
fi

shopt -s nullglob
JARS=("$REPO_ROOT"/jvector-examples/target/jvector-examples-*-jar-with-dependencies.jar)
if [[ ${#JARS[@]} != 1 ]]; then
  echo "Build exactly one examples assembly with Maven; see docs/ash-pq-benchmark.md." >&2
  exit 2
fi
JAR="${JARS[0]}"
mkdir -p "$CASE_DIRECTORY"
CASE_DIRECTORY="$(cd "$CASE_DIRECTORY" && pwd)"
CASE_ID="$(sha256sum "$CONFIG" | cut -d ' ' -f 1)"
if [[ -f "$CASE_DIRECTORY/case-id.txt" && "$(cat "$CASE_DIRECTORY/case-id.txt")" != "$CASE_ID" ]]; then
  echo "Case directory belongs to a different index configuration; use its matching YAML." >&2
  exit 2
fi
if [[ "$MODE" == construction ]]; then
  for cache in compressor_cache index_cache; do
    if [[ -d "$CASE_DIRECTORY/$cache" ]] && [[ -n "$(ls -A "$CASE_DIRECTORY/$cache")" ]]; then
      echo "Fresh construction requires empty caches; use a new case directory." >&2
      exit 2
    fi
  done
else
  [[ -d "$CASE_DIRECTORY/index_cache" && -d "$CASE_DIRECTORY/compressor_cache" ]] || {
    echo "Construct this case first; query runs must reuse its model and graph." >&2; exit 2;
  }
fi

# BenchYAML reads dataset metadata and catalogs relative to its working directory.
# Install only the public catalog, with its download cache pointed at
# JVECTOR_BENCH_DATA_DIR, so every case shares one copy of each dataset.
CASE_CONFIG="$CASE_DIRECTORY/jvector-examples/yaml-configs"
mkdir -p "$CASE_CONFIG/dataset-catalogs"
cp "$REPO_ROOT/jvector-examples/yaml-configs/dataset-metadata.yml" "$CASE_CONFIG/"
cp "$REPO_ROOT/jvector-examples/yaml-configs/datasets.yml" "$CASE_CONFIG/"
sed "s|\${DATASET_CACHE_DIR:-dataset_cache}|$DATA_ROOT|g" "$CATALOG" > "$CASE_CONFIG/dataset-catalogs/public-catalog.yaml"
printf '%s\n' "$CASE_ID" > "$CASE_DIRECTORY/case-id.txt"
mkdir -p "$CASE_DIRECTORY/runs"
RUN_DIRECTORY="$(mktemp -d "$CASE_DIRECTORY/runs/$MODE-c$CORES-XXXXXX")"
cp "$CONFIG" "$RUN_DIRECTORY/index.yml"
# JVECTOR_BENCH_OVERQUERY=<x> runs one search case per query JVM, so its single timed
# pass is the first search the JVM makes. The case id stays that of the full index
# YAML; only the search sweep is narrowed.
LAUNCH_CONFIG="$CONFIG"
if [[ -n "${JVECTOR_BENCH_OVERQUERY:-}" ]]; then
  [[ "$MODE" != construction ]] || { echo "JVECTOR_BENCH_OVERQUERY applies to query runs only" >&2; exit 2; }
  [[ "$JVECTOR_BENCH_OVERQUERY" =~ ^[0-9]+\.[0-9]+$ ]] || { echo "JVECTOR_BENCH_OVERQUERY must look like 1.0" >&2; exit 2; }
  grep -qE "^    10: \[1\.0, 2\.0\]$" "$CONFIG" || { echo "Index YAML lacks the expected overquery sweep" >&2; exit 2; }
  LAUNCH_CONFIG="$RUN_DIRECTORY/index-overquery-$JVECTOR_BENCH_OVERQUERY.yml"
  sed -E "s/^    10: \[1\.0, 2\.0\]$/    10: [$JVECTOR_BENCH_OVERQUERY]/" "$CONFIG" > "$LAUNCH_CONFIG"
fi
cp "$RUN_CONFIG" "$RUN_DIRECTORY/run.yml"
cp "$CASE_CONFIG/dataset-catalogs/public-catalog.yaml" "$RUN_DIRECTORY/catalog.yaml"
printf '%s\n' "$SOURCE_REF" > "$RUN_DIRECTORY/source-ref.txt"
sha256sum "$JAR" > "$RUN_DIRECTORY/jar.sha256"
"$JAVA_BIN" -version > "$RUN_DIRECTORY/java-version.txt" 2>&1
lscpu --extended=CPU,CORE,SOCKET,NODE,ONLINE > "$RUN_DIRECTORY/cpu-topology.txt"

# Native query kernels execute on the Java search workers. Construction has
# its own CPU budget for training and encoding. Avoid nested query thread pools.
NATIVE_THREADS=1
if [[ "$MODE" == construction ]]; then NATIVE_THREADS="$THREADS"; fi
export MKL_NUM_THREADS="$NATIVE_THREADS" OMP_NUM_THREADS="$NATIVE_THREADS"
export OPENBLAS_NUM_THREADS="$NATIVE_THREADS" GOTO_NUM_THREADS="$NATIVE_THREADS"
export MKL_DYNAMIC=FALSE ONEDNN_VERBOSE=0

NUMA_POLICY="${JVECTOR_BENCH_NUMA_POLICY:-bind:0}"
case "$NUMA_POLICY" in
  bind:*) MEMORY_ARGS=("--membind=${NUMA_POLICY#bind:}");;
  interleave:*) MEMORY_ARGS=("--interleave=${NUMA_POLICY#interleave:}");;
  *) echo "JVECTOR_BENCH_NUMA_POLICY must be bind:<nodes> or interleave:<nodes>" >&2; exit 2;;
esac
JAVA_ARGS=(--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
  -da -Xms2g "-Xmx${JVECTOR_BENCH_HEAP:-128g}" "-XX:ActiveProcessorCount=$THREADS"
  -Djvector.vectorization_provider=native -Djvector.parallel_writer=true
  "-Djvector.physical_core_count=$THREADS" "-Djvector.bench.queryThreads=$((4 * THREADS))"
  -Djvector.bench.verifyConcurrentResults=true
  "-Djvector.bench.adaptive=$([[ "${JVECTOR_BENCH_ADAPTIVE:-0}" == 1 ]] && echo true || echo false)"
  "-Djvector.bench.sourceRef=$SOURCE_REF" "-Djvector.bench.runConfig=$RUN_CONFIG"
  -cp "$JAR" io.github.jbellis.jvector.example.BenchYAML "$LAUNCH_CONFIG")
LAUNCH=(numactl "--physcpubind=$CPU_LIST" "${MEMORY_ARGS[@]}" "$JAVA_BIN" "${JAVA_ARGS[@]}")
{
  printf 'cd %q\n' "$CASE_DIRECTORY"
  printf 'JVECTOR_BENCH_DATA_DIR=%q ' "$DATA_ROOT"
  printf 'MKL_NUM_THREADS=%q OMP_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q GOTO_NUM_THREADS=%q MKL_DYNAMIC=FALSE ONEDNN_VERBOSE=0 ' \
    "$MKL_NUM_THREADS" "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$GOTO_NUM_THREADS"
  printf '%q ' "${LAUNCH[@]}"
  printf '\n'
} > "$RUN_DIRECTORY/command.sh"
echo "Artifacts: $RUN_DIRECTORY"
cd "$CASE_DIRECTORY"
"${LAUNCH[@]}" > "$RUN_DIRECTORY/console.log" 2>&1 &
JVM_PID=$!
printf '%s\n' "$JVM_PID" > "$RUN_DIRECTORY/jvm.pid"
trap 'kill "$JVM_PID" 2>/dev/null || true; wait "$JVM_PID" 2>/dev/null || true; exit 130' INT TERM
STATUS=0
wait "$JVM_PID" || STATUS=$?
trap - INT TERM
cat "$RUN_DIRECTORY/console.log"
printf '%s\n' "$STATUS" > "$RUN_DIRECTORY/exit-code.txt"
if [[ "$STATUS" != 0 ]]; then exit "$STATUS"; fi
if [[ "$MODE" != construction ]] && grep -q 'Building graph index' "$RUN_DIRECTORY/console.log"; then
  echo "Query invocation rebuilt an index; do not treat this as a cached-index result." >&2
  exit 1
fi

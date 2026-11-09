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
set -euo pipefail

if [[ $# != 5 ]]; then
  echo "Usage: $0 <case.yml> <construction|direct|queued> <physical-cores> <cpu-list> <case-directory>" >&2
  echo "Example: $0 mpnet-2bit-ash.yml queued 8 0-7 /data/bench/mpnet-2bit-ash" >&2
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
lscpu --parse=CPU,CORE,SOCKET,ONLINE | awk -F, -v selection="$CPU_LIST" -v budget="$CORES" '
  BEGIN {
    count = split(selection, pieces, ",")
    for (i = 1; i <= count; i++) {
      n = split(pieces[i], bounds, "-")
      first = bounds[1] + 0; last = n == 1 ? first : bounds[2] + 0
      if (last < first || last - first + 1 > budget) { bad = 1; exit 2 }
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
    if (bad || wanted != budget || found != budget || cores != budget) {
      print "CPU selection must contain exactly the budgeted number of distinct online physical cores." > "/dev/stderr"
      exit 2
    }
  }'
DATA_ROOT="${JVECTOR_BENCH_DATA_DIR:-}"
if [[ "$DATA_ROOT" != /* || ! -d "$DATA_ROOT" ]]; then
  echo "Set JVECTOR_BENCH_DATA_DIR to the absolute local dataset directory; see the catalog template." >&2
  exit 2
fi
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
CATALOG_NAME="${JVECTOR_BENCH_CATALOG:-catalog-1m.yml}"
case "$CATALOG_NAME" in catalog-1m.yml|catalog-large.yml) ;; *) echo "Unknown dataset catalog: $CATALOG_NAME" >&2; exit 2;; esac
CATALOG="$REPO_ROOT/jvector-examples/yaml-configs/ash-pq/$CATALOG_NAME"
[[ -f "$CATALOG" ]] || { echo "Dataset catalog not found: $CATALOG" >&2; exit 2; }
[[ -f "$CONFIG" && "$CONFIG_NAME" != run-* && "$CONFIG_NAME" != catalog-* && "$CONFIG_NAME" != */* ]] || {
  echo "Unknown index config: $CONFIG_NAME" >&2; exit 2;
}

if [[ "$MODE" != construction ]]; then
  # The checked-in recipes put delivery under the ASH search parameters.
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

# The standard loader resolves metadata/catalogs relative to the working
# directory. Install only the checked-in local mapping, avoiding ambiguous
# duplicate aliases from other dataset catalogs. Numerical code is all in the JAR.
CASE_CONFIG="$CASE_DIRECTORY/jvector-examples/yaml-configs"
mkdir -p "$CASE_CONFIG/dataset-catalogs"
cp "$REPO_ROOT/jvector-examples/yaml-configs/dataset-metadata.yml" "$CASE_CONFIG/"
cp "$REPO_ROOT/jvector-examples/yaml-configs/datasets.yml" "$CASE_CONFIG/"
# Remove only the catalog name created by the original version of this recipe.
rm -f "$CASE_CONFIG/dataset-catalogs/ash-pq-1m.yml"
cp "$CATALOG" "$CASE_CONFIG/dataset-catalogs/ash-pq-local.yml"
printf '%s\n' "$CASE_ID" > "$CASE_DIRECTORY/case-id.txt"
mkdir -p "$CASE_DIRECTORY/runs"
RUN_DIRECTORY="$(mktemp -d "$CASE_DIRECTORY/runs/$MODE-c$CORES-XXXXXX")"
cp "$CONFIG" "$RUN_DIRECTORY/index.yml"
cp "$RUN_CONFIG" "$RUN_DIRECTORY/run.yml"
cp "$CATALOG" "$RUN_DIRECTORY/catalog.yml"
printf '%s\n' "$SOURCE_REF" > "$RUN_DIRECTORY/source-ref.txt"
sha256sum "$JAR" > "$RUN_DIRECTORY/jar.sha256"
"$JAVA_BIN" -version > "$RUN_DIRECTORY/java-version.txt" 2>&1
lscpu --extended=CPU,CORE,SOCKET,NODE,ONLINE > "$RUN_DIRECTORY/cpu-topology.txt"

# Native query kernels execute on the Java search workers. Construction has
# its own CPU budget for training and encoding. Avoid nested query thread pools.
NATIVE_THREADS=1
if [[ "$MODE" == construction ]]; then NATIVE_THREADS="$CORES"; fi
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
  -da -Xms2g "-Xmx${JVECTOR_BENCH_HEAP:-128g}" "-XX:ActiveProcessorCount=$CORES"
  -Djvector.vectorization_provider=native -Djvector.parallel_writer=true
  "-Djvector.physical_core_count=$CORES" "-Djvector.bench.queryThreads=$((4 * CORES))"
  -Djvector.bench.verifyConcurrentResults=true -Djvector.bench.throughputMinSeconds=3.0
  "-Djvector.bench.sourceRef=$SOURCE_REF" "-Djvector.bench.runConfig=$RUN_CONFIG"
  -cp "$JAR" io.github.jbellis.jvector.example.BenchYAML "$CONFIG")
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

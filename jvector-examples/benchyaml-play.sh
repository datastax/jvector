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

# Build BenchYAML from this checkout, then run one index-parameters YAML in one JVM.
# Every launch rebuilds Java and native code, so a run always reflects the current
# source. The YAML's construction.workers and search.workers set concurrency; this
# script sets where the JVM may run. See jvector-examples/bench-profiles/README.md.
#
#   benchyaml-play.sh <index.yml>    build, then run
#   benchyaml-play.sh --build-only   build and print JAR=, SOURCE_REF= and JAR_SHA256=
#
# BENCH_RUN_CONFIG selects the run YAML (default jvector-examples/yaml-configs/run-config.yml).
# Machine settings come from the server profile (BENCH_PROFILE, or bench-profiles/local/profile.env).
set -euo pipefail

repo=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
profiles="$repo/jvector-examples/bench-profiles"

profile=${BENCH_PROFILE:-$profiles/local/profile.env}
[[ $profile == */* ]] || profile="$profiles/local/$profile.env"
[[ -f $profile ]] || { echo "Server profile not found: $profile. Copy bench-profiles/example.env to bench-profiles/local/profile.env and edit it." >&2; exit 2; }
# shellcheck source=/dev/null
source "$profile"
jdk=${BENCH_JDK:?The server profile must set BENCH_JDK.}

# Maven and native build output goes to build.log under the work root; the console gets one line.
build() {
  local marker start jars log
  marker=$(mktemp)
  start=$(date +%s)
  log="${BENCH_WORK_ROOT:-$repo/jvector-examples/target}/build.log"
  mkdir -p "$(dirname "$log")"
  printf 'Building BenchYAML from %s ... ' "$repo" >&2
  if ! (cd "$repo" && JAVA_HOME="$jdk" PATH="$jdk/bin:$PATH" JAVA_TOOL_OPTIONS= JDK_JAVA_OPTIONS= \
        mvn -B -q package -DskipTests -pl jvector-examples -am) > "$log" 2>&1; then
    printf 'failed. Last lines of %s:\n' "$log" >&2
    tail -40 "$log" >&2
    exit 1
  fi
  printf 'done in %s s (log: %s)\n' "$(( $(date +%s) - start ))" "$log" >&2
  mapfile -t jars < <(find "$repo/jvector-examples/target" -maxdepth 1 -name '*-jar-with-dependencies.jar' -newer "$marker")
  rm -f "$marker"
  if [[ ${#jars[@]} != 1 ]]; then
    echo "Expected the build to write one jar-with-dependencies in jvector-examples/target; found ${#jars[@]}." >&2
    exit 1
  fi
  JAR=${jars[0]}
  SOURCE_REF=$(git -C "$repo" rev-parse HEAD)
  [[ -z $(git -C "$repo" status --porcelain --untracked-files=no) ]] || SOURCE_REF="$SOURCE_REF-dirty"
  JAR_SHA256=$(sha256sum "$JAR" | cut -d' ' -f1)
}

if [[ ${1:-} == --build-only ]]; then
  build
  printf 'JAR=%q\nSOURCE_REF=%q\nJAR_SHA256=%q\n' "$JAR" "$SOURCE_REF" "$JAR_SHA256"
  exit 0
fi

index=${1:?Pass an index-parameters YAML file.}
[[ $index == /* ]] || index="$repo/$index"
[[ -f $index ]] || { echo "Index config not found: $index" >&2; exit 2; }
run_config=${BENCH_RUN_CONFIG:-jvector-examples/yaml-configs/run-config.yml}
[[ $run_config == /* ]] || run_config="$repo/$run_config"
[[ -f $run_config ]] || { echo "Run config not found: $run_config" >&2; exit 2; }
work_root=${BENCH_WORK_ROOT:?The server profile must set BENCH_WORK_ROOT.}

# One hardware thread on each of the first BENCH_CORES physical cores, in CPU order.
mapfile -t physical < <(lscpu -p=CPU,CORE,SOCKET | awk -F, '!/^#/ && !seen[$3":"$2]++ {print $1}')
cores=${BENCH_CORES:-${#physical[@]}}
(( cores >= 1 && cores <= ${#physical[@]} )) || { echo "BENCH_CORES=$cores must be between 1 and the ${#physical[@]} physical cores." >&2; exit 2; }
cpus=$(printf '%s\n' "${physical[@]:0:cores}" | paste -sd, -)
# Print CPU lists as ranges, for example 0-47.
cpu_ranges=$(printf '%s\n' "${physical[@]:0:cores}" | sort -n | awk 'NR==1{a=b=$1; next} $1==b+1{b=$1; next} {printf "%s%s,", a, (a==b?"":"-"b); a=b=$1} END{printf "%s%s", a, (a==b?"":"-"b)}')
nodes=${BENCH_NODES:-$(lscpu -p=NODE | grep -v '^#' | sort -un | paste -sd, -)}

# Each recipe gets its own working directory, so caches and logs never mix between experiments.
# Unsaved indexes are written and reopened for queries under BENCH_TMP_DIR (java.io.tmpdir),
# by default the experiment's tmp directory on the same storage as BENCH_WORK_ROOT.
recipe=${index#"$repo/jvector-examples/yaml-configs/index-parameters/"}
[[ $recipe != /* ]] || recipe="other/$(basename "$recipe")"
work="$work_root/${recipe%.yml}"

mkdir -p "$work_root"
exec 9>"$work_root/benchyaml.lock"
flock -n 9 || { echo "Another BenchYAML launch holds $work_root/benchyaml.lock." >&2; exit 1; }
if pgrep -f '[.]example[.]BenchYAML' >/dev/null; then echo "A BenchYAML JVM is already running on this machine." >&2; exit 1; fi

build

# BenchYAML reads dataset metadata and catalogs relative to its working directory.
catalogs="$work/jvector-examples/yaml-configs/dataset-catalogs"
tmp=${BENCH_TMP_DIR:-$work/tmp}
mkdir -p "$catalogs" "$tmp"
cp "$repo/jvector-examples/yaml-configs/dataset-metadata.yml" "$repo/jvector-examples/yaml-configs/datasets.yml" "$work/jvector-examples/yaml-configs/"
find "$catalogs" -mindepth 1 -delete
# A recipe that does not save its index also starts without saved models, so every run
# measures training and encoding. Recipes that reuse indexes keep both caches.
if ! grep -Eq '^[[:space:]]+useSavedIndexIfExists:[[:space:]]*Yes' "$index"; then
  for cache in compressor_cache index_cache; do
    if [[ -d $work/$cache ]]; then find "$work/$cache" -mindepth 1 -delete; fi
  done
fi
n=0
while read -r catalog data; do
  [[ -n $catalog ]] || continue
  [[ $catalog == /* ]] || catalog="$repo/$catalog"
  n=$((n + 1))
  sed "s|\${JVECTOR_BENCH_DATA_DIR}|$data|g" "$catalog" > "$catalogs/$n-$(basename "${catalog%.*}").yml"
done <<< "${BENCH_CATALOGS:?The server profile must list BENCH_CATALOGS.}"

printf '\nRecipe   %s\nCPUs     %s (%s physical cores), memory interleaved on NUMA nodes %s\nSource   %s\nJAR      %s\n         sha256 %s\nRun YAML %s\nWork     %s\nIndexes  %s\n\n' \
  "$recipe" "$cpu_ranges" "$cores" "$nodes" "$SOURCE_REF" "$JAR" "$JAR_SHA256" "$run_config" "$work" "$tmp"
debug=()
[[ -z ${BENCH_DEBUG:-} ]] || debug=("-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=*:$BENCH_DEBUG")
cd "$work"
# JVector sets BLAS threads for its own calls (one, or the construction pool's size), so the
# BLAS thread variables are cleared rather than set here.
exec env -u JAVA_TOOL_OPTIONS -u JDK_JAVA_OPTIONS -u _JAVA_OPTIONS \
  -u MKL_NUM_THREADS -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS -u GOTO_NUM_THREADS \
  MKL_DYNAMIC=FALSE ONEDNN_VERBOSE=0 \
  numactl --physcpubind="$cpus" --interleave="$nodes" "$jdk/bin/java" "${debug[@]}" \
  --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -da -Xms2g "-Xmx${BENCH_HEAP:-128g}" \
  "-XX:ActiveProcessorCount=$cores" "-Djava.io.tmpdir=$tmp" '-Djava.util.logging.SimpleFormatter.format=%5$s%6$s%n' \
  -Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8 \
  -Djvector.vectorization_provider=native -Djvector.parallel_writer=true \
  "-Djvector.physical_core_count=$cores" \
  -Djvector.bench.verifyConcurrentResults=true \
  "-Djvector.bench.sourceRef=$SOURCE_REF" "-Djvector.bench.runConfig=$run_config" \
  -cp "$JAR" io.github.jbellis.jvector.example.BenchYAML "$index"

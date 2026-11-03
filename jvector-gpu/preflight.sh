#!/usr/bin/env bash
#
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
#
# Checks that everything jvector-gpu/run.sh needs is in place, before a long run:
# the JDK, TornadoVM, the GPU and its memory, cuVS, the built jars, host memory and disk, and the dataset.
#
#   jvector-gpu/preflight.sh [dataset]      e.g. jvector-gpu/preflight.sh ada002-1M
#
# Prints one line per check (OK, WARN or FAIL) and exits 1 if any check fails.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET="${1:-}"
CUVS_DIR="${JVECTOR_GPU_CUVS:-$HOME/.jvector-gpu/cuvs}"
MIN_JDK=22
MIN_TORNADO=8.0.0
MIN_COMPUTE=8.0         # Ampere or newer
MIN_DRIVER=525          # CUDA 12 (cuVS cu12 wheels)
HEAP_GB=48              # run.sh's default -Xmx

fails=0
warns=0
ok()   { printf '  OK    %-12s %s\n' "$1" "$2"; }
warn() { printf '  WARN  %-12s %s\n' "$1" "$2"; warns=$((warns + 1)); }
fail() { printf '  FAIL  %-12s %s\n' "$1" "$2"; fails=$((fails + 1)); }
# version_ge A B: A >= B for dotted versions
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }
gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1024 / 1024 / 1024 }'; }

echo "jvector-gpu preflight${DATASET:+ for $DATASET}"

# --- JDK: Maven and the tornado launcher both use JAVA_HOME
JAVA_BIN="${JAVA_HOME:+$JAVA_HOME/bin/}java"
if ! command -v "$JAVA_BIN" >/dev/null 2>&1; then
    fail jdk "no java found (set JAVA_HOME to JDK $MIN_JDK or newer)"
else
    jv=$("$JAVA_BIN" -XshowSettings:properties -version 2>&1 | awk -F'= ' '/java.specification.version/ {print $2}')
    if [ -z "${JAVA_HOME:-}" ]; then
        warn jdk "JAVA_HOME is not set; the tornado launcher needs it (java on the PATH is $jv)"
    elif [ "${jv%%.*}" -ge "$MIN_JDK" ] 2>/dev/null; then
        ok jdk "JDK $jv at $JAVA_HOME"
    else
        fail jdk "JDK $jv at $JAVA_HOME; need $MIN_JDK or newer (JDK 21 fails with 'release version 22 not supported')"
    fi
fi

# --- TornadoVM: version, CUDA backend, cuVS provider
if ! command -v tornado >/dev/null 2>&1; then
    fail tornadovm "'tornado' is not on the PATH (TornadoVM $MIN_TORNADO+ with the CUDA backend: add \$TORNADOVM_HOME/bin)"
else
    TVM_HOME="${TORNADOVM_HOME:-$(cd "$(dirname "$(command -v tornado)")/.." && pwd)}"
    tv=$(tornado --version 2>/dev/null | awk -F= '/^version=/ {print $2}')
    backends=$(awk -F= '/^tornado.backends=/ {print $2}' "$TVM_HOME"/etc/*backend* 2>/dev/null | head -1)
    cuvs_jar=$(ls "$TVM_HOME"/share/java/tornado/tornado-cuvs-*.jar 2>/dev/null | head -1)
    if [ -z "$tv" ]; then
        fail tornadovm "could not read the version from 'tornado --version' ($TVM_HOME)"
    elif ! version_ge "$tv" "$MIN_TORNADO"; then
        fail tornadovm "TornadoVM $tv at $TVM_HOME; need $MIN_TORNADO or newer (first release with tornado-cuvs)"
    elif [[ "$backends" != *cuda* ]]; then
        fail tornadovm "TornadoVM $tv has backends '${backends:-none}'; need the CUDA backend (the -cuda SDK)"
    elif [ -z "$cuvs_jar" ]; then
        fail tornadovm "TornadoVM $tv has no tornado-cuvs jar in share/java/tornado"
    else
        ok tornadovm "TornadoVM $tv, backends $backends, $(basename "$cuvs_jar")"
    fi
fi

# --- GPU: NVIDIA, compute capability, driver, memory
GPU_FREE_MB=0
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    fail gpu "no NVIDIA GPU or driver found (nvidia-smi failed)"
else
    IFS=, read -r name cc driver total free < <(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.free \
        --format=csv,noheader,nounits | head -1 | sed 's/, */,/g')
    GPU_FREE_MB=$free
    count=$(nvidia-smi -L | wc -l)
    if ! version_ge "$cc" "$MIN_COMPUTE"; then
        fail gpu "$name has compute capability $cc; cuVS needs $MIN_COMPUTE (Ampere) or newer"
    else
        ok gpu "$name, compute capability $cc${count:+ (GPU 0 of $count)}"
    fi
    if version_ge "$driver" "$MIN_DRIVER"; then
        ok driver "NVIDIA driver $driver"
    else
        fail driver "NVIDIA driver $driver; cuVS (CUDA 12) needs $MIN_DRIVER or newer"
    fi
    used=$((total - free))
    if [ "$used" -gt $((total / 10)) ]; then
        warn gpu-memory "$free MiB free of $total MiB: other processes hold $used MiB, and run.sh budgets 90% of the total for TornadoVM"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/                     /'
    else
        ok gpu-memory "$free MiB free of $total MiB"
    fi
fi

# --- cuVS (downloaded by cuvs-env.sh on the first run, ~2.1 GB)
if [ -s "$CUVS_DIR/ld_library_path" ]; then
    missing=$(tr ':' '\n' < "$CUVS_DIR/ld_library_path" | while read -r d; do [ -d "$d" ] || echo "$d"; done | head -1)
    v=$(ls -d "$CUVS_DIR"/libcuvs_cu12-*.dist-info 2>/dev/null | sed 's/.*libcuvs_cu12-\(.*\)\.dist-info/\1/' | head -1)
    if [ -n "$missing" ]; then
        fail cuvs "$CUVS_DIR is incomplete ($missing is gone); delete it and run again"
    else
        ok cuvs "cuVS ${v:-?} in $CUVS_DIR"
    fi
else
    # df on the nearest directory that exists: on a fresh machine ~/.jvector-gpu does not yet
    parent="$CUVS_DIR"
    while [ ! -d "$parent" ] && [ "$parent" != "/" ]; do parent=$(dirname "$parent"); done
    free_kb=$(df -Pk "$parent" 2>/dev/null | awk 'NR==2 {print $4}')
    if ! python3 -m pip --version >/dev/null 2>&1; then
        fail cuvs "not installed in $CUVS_DIR, and python3 -m pip is missing to download it"
    elif [ "${free_kb:-0}" -lt $((4 * 1024 * 1024)) ]; then
        fail cuvs "not installed; the download needs ~4 GB free near $CUVS_DIR, only $(gb $((free_kb * 1024))) free"
    else
        warn cuvs "not installed yet: the first run downloads ~2.1 GB into $CUVS_DIR (needs network)"
    fi
fi

# --- build tools: the default build compiles jvector-native (meson, ninja, g++ and the Highway submodule)
for tool in meson ninja; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        fail build-tools "$tool is missing; install it with: python3 -m pip install --user meson ninja"
    fi
done
CXX_BIN="${CXX:-g++}"
min_gcc=11
[ "$(uname -m)" = "x86_64" ] && min_gcc=12   # Highway's AVX3_SPR target needs GCC 12 or newer
if ! command -v "$CXX_BIN" >/dev/null 2>&1; then
    fail build-tools "no C++ compiler ($CXX_BIN); need GCC $min_gcc or newer"
else
    gcc_v=$("$CXX_BIN" -dumpfullversion -dumpversion 2>/dev/null | head -1)
    if version_ge "$gcc_v" "$min_gcc"; then
        command -v meson >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1 \
            && ok build-tools "meson $(meson --version), ninja $(ninja --version), $CXX_BIN $gcc_v"
    else
        fail build-tools "$CXX_BIN is GCC $gcc_v; jvector-native needs GCC $min_gcc or newer (RHEL/Alma 9: dnf install gcc-toolset-14 && source /opt/rh/gcc-toolset-14/enable)"
    fi
fi
if [ ! -f "$ROOT/jvector-native/src/main/native/third_party/highway/hwy/highway.h" ]; then
    fail build-tools "the Highway submodule is missing: git submodule update --init"
fi

# --- the built module
GPU_JAR=$(ls "$ROOT"/jvector-gpu/target/jvector-gpu-*.jar 2>/dev/null | grep -v -e javadoc -e sources | head -1)
EXAMPLES_JAR=$(ls "$ROOT"/jvector-examples/target/jvector-examples-*-jar-with-dependencies.jar 2>/dev/null | head -1)
if [ -n "$GPU_JAR" ] && [ -n "$EXAMPLES_JAR" ]; then
    ok build "$(basename "$GPU_JAR"), $(basename "$EXAMPLES_JAR")"
else
    fail build "not built: mvn -Pgpu -pl jvector-gpu,jvector-examples -am package -DskipTests"
fi

# --- host memory and disk
MEM_TOTAL_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
MEM_AVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
heap_gb=$HEAP_GB
xmx=$(grep -o -- '-Xmx[0-9]*[gG]' <<< "${JVECTOR_GPU_JVM:-}" | tail -1)
[ -n "$xmx" ] && heap_gb=${xmx//[^0-9]/}
if [ $((MEM_AVAIL_KB / 1024 / 1024)) -lt "$heap_gb" ]; then
    warn host-memory "$(gb $((MEM_AVAIL_KB * 1024))) available of $(gb $((MEM_TOTAL_KB * 1024))), below the ${heap_gb} GB heap; set JVECTOR_GPU_JVM=\"-Xmx<N>g\" lower"
else
    ok host-memory "$(gb $((MEM_AVAIL_KB * 1024))) available of $(gb $((MEM_TOTAL_KB * 1024))) (heap ${heap_gb} GB)"
fi
DISK_FREE_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')

# --- dataset: known by name, present or downloadable, and fits the GPU and the heap
if [ -z "$DATASET" ]; then
    ok disk "$(gb $((DISK_FREE_KB * 1024))) free in $ROOT"
    echo "  (pass a dataset name to check it too, e.g. jvector-gpu/preflight.sh ada002-100k)"
else
    CATALOG=$(ls "$ROOT"/jvector-examples/yaml-configs/dataset-catalogs/.catalog-cache/*.cache 2>/dev/null | head -1)
    base=$(awk -v d="$DATASET:" '$1 == d {f = 1; next} f && /^[^ ]/ {f = 0} f && $1 == "base:" {print $2; exit}' $CATALOG 2>/dev/null)
    cache="${DATASET_CACHE_DIR:-$ROOT/dataset_cache}"
    file=""
    [ -n "$base" ] && file=$(find -L "$cache" "$ROOT/fvec" -name "$base" -type f 2>/dev/null | head -1)
    if [ -n "$file" ]; then
        bytes=$(stat -L -c %s "$file")
        dim=$(od -An -t d4 -N4 "$file" | tr -d ' ')
        n=$((bytes / (4 + 4 * dim)))
        # the query and ground-truth files sit next to the base vectors
        absent=""
        for key in query gt; do
            f=$(awk -v d="$DATASET:" -v k="$key:" '$1 == d {f = 1; next} f && /^[^ ]/ {f = 0} f && $1 == k {print $2; exit}' $CATALOG)
            [ -n "$f" ] && [ ! -s "$(dirname "$file")/$f" ] && absent="$absent $f"
        done
        if [ -n "$absent" ]; then
            warn dataset "$DATASET: $n x $dim present, but not$absent; the first run fetches them (needs network)"
        else
            ok dataset "$DATASET: $n x $dim, $(gb "$bytes") in $(dirname "$file")"
        fi
        # the GPU keeps the vectors in FP16 (dimensions padded to 16) next to cuVS's working set
        need_mb=$(( n * ((dim + 15) / 16 * 16) * 2 / 1024 / 1024 ))
        if [ "$GPU_FREE_MB" -gt 0 ] && [ "$need_mb" -gt $((GPU_FREE_MB * 9 / 10)) ]; then
            fail gpu-fit "the vectors need ~$need_mb MiB on the GPU in FP16, more than 90% of the $GPU_FREE_MB MiB free"
        elif [ "$GPU_FREE_MB" -gt 0 ]; then
            ok gpu-fit "the vectors take ~$need_mb MiB on the GPU in FP16 of $GPU_FREE_MB MiB free"
        fi
        # the heap holds the float32 vectors, the graph and the PQ codes: ~3x the base file, an estimate
        need_heap_gb=$(( bytes * 3 / 1024 / 1024 / 1024 + 1 ))
        if [ "$need_heap_gb" -gt "$heap_gb" ]; then
            warn heap-fit "$DATASET needs a heap of ~${need_heap_gb} GB, over the ${heap_gb} GB set (JVECTOR_GPU_JVM=\"-Xmx${need_heap_gb}g\")"
        else
            ok heap-fit "$DATASET fits the ${heap_gb} GB heap (~${need_heap_gb} GB)"
        fi
        # graphs, compaction segments and PQ caches are written next to the repository
        need_disk_kb=$(( bytes * 4 / 1024 ))
        if [ "$DISK_FREE_KB" -lt "$need_disk_kb" ]; then
            warn disk "$(gb $((DISK_FREE_KB * 1024))) free in $ROOT; indexes and compaction segments may need ~$(gb $((need_disk_kb * 1024)))"
        else
            ok disk "$(gb $((DISK_FREE_KB * 1024))) free in $ROOT"
        fi
    elif [ -n "$base" ]; then
        warn dataset "$DATASET is in the catalog but not downloaded yet: the first run fetches it into $cache (needs network)"
        ok disk "$(gb $((DISK_FREE_KB * 1024))) free in $ROOT"
    elif grep -qx "  - $DATASET" "$ROOT/jvector-examples/yaml-configs/datasets.yml" 2>/dev/null \
         || grep -q "^$DATASET:" "$ROOT/jvector-examples/yaml-configs/dataset-metadata.yml" 2>/dev/null; then
        warn dataset "$DATASET is a known name, not downloaded yet; the first run fetches it (needs network)"
        ok disk "$(gb $((DISK_FREE_KB * 1024))) free in $ROOT"
    else
        fail dataset "unknown dataset '$DATASET' (see jvector-examples/yaml-configs/datasets.yml)"
    fi
fi

echo
if [ "$fails" -gt 0 ]; then
    echo "preflight: $fails check(s) failed, $warns warning(s)"
    exit 1
fi
[ "$warns" -gt 0 ] && echo "preflight: ready, with $warns warning(s)" || echo "preflight: ready"

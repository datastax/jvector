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
# Runs JVector with the GPU graph build, under the TornadoVM launcher.
#
#   jvector-gpu/run.sh bench <dataset regex> [-Dprop=value ...]   JVector's Bench, e.g. jvector-gpu/run.sh bench ada002-100k
#   jvector-gpu/run.sh cpu   <dataset regex> [-Dprop=value ...]   the same Bench without the accelerator (baseline)
#   jvector-gpu/run.sh check [base.fvecs query.fvecs gt.ivecs COSINE|DOT_PRODUCT|EUCLIDEAN] [-Dprop=value ...]
#                                                                  GPU build vs JVector's build (GpuBuildCheck)
#   jvector-gpu/run.sh pq      <dataset> [M] [K]                    product quantization, JVector vs GPU (PqBench)
#   jvector-gpu/run.sh compact <dataset> [segments] [workDir]       compaction, JVector vs GPU (CompactionBench)
#   jvector-gpu/run.sh wikipedia <dir> gpu|cpu                      graph build on Cohere Wikipedia 1M/10M x 1024, prepared
#                                                                  by jvector-gpu/datasets/wikipedia.sh (WikipediaBench)
#   jvector-gpu/run.sh groundtruth base.f32 query.f32 dim out.ivecs [k]   exact top-k by inner product on the GPU
#   jvector-gpu/run.sh nvq     <dataset>                            NVQ encoding, JVector vs GPU (NvqBench)
#   jvector-gpu/run.sh phases  <dataset> [workDir]                  where an on-disk build's time goes (WritePhases)
#
# jvector-gpu/preflight.sh [dataset] checks the JDK, TornadoVM, GPU, cuVS, build, memory, disk and dataset first.
#
# Needs: `tornado` on the PATH (TornadoVM SDK with the CUDA backend and tornado-cuvs), and the module built with
#   mvn -Pgpu -pl jvector-gpu -am package -DskipTests
# cuVS is set up by jvector-gpu/cuvs-env.sh, in JVECTOR_GPU_CUVS (default ~/.jvector-gpu/cuvs). JVM flags can be
# added with JVECTOR_GPU_JVM, e.g.
# JVECTOR_GPU_JVM="-Xmx56g -Dtornado.device.memory=22GB".
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-}"
[ $# -gt 0 ] && shift

case "$MODE" in
    bench) ACCELERATOR="-Djvector.graph.accelerator=tornadovm"; MAIN=io.github.jbellis.jvector.example.Bench ;;
    cpu)   MAIN=io.github.jbellis.jvector.example.Bench ;;
    check) MAIN=io.github.jbellis.jvector.gpu.GpuBuildCheck ;;
    pq) MAIN=io.github.jbellis.jvector.gpu.bench.PqBench ;;
    compact) MAIN=io.github.jbellis.jvector.gpu.bench.CompactionBench ;;
    wikipedia) MAIN=io.github.jbellis.jvector.gpu.bench.WikipediaBench ;;
    groundtruth) MAIN=io.github.jbellis.jvector.gpu.bench.GroundTruth ;;
    nvq) MAIN=io.github.jbellis.jvector.gpu.bench.NvqBench ;;
    phases) MAIN=io.github.jbellis.jvector.gpu.bench.WritePhases ;;
    *) sed -n '19,32p' "$0"; exit 1 ;;
esac

command -v tornado >/dev/null || { echo "run.sh: 'tornado' is not on the PATH (install the TornadoVM SDK and add its bin/)" >&2; exit 1; }
# pass the directory explicitly: a sourced script would otherwise see this script's arguments
source "$ROOT/jvector-gpu/cuvs-env.sh" "${JVECTOR_GPU_CUVS:-$HOME/.jvector-gpu/cuvs}"

GPU_JAR=$(ls "$ROOT"/jvector-gpu/target/jvector-gpu-*.jar 2>/dev/null | grep -v -e javadoc -e sources | head -1)
EXAMPLES_JAR=$(ls "$ROOT"/jvector-examples/target/jvector-examples-*-jar-with-dependencies.jar 2>/dev/null | head -1)
[ -n "$GPU_JAR" ] && [ -n "$EXAMPLES_JAR" ] || { echo "run.sh: build first: mvn -Pgpu -pl jvector-gpu,jvector-examples -am package -DskipTests" >&2; exit 1; }

# leading arguments that are not -D flags go to the program; -D flags go to the JVM
ARGS=()
PROPS=()
for a in "$@"; do
    if [[ "$a" == -D* ]]; then PROPS+=("$a"); else ARGS+=("$a"); fi
done
# TornadoVM's device memory budget: 90% of the GPU's memory (the pruner keeps the vectors there in FP16)
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 4096)
JVM="-Xmx48g -Dtornado.device.memory=$((GPU_MB * 9 / 10))MB --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED ${ACCELERATOR:-} ${JVECTOR_GPU_JVM:-} ${PROPS[*]:-}"

cd "$ROOT"
# the TornadoVM launcher passes program arguments with --params
if [ ${#ARGS[@]} -gt 0 ]; then
    exec tornado --jvm="$JVM" -cp "$GPU_JAR:$EXAMPLES_JAR" "$MAIN" --params "${ARGS[*]}"
fi
exec tornado --jvm="$JVM" -cp "$GPU_JAR:$EXAMPLES_JAR" "$MAIN"


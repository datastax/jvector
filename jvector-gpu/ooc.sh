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
# One out-of-core compaction under a host memory limit, as in the out-of-core results: OnDiskGraphIndexCompactor on the
# CPU, or with the GPU accelerator, in its own process inside a cgroup with no swap.
#
#   jvector-gpu/ooc.sh <memoryMax> <heap> <deviceMemory> cpu|gpu <segmentDir> <segments> <similarity> <out>
#
# e.g., for the ada002-1M row (segments from `run.sh walk ada002-1M 4 target/compaction cpu`):
#   jvector-gpu/ooc.sh 8G 4g 4GB cpu target/compaction/ada002-1M 4 COSINE target/ooc/ada002-1M-cpu.idx
#   jvector-gpu/ooc.sh 8G 4g 4GB gpu target/compaction/ada002-1M 4 COSINE target/ooc/ada002-1M-gpu.idx
#   jvector-gpu/run.sh ooc eval ada002-1M target/ooc/ada002-1M-cpu.idx target/ooc/ada002-1M-gpu.idx
#
# The limit covers heap, native memory and the page cache of the merge's files. It prints the time, the cgroup's
# peak memory and whether anything was OOM-killed. Needs systemd with cgroup v2 and user delegation of the memory
# controller (`systemd-run --user --scope -p MemoryMax=1G true` must work), and what run.sh needs.
set -euo pipefail

[ $# -eq 8 ] || { sed -n '17,30p' "$0"; exit 1; }
MEM=$1 HEAP=$2 DEVICE=$3 MODE=$4 SEGMENTS=$5 PARTS=$6 SIMILARITY=$7 OUT=$8
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$MODE" in cpu|gpu) ;; *) echo "ooc.sh: mode is cpu or gpu, not $MODE" >&2; exit 1 ;; esac
command -v systemd-run >/dev/null || { echo "ooc.sh: needs systemd-run (cgroup v2)" >&2; exit 1; }
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 -q true 2>/dev/null \
    || { echo "ooc.sh: 'systemd-run --user --scope -p MemoryMax=...' does not work here (no user cgroup delegation)" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

# the run should not inherit another run's dirty pages
sync
echo "== $MODE: MemoryMax=$MEM, heap $HEAP, device memory $DEVICE"
export JVECTOR_GPU_JVM="-Xmx$HEAP ${JVECTOR_GPU_JVM:-}"
systemd-run --user --scope -p MemoryMax="$MEM" -p MemorySwapMax=0 -q --setenv=JVECTOR_GPU_JVM="$JVECTOR_GPU_JVM" \
    bash -c 'CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
        rc=0
        "$0"/jvector-gpu/run.sh ooc compact "$1" "$2" "$3" "$4" "$5" -Dtornado.device.memory="$6" || rc=$?
        echo "exit $rc, peak memory $(( $(cat $CG/memory.peak) / 1048576 )) MB, $(grep oom_kill $CG/memory.events)"
        exit $rc' "$ROOT" "$SEGMENTS" "$PARTS" "$SIMILARITY" "$MODE" "$OUT" "$DEVICE"

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
# Prepares Cohere's English Wikipedia embeddings (wikipedia-2023-11-embed-multilingual-v3, 1024 dimensions, unit
# vectors) for jvector-gpu/run.sh wikipedia:
#   base.f32         the first 1M or 10M rows, raw little-endian float32
#   query.f32        10,000 held-out rows (from file 0100, which the base never uses)
#   gt_ip_100.ivecs  their exact top-100 neighbours by inner product (= cosine on unit vectors)
#
#   jvector-gpu/datasets/wikipedia.sh <dir> [1M|10M]      (default 1M)
#
# Downloads 100k rows per file from Hugging Face: 1M is 11 files, about 2.3 GB, then about 4 GB on disk; 10M is
# 101 files, about 21 GB, then about 63 GB. Needs curl and python3 with numpy and pyarrow
# (python3 -m pip install --user numpy pyarrow). Re-running resumes where it stopped.
set -euo pipefail

DIR="${1:?usage: wikipedia.sh <dir> [1M|10M]}"
SIZE="${2:-1M}"
case "$SIZE" in
    1M) FILES=10 ;;
    10M) FILES=100 ;;
    *) echo "wikipedia.sh: size must be 1M or 10M" >&2; exit 1 ;;
esac
HERE="$(cd "$(dirname "$0")" && pwd)"
URL="https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3/resolve/main/en"
python3 -c 'import numpy, pyarrow' 2>/dev/null \
    || { echo "wikipedia.sh: needs numpy and pyarrow: python3 -m pip install --user numpy pyarrow" >&2; exit 1; }

mkdir -p "$DIR/parquet"
for f in $(seq 0 "$FILES"); do   # files 0 .. FILES-1 for the base, file FILES for the queries
    name=$(printf '%04d.parquet' "$f")
    [ "$f" -eq "$FILES" ] && name=0100.parquet
    if [ ! -s "$DIR/parquet/$name" ]; then
        echo "downloading $name" >&2
        curl -sfL --retry 5 -o "$DIR/parquet/$name.part" "$URL/$name"
        mv "$DIR/parquet/$name.part" "$DIR/parquet/$name"
    fi
done

[ -s "$DIR/query.f32" ] || python3 "$HERE/wikipedia_convert.py" "$DIR" "$FILES"
if [ ! -s "$DIR/gt_ip_100.ivecs" ]; then
    # exact neighbours on the GPU when the module is built and TornadoVM is on the PATH (1M: 3 s), else numpy (1M: 1 min)
    if command -v tornado >/dev/null && ls "$HERE"/../target/jvector-gpu-*.jar >/dev/null 2>&1; then
        "$HERE/../run.sh" groundtruth "$DIR/base.f32" "$DIR/query.f32" 1024 "$DIR/gt_ip_100.ivecs" 100
    else
        python3 "$HERE/ground_truth.py" "$DIR/base.f32" "$DIR/query.f32" 1024 "$DIR/gt_ip_100.ivecs"
    fi
fi
echo "ready: $DIR ($SIZE). Run: jvector-gpu/run.sh wikipedia $DIR gpu" >&2

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
# Puts the NVIDIA cuVS C library (libcuvs_c) and its dependencies on LD_LIBRARY_PATH.
#
#   source jvector-gpu/cuvs-env.sh [dir]     (default dir: ~/.jvector-gpu/cuvs)
#
# The first time, it downloads NVIDIA's libcuvs-cu12 wheels with their dependencies and unpacks them into dir.
# The wheels are only used as archives of shared libraries, so any python3 with pip works (the wheels themselves
# require Python 3.11+ to be pip-installed). With conda, `conda install -c rapidsai -c conda-forge libcuvs` and
# $CONDA_PREFIX/lib on LD_LIBRARY_PATH do the same.

CUVS_VERSION="${CUVS_VERSION:-26.8.*}"
CUVS_DIR="${1:-$HOME/.jvector-gpu/cuvs}"

if [ ! -s "$CUVS_DIR/ld_library_path" ]; then
    echo "Installing cuVS $CUVS_VERSION into $CUVS_DIR" >&2
    mkdir -p "$CUVS_DIR/wheels"
    python3 -m pip download --quiet --timeout 120 --retries 10 --only-binary=:all: --python-version 3.12 \
        --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
        --extra-index-url https://pypi.nvidia.com -d "$CUVS_DIR/wheels" "libcuvs-cu12==$CUVS_VERSION" \
        || { echo "cuvs-env.sh: downloading the cuVS wheels failed" >&2; return 1 2>/dev/null || exit 1; }
    for wheel in "$CUVS_DIR"/wheels/*.whl; do
        python3 -m zipfile -e "$wheel" "$CUVS_DIR"
    done
    rm -rf "$CUVS_DIR/wheels"
    find "$CUVS_DIR" -name '*.so*' -type f -printf '%h\n' | sort -u | paste -sd: > "$CUVS_DIR/ld_library_path"
fi

export LD_LIBRARY_PATH="$(cat "$CUVS_DIR/ld_library_path")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

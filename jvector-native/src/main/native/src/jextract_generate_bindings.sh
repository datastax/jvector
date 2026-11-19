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

# Generate Java bindings for the native library.
# Only needs to be run when the native header changes.

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MODULE_ROOT="${REPO_ROOT}/jvector-native"
SCRIPT_DIR="${REPO_ROOT}/jvector-native/src/main/native/src"
JAVA_OUT_DIR="${MODULE_ROOT}/src/main/java"

# The user is allowed to specify a custom jextract path via env var.
# This is useful if there are multiple jextract versions on the system.
if [[ -z "${JEXTRACT_BIN:-}" ]]; then
  if command -v jextract &> /dev/null
  then
    JEXTRACT_BIN=jextract
  else
    echo 1>&2 "ERROR: jextract could not be found, please install it if you need to update bindings."
    exit 1
  fi
fi

jextract_version="$("$JEXTRACT_BIN" --version 2>&1 | head -1 | cut -d' ' -f2)"
want_jextract_version=22
if [[ "$jextract_version" != "$want_jextract_version" ]] ;then
    echo 1>&2 "WARNING: got jextract version [${jextract_version}], expected [${want_jextract_version}]."
    echo 1>&2 "WARNING: Generated bindings may cause compile errors."
fi

"$JEXTRACT_BIN" \
  --output "${JAVA_OUT_DIR}" \
  -t io.github.jbellis.jvector.vector.cnative \
  -I "${SCRIPT_DIR}" \
  --header-class-name NativeSimdOps \
  "${SCRIPT_DIR}/jvector_simd.h"

# Set critical linker option with heap-based segments for all generated methods
sed -i 's/DESC)/DESC, Linker.Option.critical(true))/g' \
  "${JAVA_OUT_DIR}/io/github/jbellis/jvector/vector/cnative/NativeSimdOps.java"

# Optional oneDNN ASH query-projection add-on (built with -Dnative.onednn=true).
# These calls do long-running oneDNN work, so they are left as ordinary
# (non-critical) downcalls; do not apply the critical-option edit above.
ONEDNN_DIR="${MODULE_ROOT}/src/main/onednn/src"
"$JEXTRACT_BIN" \
  --output "${JAVA_OUT_DIR}" \
  -t io.github.jbellis.jvector.vector.cnative \
  -I "${ONEDNN_DIR}" \
  --header-class-name NativeAshOneDnn \
  --include-function jvector_ash_fp16_abi \
  --include-function jvector_ash_fp16_create \
  --include-function jvector_ash_fp16_destroy \
  --include-function jvector_ash_fp16_bytes \
  --include-function jvector_ash_fp16_project \
  --include-function jvector_ash_fp16_project_batch \
  --include-function jvector_ash_fp32_project_batch \
  --include-function jvector_ash_int8_create \
  --include-function jvector_ash_int8_destroy \
  --include-function jvector_ash_int8_bytes \
  --include-function jvector_ash_int8_project_batch \
  --include-struct jvector_ash_fp16_result \
  --include-typedef jvector_ash_fp16_result \
  --include-constant JV_ASH_FP16_OK \
  --include-constant JV_ASH_FP16_UNSUPPORTED \
  --include-constant JV_ASH_FP16_INVALID_ARGUMENT \
  --include-constant JV_ASH_FP16_RANGE \
  --include-constant JV_ASH_FP16_NO_MEMORY \
  --include-constant JV_ASH_FP16_ERROR \
  "${ONEDNN_DIR}/jvector_ash_fp16.h"

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

# Generate the Java FFI bindings (ffi/) for fastervector's C API from fvec_capi.h.
# Only needs to be run when the header changes. The output is committed.
#
# Environment:
#   JEXTRACT_BIN  jextract to use (default: jextract on PATH). Must be jextract 22.
#   FVEC_CAPI_H   header to generate from (default: the staged native/linux-x86_64/fvec_capi.h)

set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
JAVA_OUT_DIR="${MODULE_ROOT}/src/main/java"
FFI_PACKAGE=io.github.jbellis.jvector.fastervector.ffi
FFI_DIR="${JAVA_OUT_DIR}/${FFI_PACKAGE//.//}"
COMMITTED_HEADER="${MODULE_ROOT}/src/main/native/fvec_capi.h"
FVEC_CAPI_H="${FVEC_CAPI_H:-${MODULE_ROOT}/native/linux-x86_64/fvec_capi.h}"

if [[ -z "${JEXTRACT_BIN:-}" ]]; then
  if command -v jextract &> /dev/null; then
    JEXTRACT_BIN=jextract
  else
    echo 1>&2 "ERROR: jextract not found. Install jextract 22 and put it on PATH or set JEXTRACT_BIN."
    exit 1
  fi
fi

# jextract newer than 22 emits SymbolLookup.findOrThrow, which doesn't compile at --release 22
jextract_version="$("$JEXTRACT_BIN" --version 2>&1 | head -1 | cut -d' ' -f2)"
want_jextract_version=22
if [[ "$jextract_version" != "$want_jextract_version" ]]; then
  echo 1>&2 "ERROR: got jextract version [${jextract_version}], expected [${want_jextract_version}]."
  exit 1
fi

if [[ ! -f "$FVEC_CAPI_H" ]]; then
  echo 1>&2 "ERROR: header not found: ${FVEC_CAPI_H}"
  echo 1>&2 "Stage it with scripts/stage-fastervector-native.sh, or set FVEC_CAPI_H."
  exit 1
fi

# Start clean so symbols removed from the header don't linger
rm -rf "$FFI_DIR"

"$JEXTRACT_BIN" \
  --output "$JAVA_OUT_DIR" \
  --target-package "$FFI_PACKAGE" \
  --header-class-name FvecCapi \
  "$FVEC_CAPI_H"

LICENSE_HEADER='/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
'
for f in "$FFI_DIR"/*.java; do
  { printf '%s\n' "$LICENSE_HEADER"; cat "$f"; } > "$f.tmp"
  mv "$f.tmp" "$f"
done

# Keep the header these bindings came from, so header changes show up as a diff
if ! cmp -s "$FVEC_CAPI_H" "$COMMITTED_HEADER"; then
  cp "$FVEC_CAPI_H" "$COMMITTED_HEADER"
fi

echo "Generated $(ls "$FFI_DIR" | wc -l | tr -d ' ') files in ${FFI_DIR} from ${FVEC_CAPI_H}"

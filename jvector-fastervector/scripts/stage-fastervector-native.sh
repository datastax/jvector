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

# Build fastervector's C API library and stage it, with libjvector.so.0 and fvec_capi.h,
# into jvector-fastervector/native/linux-x86_64/ (git-ignored). Only reads from the
# fastervector checkout. Linux x86_64 only.
#
# Usage: stage-fastervector-native.sh <fastervector-checkout> [--no-build]
#   --no-build   stage what the checkout already built; skip cargo
#
# The checkout should be a pinned worktree with submodules initialized:
#   git worktree add --detach ../FasterVector-pinned origin/java-bindings
#   git -C ../FasterVector-pinned submodule update --init --recursive

set -eo pipefail

usage() {
  echo 1>&2 "Usage: $0 <fastervector-checkout> [--no-build]"
  exit 2
}

FV=""
BUILD=1
for arg in "$@"; do
  case "$arg" in
    --no-build) BUILD=0 ;;
    -h|--help) usage ;;
    -*) echo 1>&2 "ERROR: unknown option $arg"; usage ;;
    *) [[ -z "$FV" ]] || usage; FV="$arg" ;;
  esac
done
[[ -n "$FV" ]] || usage
FV="$(cd "$FV" && pwd)"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 1>&2 "ERROR: fastervector's native libraries build only on Linux x86_64 (this is $(uname -s) $(uname -m))."
  exit 1
fi

MODULE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${MODULE_ROOT}/native/linux-x86_64"
COMMITTED_HEADER="${MODULE_ROOT}/src/main/native/fvec_capi.h"
SIMD_BUILD_DIR="${JVECTOR_SIMD_BUILD_DIR:-${FV}/crates/fvec-highway-sys/cpp/build}"
RUNTIME_LIB="${FV}/target/release/libfvec_runtime.so"
JVECTOR_LIB="${SIMD_BUILD_DIR}/libjvector.so.0"

if [[ ! -f "${FV}/fvec_capi.h" || ! -d "${FV}/crates/fvec-runtime" ]]; then
  echo 1>&2 "ERROR: ${FV} doesn't look like a fastervector checkout (no fvec_capi.h or crates/fvec-runtime)."
  exit 1
fi

if [[ "$BUILD" == 1 ]]; then
  if [[ -z "$(ls -A "${FV}/crates/fvec-highway-sys/cpp/third_party/highway" 2>/dev/null)" ]]; then
    echo 1>&2 "ERROR: the Highway submodule is missing. Run: git -C ${FV} submodule update --init --recursive"
    exit 1
  fi
  # Make MKL visible to fvec-mkl-sys. setvars.sh isn't safe under set -u.
  if [[ -z "${MKLROOT:-}" ]]; then
    for setvars in /opt/intel/oneapi/setvars.sh ~/intel/oneapi/setvars.sh; do
      if [[ -f "$setvars" ]]; then
        # shellcheck disable=SC1090
        source "$setvars" >/dev/null 2>&1 || true
        break
      fi
    done
  fi
  echo "Building fvec-runtime in ${FV}"
  (cd "$FV" && cargo build --release -p fvec-runtime)
fi

set -u

for lib in "$RUNTIME_LIB" "$JVECTOR_LIB"; do
  if [[ ! -e "$lib" ]]; then
    if [[ "$BUILD" == 0 ]]; then
      echo 1>&2 "ERROR: ${lib} not found. Build the checkout first, or run without --no-build."
    else
      echo 1>&2 "ERROR: ${lib} not found after the build."
    fi
    exit 1
  fi
done

# Check the outputs (Part 1, step 4 of the integration plan)
soname="$(readelf -d "$JVECTOR_LIB" | sed -n 's/.*Library soname: \[\(.*\)\]/\1/p')"
if [[ "$soname" != libjvector.so.0 ]]; then
  echo 1>&2 "ERROR: ${JVECTOR_LIB} has SONAME [${soname}], expected [libjvector.so.0]."
  exit 1
fi
if ! readelf -d "$RUNTIME_LIB" | grep -q 'NEEDED.*libmkl_core'; then
  echo 1>&2 "WARNING: libfvec_runtime.so was built without MKL (no libmkl_core dependency)."
  echo 1>&2 "WARNING: it falls back to slower GEMMs; don't use this build for measurements."
fi
if ldd "$RUNTIME_LIB" | grep -q 'not found'; then
  echo 1>&2 "WARNING: some dependencies of libfvec_runtime.so are not found on this host:"
  ldd "$RUNTIME_LIB" | grep 'not found' 1>&2
fi
nfuncs="$(nm -D --defined-only "$RUNTIME_LIB" | grep -c ' fvec_' || true)"

mkdir -p "$DEST"
cp -L "$RUNTIME_LIB" "${DEST}/libfvec_runtime.so"
cp -L "$JVECTOR_LIB" "${DEST}/libjvector.so.0"
cp "${FV}/fvec_capi.h" "${DEST}/fvec_capi.h"

commit="$(git -C "$FV" rev-parse HEAD)"
# A pinned worktree is usually detached; fall back to the remote branch whose tip is HEAD
branch="$(git -C "$FV" symbolic-ref --short -q HEAD \
  || git -C "$FV" branch -r --points-at HEAD | grep -v ' -> ' | head -1 | sed 's|^ *[^/]*/||' \
  || true)"
dirty=""
if [[ -n "$(git -C "$FV" status --porcelain --untracked-files=no)" ]]; then
  dirty=" (with uncommitted changes)"
  echo 1>&2 "WARNING: ${FV} has uncommitted changes; the staged build doesn't match ${commit}."
fi
cat > "${DEST}/BUILD-INFO" <<EOF
fastervector.branch=${branch:-unknown}
fastervector.commit=${commit}${dirty}
cargo.features=default
build.host=$(hostname)
build.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "Staged into ${DEST}: libfvec_runtime.so (${nfuncs} fvec_* functions), libjvector.so.0, fvec_capi.h, BUILD-INFO"
cat "${DEST}/BUILD-INFO"

if [[ ! -f "$COMMITTED_HEADER" ]] || ! cmp -s "${DEST}/fvec_capi.h" "$COMMITTED_HEADER"; then
  echo 1>&2 "WARNING: the staged fvec_capi.h differs from src/main/native/fvec_capi.h, the header the"
  echo 1>&2 "WARNING: committed ffi/ classes were generated from. Regenerate them:"
  echo 1>&2 "WARNING:   ./mvnw -Pfastervector,fastervector.jextract -pl jvector-fastervector -am compile"
fi

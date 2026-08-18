#!/bin/bash

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

# Sanity-checks the libjvector.so that actually ships inside the built
# jvector-native jar (not the loose copy in src/main/resources): correct
# architecture, resolvable dynamic dependencies, and presence of the expected
# exported symbols. Bound to the `verify` phase so it runs as part of
# `mvn verify`. Non-fatal (warns and skips) for any check whose tool isn't
# available on the current OS (e.g. readelf/ldd on macOS).

set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "$0")" && pwd)"
JAR="${1:-}"

if [ -z "${JAR}" ]; then
  JAR=$(find "${MODULE_ROOT}/target" -maxdepth 1 -name 'jvector-native-*.jar' \
          ! -name '*-sources.jar' ! -name '*-javadoc.jar' | head -1)
fi

if [ -z "${JAR}" ] || [ ! -f "${JAR}" ]; then
  echo "ERROR: no jvector-native-*.jar found under ${MODULE_ROOT}/target. Run 'mvn package' first." >&2
  exit 1
fi

LIBNAME="libjvector.so"
WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}"' EXIT

if ! unzip -p "${JAR}" "${LIBNAME}" > "${WORKDIR}/${LIBNAME}" 2>/dev/null || [ ! -s "${WORKDIR}/${LIBNAME}" ]; then
  echo "ERROR: ${LIBNAME} not found (or empty) inside ${JAR}." >&2
  echo "       This is expected if ${JAR} was not built on a unix/amd64 host: the native" >&2
  echo "       library is only compiled and packaged there (see unix-amd64-profile in" >&2
  echo "       jvector-native/pom.xml). On e.g. Apple Silicon (arm64) Macs, the native" >&2
  echo "       build is skipped entirely, so the jar never contains ${LIBNAME}." >&2
  exit 1
fi

LIB="${WORKDIR}/${LIBNAME}"
FAIL=0

printf '\n== %s (from %s) ==\n' "${LIBNAME}" "$(basename "${JAR}")"

echo "-- file --"
file "${LIB}"

echo "-- architecture (readelf -h) --"
if command -v readelf &>/dev/null; then
  MACHINE_LINE=$(readelf -h "${LIB}" | grep -i 'Machine:')
  echo "${MACHINE_LINE}"
  if [[ "${MACHINE_LINE}" != *"X86-64"* ]]; then
    echo "ERROR: expected an x86-64 shared object, got: ${MACHINE_LINE}" >&2
    FAIL=1
  fi
else
  echo "WARNING: readelf not available on this OS, skipping architecture check (rely on 'file' output above)." >&2
fi

echo "-- dynamic dependencies (ldd) --"
if command -v ldd &>/dev/null; then
  LDD_OUT=$(ldd "${LIB}" 2>&1 || true)
  echo "${LDD_OUT}"
  if echo "${LDD_OUT}" | grep -qi "not found"; then
    echo "ERROR: unresolved shared library dependencies detected above." >&2
    FAIL=1
  fi
else
  echo "WARNING: ldd not available on this OS (e.g. macOS); use 'otool -L ${LIBNAME}' manually if needed." >&2
fi

echo "-- exported symbols (nm) --"
REQUIRED_SYMBOLS=(
  jvector_simd_get_active_isa
  jvector_simd_get_max_isa_env
  dot_product_f32
  cosine_f32
  euclidean_f32
)
if command -v nm &>/dev/null; then
  SYMBOLS=$(nm -D "${LIB}" 2>/dev/null || nm "${LIB}" 2>/dev/null || true)
  for sym in "${REQUIRED_SYMBOLS[@]}"; do
    # Mach-O (macOS) nm output underscore-prefixes C symbols; ELF (Linux) does not.
    if echo "${SYMBOLS}" | grep -qE "[[:space:]]_?${sym}\$"; then
      echo "  OK      ${sym}"
    else
      echo "  MISSING ${sym}" >&2
      FAIL=1
    fi
  done
else
  echo "WARNING: nm not available, skipping exported-symbol check." >&2
fi

echo
if [ "${FAIL}" -ne 0 ]; then
  echo "libjvector.so verification FAILED" >&2
  exit 1
fi

echo "libjvector.so verification passed"

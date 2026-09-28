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

# Sanity-checks both libjvector-x86_64.so and libjvector-aarch64.so that ship
# inside the built jvector-native jar (not the loose copies in
# src/main/resources): correct architecture, resolvable dynamic dependencies,
# and presence of the expected exported symbols. Both libraries MUST be present
# and pass all checks — this script is used in the release process where the
# jar is built with -Dnative.crossarch. Non-fatal (warns and skips) for any
# check whose tool isn't available on the current OS (e.g. readelf/ldd on
# macOS).
#
# Usage: verify_native_lib.sh [<jar>]
# Defaults: jar auto-detected from target/.

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

REQUIRED_SYMBOLS=(
  jvector_simd_get_active_isa
  jvector_simd_get_max_isa_env
  dot_product_f32
  cosine_f32
  euclidean_f32
)

WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}"' EXIT

OVERALL_FAIL=0
VERIFIED=0

for ARCH in x86_64 aarch64; do
  LIBNAME="libjvector-${ARCH}.so"

  printf '\n== %s (from %s) ==\n' "${LIBNAME}" "$(basename "${JAR}")"

  # Both libraries are required; missing either one is a hard failure.
  if ! unzip -p "${JAR}" "${LIBNAME}" > "${WORKDIR}/${LIBNAME}" 2>/dev/null \
      || [ ! -s "${WORKDIR}/${LIBNAME}" ]; then
    rm -f "${WORKDIR}/${LIBNAME}"
    echo "ERROR: ${LIBNAME} not found (or empty) inside ${JAR}." >&2
    echo "       Release jars must be built with -Dnative.crossarch to bundle both libraries." >&2
    OVERALL_FAIL=1
    continue
  fi

  LIB="${WORKDIR}/${LIBNAME}"
  FAIL=0

  echo "-- file --"
  file "${LIB}"

  echo "-- architecture (readelf -h) --"
  if command -v readelf &>/dev/null; then
    MACHINE_LINE=$(readelf -h "${LIB}" | grep -i 'Machine:')
    echo "${MACHINE_LINE}"
    if [ "${ARCH}" = "aarch64" ]; then
      EXPECTED_MACHINE="AArch64"
    else
      EXPECTED_MACHINE="X86-64"
    fi
    if [[ "${MACHINE_LINE}" != *"${EXPECTED_MACHINE}"* ]]; then
      echo "ERROR: expected a ${EXPECTED_MACHINE} shared object, got: ${MACHINE_LINE}" >&2
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

  # Note that an exhaustive check of exported symbols is not strictly necessary, because the
  # Java code will load but fail at runtime if any of the expected symbols are missing. But
  # this check is a useful sanity-check to catch any accidental changes to the native code that
  # would break the Java code, and to catch any accidental changes to the build process that
  # would result in a library that doesn't export the expected symbols.
  echo "-- exported symbols (nm) --"
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
    echo "${LIBNAME} verification FAILED" >&2
    OVERALL_FAIL=1
  else
    echo "${LIBNAME} verification passed"
    VERIFIED=$((VERIFIED + 1))
  fi
done

EXPECTED=2
echo
if [ "${OVERALL_FAIL}" -ne 0 ] || [ "${VERIFIED}" -lt "${EXPECTED}" ]; then
  echo "Release verification FAILED: expected both libjvector-x86_64.so and libjvector-aarch64.so to be present and pass all checks." >&2
  exit 1
fi

echo "All verified libraries passed"

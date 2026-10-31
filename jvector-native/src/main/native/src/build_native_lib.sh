#!/bin/bash

# fail on error
set -e
# print commands as they are executed
set -x

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

# ---------------------------------------------------------------------------
# Path anchors — all derived from the git repository root so the script works
# regardless of the working directory it is invoked from (Maven sets
# workingDirectory to the src directory, but developers may run it from
# anywhere inside the repo).
# ---------------------------------------------------------------------------
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
NATIVE_DIR="${REPO_ROOT}/jvector-native/src/main/native"
MODULE_ROOT="${REPO_ROOT}/jvector-native"

HIGHWAY_DIR="${NATIVE_DIR}/third_party/highway"
# Drop the .so files into target/meson-build/ so that mvn clean removes them
# along with every other build artefact, and Maven's resource plugin picks them
# up from there instead of from src/main/resources/.
RESOURCES_DIR="${MODULE_ROOT}/target/meson-build"

if [ "$1" == "--auto-install-deps" ] ; then AUTO_INSTALL_DEPS=true ; shift ; fi
printf "AUTO_INSTALL_DEPS=%s\n" "${AUTO_INSTALL_DEPS}"

# Accept buildtype parameter (default: release)
BUILDTYPE="${1:-release}"
if [ "$BUILDTYPE" != "release" ] && [ "$BUILDTYPE" != "debug" ] && [ "$BUILDTYPE" != "debugoptimized" ]; then
  echo "WARNING: Invalid buildtype '$BUILDTYPE'. Using 'release' instead."
  echo "         Valid values: release, debug, debugoptimized"
  BUILDTYPE="release"
fi
printf "BUILDTYPE=%s\n" "${BUILDTYPE}"

# Accept crossarch flag (default: false).
# When true, builds both the native arch library AND cross-compiles for the other arch.
#   On x86_64: builds libjvector-x86_64.so  + cross-compiles libjvector-aarch64.so
#   On aarch64: builds libjvector-aarch64.so + cross-compiles libjvector-x86_64.so
CROSSARCH="${2:-false}"
printf "CROSSARCH=%s\n" "${CROSSARCH}"

# Optional FP16 query library; it never changes libjvector's dependencies.
ONEDNN="${3:-false}"
case "${ONEDNN}" in
  true) ONEDNN_FEATURE=enabled ;;
  false) ONEDNN_FEATURE=disabled ;;
  *) echo "ERROR: oneDNN flag must be true or false."; exit 1 ;;
esac

mkdir -p "${RESOURCES_DIR}"
# target/meson-build/ is under target/, which IS cleaned by mvn clean.
# src/main/resources/ is intentionally NOT written to any more.

# Check that the Google Highway submodule has been initialised
if [ ! -f "${HIGHWAY_DIR}/hwy/highway.h" ]; then
  echo "ERROR: Google Highway submodule not found at ${HIGHWAY_DIR}."
  echo "       Run the following command from the repository root to fix this:"
  echo ""
  echo "         git submodule update --init"
  echo ""
  exit 1
fi

# Desired minimum GCC version
MIN_GCC_VERSION=11

# Ensures $1 (a command) is available. If not and AUTO_INSTALL_DEPS=true, runs
# the Ubuntu apt/pip install given in $2; otherwise prints $2 as a hint and exits.
# Usage: require_cmd <cmd> <ubuntu-install-cmd>
require_cmd() {
  local cmd="$1" ubuntu_install="$2"
  if command -v "${cmd}" &> /dev/null; then return; fi
  if [ "${AUTO_INSTALL_DEPS}" == "true" ]; then
    LSB_RELEASE=$(lsb_release --id --short)
    printf "LSB_RELEASE=%s\n" "${LSB_RELEASE}"
    if [ "${LSB_RELEASE}" == "Ubuntu" ]; then
      eval "sudo apt-get update && ${ubuntu_install}"
    else
      printf "distribution %s needs a '%s' install command in %s\n" "${LSB_RELEASE}" "${cmd}" "${0}" ; exit 2
    fi
  else
    printf "'%s' is not installed. To install it, run: %s\n" "${cmd}" "${ubuntu_install}" ; exit 2
  fi
}

require_cmd meson  "sudo apt-get install -y meson"
require_cmd ninja  "sudo apt-get install -y ninja-build"

# ---------------------------------------------------------------------------
# Detect the host CPU architecture.
# ---------------------------------------------------------------------------
HOST_ARCH="$(uname -m)"   # e.g. x86_64 or aarch64
printf "HOST_ARCH=%s\n" "${HOST_ARCH}"

# ---------------------------------------------------------------------------
# build_native <arch>
#   Compiles the library for <arch> using either a native build (when arch ==
#   HOST_ARCH) or a cross-compile via the bundled cross file.
#   Output: RESOURCES_DIR/libjvector-<arch>.so
# ---------------------------------------------------------------------------
build_native() {
  local ARCH="$1"
  local BUILD_DIR="${MODULE_ROOT}/target/meson-build-${ARCH}"
  local OUT_SO="${RESOURCES_DIR}/libjvector-${ARCH}.so"

  rm -rf "${OUT_SO}"

  # Older Meson versions only accept --wipe for an already configured build.
  local setup_options=()
  if [[ -f "${BUILD_DIR}/meson-private/coredata.dat" ]]; then
    setup_options+=(--wipe)
  fi

  if [ "${ARCH}" == "${HOST_ARCH}" ]; then
    # ---- Native build ------------------------------------------------------
    require_cmd g++ "sudo apt-get install -y g++"

    CURRENT_GPP_VERSION=$(g++ -dumpversion)
    if [ "$(printf '%s\n' "$MIN_GCC_VERSION" "$CURRENT_GPP_VERSION" | sort -V | head -n1)" != "$MIN_GCC_VERSION" ]; then
      echo "WARNING: g++ version $CURRENT_GPP_VERSION is too old. Please upgrade to g++ $MIN_GCC_VERSION or newer."
      exit 1
    fi

    meson setup "${BUILD_DIR}" "${NATIVE_DIR}" \
        "${setup_options[@]}" \
        --buildtype="${BUILDTYPE}" -Donednn="${ONEDNN_FEATURE}"
  else
    # ---- Cross-compile build -----------------------------------------------
    # Only x86_64 <-> aarch64 is supported.
    if [ "${ARCH}" == "aarch64" ]; then
      CROSS_TOOLCHAIN="aarch64-linux-gnu-g++"
      CROSS_INSTALL="sudo apt-get install -y g++-aarch64-linux-gnu"
      CROSS_FILE="${NATIVE_DIR}/aarch64-cross.ini"
    elif [ "${ARCH}" == "x86_64" ]; then
      CROSS_TOOLCHAIN="x86_64-linux-gnu-g++"
      CROSS_INSTALL="sudo apt-get install -y g++-x86-64-linux-gnu"
      CROSS_FILE="${NATIVE_DIR}/x86_64-cross.ini"
    else
      echo "ERROR: No cross-compile support for arch '${ARCH}'." ; exit 1
    fi

    require_cmd "${CROSS_TOOLCHAIN}" "${CROSS_INSTALL}"

    meson setup "${BUILD_DIR}" "${NATIVE_DIR}" \
        "${setup_options[@]}" \
        --cross-file "${CROSS_FILE}" \
        --buildtype="${BUILDTYPE}" -Donednn="${ONEDNN_FEATURE}"
  fi

  meson compile -C "${BUILD_DIR}"

  # The versioned .so (e.g. libjvector.so.0.1.0) is the real file; symlinks point to it.
  SOFILE=$(find "${BUILD_DIR}" -maxdepth 1 -name 'libjvector.so.*' -type f | head -1)
  if [ -z "${SOFILE}" ]; then
    echo "ERROR: libjvector.so not found in ${BUILD_DIR} after ${ARCH} build."
    exit 1
  fi
  cp "${SOFILE}" "${OUT_SO}"
  printf "Built: %s\n" "${OUT_SO}"

  local FP16_OUT="${RESOURCES_DIR}/libjvector_ash_fp16-${ARCH}.so"
  if [ "${ONEDNN}" == "true" ]; then
    cp "${BUILD_DIR}/libjvector_ash_fp16.so" "${FP16_OUT}"
    # License texts travel with the optional static dependency in either jar.
    cp "${NATIVE_DIR}/subprojects/onednn/LICENSE" "${RESOURCES_DIR}/oneDNN-LICENSE.txt"
    cp "${NATIVE_DIR}/subprojects/onednn/THIRD-PARTY-PROGRAMS" "${RESOURCES_DIR}/oneDNN-THIRD-PARTY.txt"
  else
    # A later build without the add-on must not package a stale enabled build.
    rm -f "${FP16_OUT}"
  fi
}

# ---------------------------------------------------------------------------
# Determine which archs to build.
# ---------------------------------------------------------------------------
if [ "${HOST_ARCH}" == "x86_64" ]; then
  OTHER_ARCH="aarch64"
elif [ "${HOST_ARCH}" == "aarch64" ]; then
  OTHER_ARCH="x86_64"
else
  echo "ERROR: Unsupported host architecture '${HOST_ARCH}'. Supported: x86_64, aarch64."
  exit 1
fi

if [ "${ONEDNN}" == "false" ]; then
  # Maven copies resources without removing old files from target/classes.
  # Remove only this optional component from both resource locations.
  for resource_dir in "${RESOURCES_DIR}" "${MODULE_ROOT}/target/classes"; do
    rm -f "${resource_dir}"/libjvector_ash_fp16-*.so \
          "${resource_dir}/oneDNN-LICENSE.txt" "${resource_dir}/oneDNN-THIRD-PARTY.txt"
  done
fi

# Always build the native arch.
build_native "${HOST_ARCH}"

# Optionally cross-compile the other arch.
if [ "${CROSSARCH}" == "true" ]; then
  build_native "${OTHER_ARCH}"
fi

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

# ---------------------------------------------------------------------------
# Detect the host operating system and CPU architecture.
#   HOST_OS   : linux | windows           (windows = Git Bash / MSYS2 / Cygwin)
#   HOST_ARCH : x86_64 | aarch64
# Supported platforms: linux x86_64, linux aarch64, windows x86_64.
# ---------------------------------------------------------------------------
case "$(uname -s)" in
  Linux*)               HOST_OS="linux" ;;
  MINGW*|MSYS*|CYGWIN*) HOST_OS="windows" ;;
  *) echo "ERROR: Unsupported operating system '$(uname -s)'. Supported: Linux, Windows." ; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64|amd64)  HOST_ARCH="x86_64" ;;
  aarch64|arm64) HOST_ARCH="aarch64" ;;
  *) echo "ERROR: Unsupported host architecture '$(uname -m)'. Supported: x86_64, aarch64." ; exit 1 ;;
esac
printf "HOST_OS=%s\nHOST_ARCH=%s\n" "${HOST_OS}" "${HOST_ARCH}"

case "${HOST_OS}-${HOST_ARCH}" in
  linux-x86_64|linux-aarch64|windows-x86_64) ;;
  *) echo "ERROR: ${HOST_OS} on ${HOST_ARCH} is not supported. Supported: linux x86_64/aarch64, windows x86_64." ; exit 1 ;;
esac

# Name under which the library is bundled in the jar. This must match
# LibraryLoader.resourceNameFor() in the Java sources, and the includes in the pom.
# Usage: bundled_name <arch>
bundled_name() {
  case "${HOST_OS}" in
    linux)   echo "libjvector-$1.so" ;;
    windows) echo "jvector-windows-$1.dll" ;;
  esac
}

# Desired minimum GCC version: the x86_64 build targets Sapphire Rapids, and Highway's
# HWY_AVX3_SPR target needs GCC 12 or newer (third_party/highway/hwy/detect_targets.h)
MIN_GCC_VERSION=11
if [ "${HOST_ARCH}" == "x86_64" ]; then
  MIN_GCC_VERSION=12
fi

# Where to get missing tools on platforms other than Ubuntu.
case "${HOST_OS}" in
  windows) OTHER_OS_HINT="pip install meson ninja (and install LLVM/clang, e.g. 'choco install llvm')" ;;
  *)       OTHER_OS_HINT="" ;;
esac

# Ensures $1 (a command) is available. If not and AUTO_INSTALL_DEPS=true, runs
# the Ubuntu apt/pip install given in $2; otherwise prints $2 as a hint and exits.
# On Windows nothing is installed automatically; the hint above is printed.
# Usage: require_cmd <cmd> <ubuntu-install-cmd>
require_cmd() {
  local cmd="$1" ubuntu_install="$2"
  if command -v "${cmd}" &> /dev/null; then return; fi
  if [ "${HOST_OS}" != "linux" ]; then
    printf "'%s' is not installed. To install it, run: %s\n" "${cmd}" "${OTHER_OS_HINT}" ; exit 2
  fi
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
# build_native <arch>
#   Compiles the library for <arch> using either a native build (when arch ==
#   HOST_ARCH) or a cross-compile via the bundled cross file (Linux only).
#   Output: RESOURCES_DIR/<bundled_name arch>
#           (libjvector-<arch>.so or jvector-windows-<arch>.dll)
# ---------------------------------------------------------------------------
build_native() {
  local ARCH="$1"
  local BUILD_DIR="${MODULE_ROOT}/target/meson-build-${ARCH}"
  local OUT_LIB="${RESOURCES_DIR}/$(bundled_name "${ARCH}")"

  rm -rf "${OUT_LIB}"

  if [ "${ARCH}" == "${HOST_ARCH}" ]; then
    # ---- Native build ------------------------------------------------------
    case "${HOST_OS}" in
      linux)
        # meson compiles with $CXX when it is set, so check that compiler
        CXX_CMD="${CXX:-g++}"
        require_cmd "${CXX_CMD}" "sudo apt-get install -y g++"

        CURRENT_GPP_VERSION=$(${CXX_CMD} -dumpfullversion -dumpversion)
        if [ "$(printf '%s\n' "$MIN_GCC_VERSION" "$CURRENT_GPP_VERSION" | sort -V | head -n1)" != "$MIN_GCC_VERSION" ]; then
          echo "ERROR: ${CXX_CMD} version $CURRENT_GPP_VERSION is too old. Please upgrade to g++ $MIN_GCC_VERSION or newer, or set CXX to one."
          exit 1
        fi
        ;;
      windows)
        # Use clang (not MSVC): the ISA variants rely on GCC-style -march flags and
        # Highway's AVX-512 targets, which MSVC's cl.exe does not support. clang targets
        # the MSVC ABI and links with the MSVC linker or lld, so a Visual Studio build
        # environment (vcvars / "x64 Native Tools" prompt) must be active.
        export CXX="${CXX:-clang++}"
        export CC="${CC:-clang}"
        export CXX_LD="${CXX_LD:-lld}"
        require_cmd "${CXX}" ""
        if [ -z "${LIB:-}" ]; then
          echo "WARNING: the LIB environment variable is unset; the Visual Studio build environment"
          echo "         does not appear to be active, so linking will likely fail. Run from an"
          echo "         'x64 Native Tools Command Prompt' or use ilammy/msvc-dev-cmd in CI."
        fi
        ;;
    esac

    meson setup "${BUILD_DIR}" "${NATIVE_DIR}" \
        --wipe \
        --buildtype="${BUILDTYPE}"
  else
    # ---- Cross-compile build -----------------------------------------------
    # Only x86_64 <-> aarch64 on Linux is supported.
    if [ "${HOST_OS}" != "linux" ]; then
      echo "ERROR: cross-compilation is only supported on Linux." ; exit 1
    fi
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
        --wipe \
        --cross-file "${CROSS_FILE}" \
        --buildtype="${BUILDTYPE}"
  fi

  meson compile -C "${BUILD_DIR}"

  # Locate the real (non-symlink) shared library produced by meson:
  #   Linux:   the versioned file, e.g. libjvector.so.0.1.0 (symlinks point to it)
  #   Windows: jvector.dll (meson ignores 'version' on Windows)
  case "${HOST_OS}" in
    linux)   LIB_PATTERN='libjvector.so.*' ;;
    windows) LIB_PATTERN='jvector.dll' ;;
  esac
  SOFILE=$(find "${BUILD_DIR}" -maxdepth 1 -name "${LIB_PATTERN}" -type f | head -1)
  if [ -z "${SOFILE}" ]; then
    echo "ERROR: no file matching '${LIB_PATTERN}' found in ${BUILD_DIR} after ${ARCH} build."
    exit 1
  fi
  cp "${SOFILE}" "${OUT_LIB}"
  printf "Built: %s\n" "${OUT_LIB}"
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

# Always build the native arch.
build_native "${HOST_ARCH}"

# Optionally cross-compile the other arch.
if [ "${CROSSARCH}" == "true" ]; then
  if [ "${HOST_OS}" == "linux" ]; then
    build_native "${OTHER_ARCH}"
  else
    echo "WARNING: crossarch is only supported on Linux; ignoring it on ${HOST_OS}."
  fi
fi

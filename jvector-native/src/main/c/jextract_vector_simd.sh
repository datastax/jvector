#!/bin/bash

# fail on error
set -e

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

if [ "$1" == "--auto-install-gcc" ] || [ "$1" == "--auto-install-g++" ] ; then AUTO_INSTALL_GCC=true ; shift ; fi
printf "AUTO_INSTALL_GCC=%s\n" "${AUTO_INSTALL_GCC}"

mkdir -p ../resources
# compile jvector_simd_check.cpp as x86-64
# compile jvector_simd.cpp as skylake-avx512
# produce one shared library

# Check that the Google Highway submodule has been initialised
HIGHWAY_DIR="third_party/highway"
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

if ! command -v g++ &> /dev/null; then
  if [ "$AUTO_INSTALL_GCC" == "true" ]
  then
    LSB_RELEASE=$(lsb_release --id --short)
    printf "LSB_RELEASE=%s\n" "${LSB_RELEASE}"
    if [ "${LSB_RELEASE}" == "Ubuntu" ]
    then sudo apt update && sudo apt install -y g++
    else printf "distribution %s needs a g++ install command in %s\n" "${LSB_RELEASE}" "${0}" ; exit 2
    fi
  else
    echo "g++ is not installed. Please install g++ 11+ to build supporting native libraries."
    exit 2
  fi
fi

# Check g++ version
CURRENT_GPP_VERSION=$(g++ -dumpversion)

if [ "$(printf '%s\n' "$MIN_GCC_VERSION" "$CURRENT_GPP_VERSION" | sort -V | head -n1)" != "$MIN_GCC_VERSION" ]; then
    echo "WARNING: g++ version $CURRENT_GPP_VERSION is too old. Please upgrade to g++ $MIN_GCC_VERSION or newer."
    exit 1
fi

# Check if the current GCC version is greater than or equal to the minimum required version
if [ "$(printf '%s\n' "$MIN_GCC_VERSION" "$CURRENT_GPP_VERSION" | sort -V | head -n1)" != "$MIN_GCC_VERSION" ]; then
    echo "WARNING: g++ version $CURRENT_GPP_VERSION is too old. Please upgrade to g++ $MIN_GCC_VERSION or newer."
    exit 1
fi

# Check meson and ninja are available
if ! command -v meson &> /dev/null; then
    echo "meson is not installed. Please install it: pip install meson  or  sudo apt install meson"
    exit 2
fi
if ! command -v ninja &> /dev/null; then
    echo "ninja is not installed. Please install it: sudo apt install ninja-build"
    exit 2
fi

BUILD_DIR="build"
rm -rf ../resources/libjvector.so

# Configure (--wipe resets any stale configuration) then compile
meson setup "${BUILD_DIR}" \
    --wipe \
    --buildtype=release

meson compile -C "${BUILD_DIR}"

# The versioned .so (e.g. libjvector.so.0.1.0) is the real file; symlinks point to it.
# Copy it to ../resources/ as the plain libjvector.so for Java System.load().
SOFILE=$(find "${BUILD_DIR}" -maxdepth 1 -name 'libjvector.so.*' -type f | head -1)
if [ -z "${SOFILE}" ]; then
    echo "ERROR: libjvector.so not found in ${BUILD_DIR} after build."
    exit 1
fi
cp "${SOFILE}" ../resources/libjvector.so

# Generate Java source code
# Should only be run when c header changes
# Check if jextract is available before running
if ! command -v jextract &> /dev/null
then
    echo "WARNING: jextract could not be found, please install it if you need to update bindings."
    exit 0
fi

jextract \
  --output ../java \
  -t io.github.jbellis.jvector.vector.cnative \
  -I . \
  --header-class-name NativeSimdOps \
  jvector_simd.h

# Set critical linker option with heap-based segments for all generated methods
sed -i 's/DESC)/DESC, Linker.Option.critical(true))/g' ../java/io/github/jbellis/jvector/vector/cnative/NativeSimdOps.java

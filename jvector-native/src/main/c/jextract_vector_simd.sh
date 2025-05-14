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

if [ "$1" == "--auto-install-gcc" ] ; then AUTO_INSTALL_GCC=true ; shift ; fi
printf "AUTO_INSTALL_GCC=%s\n" "${AUTO_INSTALL_GCC}"

mkdir -p ../resources
# compile jvector_simd_check.c as x86-64
# compile jvector_simd.c as skylake-avx512
# produce one shared library

# Desired minimum GCC version
MIN_GCC_VERSION=11

if ! command -v gcc &> /dev/null; then
  if [ "$AUTO_INSTALL_GCC" == "true" ]
  then
    LSB_RELEASE=$(lsb_release --id --short)
    printf "LSB_RELEASE=%s\n" "${LSB_RELEASE}"
    if [ "${LSB_RELEASE}" == "Ubuntu" ]
    then sudo apt update && sudo apt install -y gcc
    else printf "distribution %s needs a gcc install command in %s\n" "${LSB_RELEASE}" "${0}" ; exit 2
    fi
  else
    echo "GCC is not installed. Please install GCC 11+ to build supporting native libraries."
    exit 2
  fi
fi

# Check if GCC is installed
CURRENT_GCC_VERSION=$(gcc -dumpversion)

# Check if the current GCC version is greater than or equal to the minimum required version
if [ "$(printf '%s\n' "$MIN_GCC_VERSION" "$CURRENT_GCC_VERSION" | sort -V | head -n1)" = "$MIN_GCC_VERSION" ]; then
    rm -rf ../resources/libjvector.so
    gcc -fPIC -O3 -march=icelake-server -c jvector_simd.c -o jvector_simd.o
    gcc -fPIC -O3 -march=x86-64 -c jvector_simd_check.c -o jvector_simd_check.o
    gcc -shared -o ../resources/libjvector.so jvector_simd_check.o jvector_simd.o
else
    echo "WARNING: GCC version $CURRENT_GCC_VERSION is too old. Please upgrade to GCC $MIN_GCC_VERSION or newer."
fi

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

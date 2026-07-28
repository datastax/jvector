#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#
# build-jvector.sh — Build the jvector multi-release jar with the JDK the node runs.
#
# The Cassandra node runs JDK 25, so jvector is built with it too. Note that the
# class-file versions in the jar come from the poms' --release targets, NOT from
# the build JDK: base is 11 (v55), jvector-twenty is 20 (v64), jvector-native is
# 22 (v66). Building under 25 yields a byte-identical jar — the JDK pin is here
# so the toolchain matches the rest of the stack and cannot drift to the 17 that
# happens to be on PATH.
#
# On a 25 runtime the JVM loads META-INF/versions/22, so vector math runs through
# NativeVectorizationProvider. That selection ALSO requires
# `--add-modules jdk.incubator.vector` on the *runtime* command line: without it
# jvector silently falls back to DefaultVectorizationProvider (scalar), even with
# jvector.experimental.enable_native_vectorization=true. Cassandra passes it in
# conf/jvm25-server.options; --verify below checks the jar can still reach the
# native provider.
#
# Do NOT pass -Dmaven.javadoc.skip=true: jvector-multirelease *copies* the
# javadoc artifact, so skipping it fails the build with
#   Could not find artifact io.github.jbellis:jvector-parent:jar:javadoc
# `mvn clean` is likewise avoided by default — it forces the javadoc fork to
# redo work that only fails in interesting ways.
#
# --deploy INSTALLS to the local maven repository. Copying the jar into
# cassandra/lib is useless on its own: cassandra's ant build declares jvector as a
# resolved dependency (<dependency groupId="io.github.jbellis" artifactId="jvector"/>)
# and re-copies it from ~/.m2 into lib/ on every build, silently overwriting
# anything placed there by hand. A hand-copied jar therefore survives exactly until
# the next cassandra build -- which is the build that would have used it.
#
# Usage:
#   ./bin/build-jvector.sh [--clean] [--verify] [--deploy]
#
# Examples:
#   ./bin/build-jvector.sh                   # package only
#   ./bin/build-jvector.sh --verify          # ...and confirm the SIMD path loads
#   ./bin/build-jvector.sh --verify --deploy # ...and install into ~/.m2
#
# Env overrides:
#   JV_JDK        JDK to build with          (default: /opt/jdk25)
#   CASS_UNIT     systemd unit to check      (default: cassandra.service)
set -euo pipefail

JV_JDK="${JV_JDK:-/opt/jdk25}"
JV_CASS_LIB="${JV_CASS_LIB:-/mnt/nvme/opt/cassandra/lib}"
CASS_UNIT="${CASS_UNIT:-cassandra.service}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

do_clean=0; do_verify=0; do_deploy=0
for arg in "$@"; do
    case "$arg" in
        --clean)  do_clean=1 ;;
        --verify) do_verify=1 ;;
        --deploy) do_deploy=1 ;;
        -h|--help) sed -n '2,39p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
    esac
done

[[ -x "$JV_JDK/bin/javac" ]] || { echo "error: no JDK at $JV_JDK (set JV_JDK)" >&2; exit 1; }
jdk_major="$("$JV_JDK/bin/java" -version 2>&1 | sed -nE 's/^openjdk version "([0-9]+).*/\1/p')"
[[ "$jdk_major" == "25" ]] || {
    echo "error: $JV_JDK reports major version '$jdk_major', expected 25." >&2
    exit 1
}
export JAVA_HOME="$JV_JDK"
export PATH="$JAVA_HOME/bin:$PATH"

cd "$REPO"
echo "==> JDK    $("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"

if [[ "$do_clean" -eq 1 ]]; then
    echo "==> mvn clean (javadoc will be rebuilt; slower and historically fragile)"
    mvn clean -q
fi

# `install` rather than `package` when deploying: the cassandra build resolves
# jvector out of the local repository, so that is the only copy that matters.
goal="package"
[[ "$do_deploy" -eq 1 ]] && goal="install"
echo "==> mvn $goal -DskipTests"
mvn "$goal" -DskipTests

# Newest matching jar, chosen without a pipeline: under `set -o pipefail` a
# reader that exits early (head, grep -q) SIGPIPEs the writer and fails the
# whole pipeline even when the match succeeded.
jar=""
for f in "$REPO"/jvector-multirelease/target/jvector-*-SNAPSHOT.jar; do
    [[ -f "$f" ]] || continue
    case "$f" in *-sources.jar|*-javadoc.jar) continue ;; esac
    [[ -z "$jar" || "$f" -nt "$jar" ]] && jar="$f"
done
[[ -n "$jar" ]] || { echo "error: no multirelease jar produced" >&2; exit 1; }

# The multi-release trees are what make this jar useful on 25; a jar missing
# them would run scalar everywhere and look merely "slow". Capture the listing
# once, then match against it in memory.
listing="$(unzip -l "$jar")"
for v in 20 22; do
    grep -q "META-INF/versions/$v/" <<<"$listing" \
        || { echo "error: $jar has no META-INF/versions/$v tree" >&2; exit 1; }
done
echo "==> built $jar ($(stat -c%s "$jar") bytes, MR trees: 20 22)"

if [[ "$do_verify" -eq 1 ]]; then
    echo "==> verifying the runtime picks the native SIMD provider"
    probe="$(mktemp -t jvprobe.XXXXXX.jsh)"
    trap 'rm -f "$probe"' EXIT
    echo 'System.out.println("provider = " + io.github.jbellis.jvector.vector.VectorizationProvider.getInstance().getClass().getName());' > "$probe"
    # Same flags conf/jvm25-server.options gives the daemon.
    got="$("$JAVA_HOME/bin/jshell" --class-path "$jar" \
            -R--add-modules=jdk.incubator.vector \
            -R--enable-native-access=ALL-UNNAMED \
            -R-Djvector.experimental.enable_native_vectorization=true \
            -s "$probe" 2>&1 | sed -nE 's/^provider = (.*)$/\1/p')"
    echo "    $got"
    case "$got" in
        *NativeVectorizationProvider|*PanamaVectorizationProvider) ;;
        *) echo "error: expected a SIMD provider, got '${got:-nothing}' — vector math would run scalar" >&2
           exit 1 ;;
    esac
fi

if [[ "$do_deploy" -eq 1 ]]; then
    if systemctl is-active --quiet "$CASS_UNIT"; then
        echo "error: $CASS_UNIT is running; swapping its jvector jar mid-flight causes" >&2
        echo "       progressive NoClassDefFoundError. Stop it first." >&2
        exit 1
    fi
    version="$(basename "$jar")"; version="${version#jvector-}"; version="${version%.jar}"
    installed="$HOME/.m2/repository/io/github/jbellis/jvector/$version/$(basename "$jar")"
    if [[ ! -f "$installed" ]]; then
        echo "error: expected the install to produce $installed" >&2
        exit 1
    fi
    if ! cmp -s "$jar" "$installed"; then
        echo "error: installed artifact differs from the jar just built:" >&2
        echo "         built:     $jar" >&2
        echo "         installed: $installed" >&2
        echo "       The cassandra build resolves the installed copy, so it would use" >&2
        echo "       something other than what was built here." >&2
        exit 1
    fi
    echo "==> installed $installed (identical to the built jar)"
    echo "    Now rebuild cassandra so it resolves this artifact into lib/ and compiles"
    echo "    against it: /mnt/nvme/opt/cassandra/bin/build-cassandra.sh --clean"
fi

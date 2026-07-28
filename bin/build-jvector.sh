#!/usr/bin/env bash
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
# Usage:
#   ./bin/build-jvector.sh [--clean] [--verify] [--deploy]
#
# Examples:
#   ./bin/build-jvector.sh                   # package only
#   ./bin/build-jvector.sh --verify          # ...and confirm the SIMD path loads
#   ./bin/build-jvector.sh --verify --deploy # ...and install into cassandra/lib
#
# Env overrides:
#   JV_JDK        JDK to build with          (default: /opt/jdk25)
#   JV_CASS_LIB   deploy target for --deploy (default: /mnt/nvme/opt/cassandra/lib)
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

echo "==> mvn package -DskipTests"
mvn package -DskipTests

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
    target="$JV_CASS_LIB/$(basename "$jar")"
    [[ -f "$target" ]] && cp -p "$target" "$target.bak"
    cp -p "$jar" "$target"
    echo "==> deployed $target (previous kept as $(basename "$target").bak)"
    echo "    Cassandra must be rebuilt against this jar if its API surface moved."
fi

# Building and installing JVector

This guide covers two ways to get JVector: using a release from Maven Central, and building it
from source for a particular platform and vector backend. ASH requires release 4.1.0-rc.1 or later.

- [Vector backends](#vector-backends)
- [Using a release from Maven Central](#using-a-release-from-maven-central)
- [Building from source](#building-from-source)
- [Checking which backend is in use](#checking-which-backend-is-in-use)

## Vector backends

JVector picks one of three vectorization providers when it starts:

| Provider | Java runtime | Platforms | Notes |
| --- | --- | --- | --- |
| Native | 22 and newer | Linux x86-64 (SSE4.2 or newer), Linux AArch64 | Fastest. Kernels built with [Google Highway](https://github.com/google/highway); the best instruction-set tier is chosen at startup. Required for INT8 and FP16 ASH query projection. |
| Panama | 20 and newer | Any | Uses the incubating Java Vector API. Used on macOS, Windows and other platforms, and on Linux when native code is unavailable. |
| Scalar | 11 and newer | Any | Plain Java. Used on Java 11 to 19. |

Native is preferred on Java 22 and newer when the native library loads; otherwise JVector falls
back to Panama, then scalar. [SIMD and native acceleration](../README.md#simd-and-native-acceleration)
lists the properties that override this choice.

The optional **oneDNN add-on** is a second native library that ASH uses for FP16 query projection,
for INT8 on processors with AVX-VNNI but not AVX-512, and for batched FP32 encoding. Without it,
ASH falls back to the Highway or Java kernels for those operations. See
[Choosing ASH query-projection precision](./ash-projection-precision.md).

## Using a release from Maven Central

JVector is published as one multi-release jar, `io.github.jbellis:jvector`.

Maven:

```xml
<dependency>
  <groupId>io.github.jbellis</groupId>
  <artifactId>jvector</artifactId>
  <version>4.1.0-rc.1</version>
</dependency>
```

Gradle (Kotlin DSL):

```kotlin
implementation("io.github.jbellis:jvector:4.1.0-rc.1")
```

### What the jar contains

- The core library, compiled for Java 11.
- The Panama provider, used on Java 20 and newer (`META-INF/versions/20`).
- The native provider, used on Java 22 and newer (`META-INF/versions/22`), with native libraries
  for Linux x86-64 and Linux AArch64, and the oneDNN add-on for both.

There is nothing else to install: the native libraries load from the jar, and oneDNN is built in
rather than linked against a system `libdnnl`. ASH training uses the system BLAS and LAPACK (MKL or
OpenBLAS) when installed and a slower Java implementation otherwise; see
[BLAS threads](./ash-configuration.md#blas-threads).

### JVM options

| Java runtime | Options | Provider |
| --- | --- | --- |
| 11 to 19 | none | Scalar |
| 20 and 21 | `--add-modules jdk.incubator.vector` | Panama |
| 22 and newer | `--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED` | Native on Linux, else Panama |

Without `--add-modules jdk.incubator.vector`, Java 20 and newer use the scalar provider. Do not
pass it on Java 11 to 15, which do not have that module. On Java 22 and newer,
`--enable-native-access` avoids the JVM's native-access warning; if native access is denied, JVector
falls back to Panama. If your application runs as named modules, grant native access to the module
that contains JVector instead of `ALL-UNNAMED`.

If you repackage JVector into a single "uber" jar, keep the `Multi-Release: true` manifest entry and
the `META-INF/versions` directories; otherwise the Panama and native providers are lost or, on older
runtimes, can fail to load.

## Building from source

### Common requirements

- JDK 22 or newer to run the build, whatever runtime you target.
- Maven 3.8 or newer, and Git.

Clone the repository and build from its root:

```bash
git clone https://github.com/datastax/jvector.git
cd jvector
mvn -DskipTests install
```

`install` places the jar in your local Maven repository as version `4.1.0-SNAPSHOT` (or whatever
version the checkout declares), ready to use as a dependency. Use `package` instead to build
without installing; the jar is `jvector-multirelease/target/jvector-<version>.jar`.

### Builds by target

| Target | Additional requirements | Command |
| --- | --- | --- |
| Linux x86-64 or AArch64, native | GCC 12 or newer (11 or newer on AArch64), [Meson](https://mesonbuild.com/), [Ninja](https://ninja-build.org/) | `mvn -DskipTests install` |
| Linux, native with oneDNN (recommended for ASH) | As above, plus CMake 3.13 or newer; adds a few minutes to the first build | `mvn -DskipTests install -Dnative.onednn=true` |
| Linux, both architectures | As above, plus a cross-compiler (`g++-aarch64-linux-gnu` on x86-64) | `mvn -DskipTests install -Dnative.crossarch` |
| Linux, both architectures with oneDNN (what a release contains) | All of the above | `mvn -DskipTests install -Dnative.crossarch -Dnative.onednn=true` |
| macOS and other platforms | None | `mvn -DskipTests install` (no native library is built; JVector uses Panama or scalar) |

On Linux the native library is part of every build, so Meson and Ninja are required there. On
Ubuntu, `bash build_native_lib.sh --auto-install-deps` (in `jvector-native/src/main/native/src`)
installs the toolchain and builds the library. Debug builds, manual Meson builds, compiler notes and
ISA tiers are covered in the [native library README](../jvector-native/src/main/native/README.md).

A jar built on one Linux architecture contains only that architecture's native library unless you
add `-Dnative.crossarch`. A jar built on macOS contains no native library and runs with Panama or
scalar everywhere.

### Building and running the examples

```bash
mvn -pl jvector-examples -am -DskipTests package -Dnative.onednn=true
```

On macOS, drop `-Dnative.onednn=true`. The [Getting started](../README.md#getting-started) section
and the [examples README](../jvector-examples/README.md) show how to run each program.

### Running the tests

[Building and testing](../README.md#building-and-testing) lists the test commands, including the
scalar tests on a Java 11 runtime.

## Checking which backend is in use

JVector logs its choice at startup, for example:

```text
INFO: Native Vector API enabled. Using NativeVectorizationProvider.
INFO: Highway ISA tier: avx3_spr, Env variable JVECTOR_MAX_ISA: (not set)
```

On Java 20 and 21, or wherever native code is unavailable, it logs `Using
PanamaVectorizationProvider`. From code, `VectorizationProvider.getInstance()` returns the provider
in use.

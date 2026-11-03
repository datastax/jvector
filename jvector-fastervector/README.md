<!--
Copyright DataStax, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# JVector fastervector

Java bindings for fastervector's IVF-ASH index: an SSD-resident inverted-file index with
ASH-compressed vectors, built and searched by fastervector's native code through the Java
Foreign Function & Memory (FFM) API.

It is a separate index type next to jvector's graph index. You use it through the
`io.github.jbellis.jvector.fastervector` API, not through `GraphIndexBuilder` or
`OnDiskGraphIndex`.

> **Status: experimental.** Linux x86_64 with AVX2 only. The native libraries are not
> published yet; you build them from a fastervector checkout and stage them here. The
> module is opt-in (`-Pfastervector`) and is not built by default or in CI.

---

## Requirements

- JDK 22 or later (the module compiles at `--release 22`).
- Linux x86_64 with AVX2. Elsewhere, loading the library throws
  `UnsupportedOperationException`.
- Intel oneAPI MKL and its OpenMP runtime `libiomp5` at run time, usually by sourcing
  oneAPI's `setvars.sh`.
- To build the native libraries: a fastervector checkout, a Rust toolchain, `meson`,
  `ninja`, and a C++ compiler.

## Layout

```
FASTERVECTOR_SOURCE                    fastervector branch and commit the code and header came from
native/linux-x86_64/                   staged native libraries (git-ignored)
scripts/stage-fastervector-native.sh   builds fastervector's C API library and stages it here
src/main/native/fvec_capi.h            the C header ffi/ was generated from
src/main/native/jextract_generate_bindings.sh
src/main/java/.../fastervector/        the Java API
src/main/java/.../fastervector/ffi/    jextract-generated bindings (don't edit)
src/test/java/.../fastervector/        tests, and example/IvfAshExample
```

## Build and test

Everything runs from the jvector repository root. Every Maven command on this module needs
`-Pfastervector` (so Maven knows about the module) and `-am` (so it builds `jvector-base`
first).

```bash
# 1. Build fastervector's native libraries and stage them into native/linux-x86_64/.
#    Use a pinned worktree with submodules initialized; --no-build stages an existing build.
jvector-fastervector/scripts/stage-fastervector-native.sh /path/to/FasterVector-pinned

# 2. Build, and run all tests. The native tests skip when the libraries can't load;
#    requireNative makes that a failure instead.
./mvnw -Pfastervector -pl jvector-fastervector -am verify \
    -Djvector.fastervector.test.requireNative=true

# 3. Run the example
./mvnw -Pfastervector -pl jvector-fastervector -am test-compile exec:exec@example
```

A jar built after staging bundles the two libraries, and the loader extracts them at run
time. A jar built without them needs `-Djvector.fastervector.native.dir` (below).

## Running

Every JVM that uses this module needs:

| Setting | Purpose |
|---|---|
| `--enable-native-access=ALL-UNNAMED` | Permits the FFM calls. Use the module name instead on the module path. |
| `-Djvector.enable_native_vectorization=false` | Required for now. fastervector bundles its own `libjvector.so.0`, which has the same SONAME and symbols as jvector-native's library. This switch stops jvector-native from loading it; jvector then uses its Panama vector code for its own math. If jvector-native's library is already loaded, `FastervectorLibrary` refuses to load and names this switch. |
| `-Djvector.fastervector.native.dir=<dir>` | Only if the libraries aren't in the jar: a directory holding `libjvector.so.0` and `libfvec_runtime.so`. |
| `LD_LIBRARY_PATH` with MKL and `libiomp5` | Resolves `libfvec_runtime.so`'s MKL dependencies. Not needed on the machine that built the libraries. |

The Maven test and example configurations already pass the two JVM flags.

## Usage

```java
try (FvecThreadPool pool = FvecThreadPool.create(8);
     IvfAshIndex index = IvfAshIndex.builder()
             .nlist(256)                          // IVF coarse centroids
             .vectors(new Array2DVecs(base))      // float[n][dim]
             .ssdPaths(List.of(Path.of("/mnt/ssd0")))
             .subfolder("my-index")
             .buildThreadPool(pool)
             .build();
     IvfAshIndexSearcher searcher = index.searcherBuilder()
             .topk(10)
             .nprobe(16)                          // centroids probed per query
             .searchThreadPool(pool)
             .build()) {
    SearchResult res = searcher.search(new FlatArrayVecs(queries, dim));
    for (NodeScore ns : res.results().get(0)) {
        long ordinal = ns.label();               // 0-based row in `base`
        float score = ns.score();                // inner product, larger is better
    }
}
```

- **Vectors.** Any `Vecs`: `Array2DVecs`, `FlatArrayVecs`, `ListVecs`, `SegmentVecs`, or
  `RandomAccessVectorValuesVecs`, which wraps a jvector `RandomAccessVectorValues` without
  copying it. Result labels are then jvector node ordinals.
- **Build.** `build()` streams the vectors in chunks of up to 2,000,000: it trains the
  centroids on the first chunk, assigns every vector, then encodes them to the SSD paths.
  `storageFactor` (default `1.0`) sets the compression: `1.0` is 32×, `2.0` is 16×.
- **On disk.** For each SSD path `i`, the index writes `<ssdPaths[i]>/<subfolder>/dev{i}.ash.blk`;
  the manifest is `<ssdPaths[0]>/<subfolder>/dev0.ash.blk.manifest`.
- **Reopen.** Pass that layout back:
  ```java
  IvfAshIndex.prepareOpen()
          .manifestPath(dir.resolve("dev0.ash.blk.manifest").toString())
          .devPaths(List.of(dir.resolve("dev0.ash.blk").toString()))
          .dim(dim)                                // optional; enables the query-dim check
          .open();
  ```
- **Results.** Similarity is inner product. Each query gets up to `topk` results, best
  first; empty slots are dropped, so a row can be shorter than `topk`.
- **Thread pools.** Without a pool, a build uses all CPUs, while a searcher uses one thread.
- **Lifetime.** Close searchers and indexes (both are `AutoCloseable`). Closing an index
  closes its searchers; using either after `close()` throws `IllegalStateException`.

## Regenerating the FFI bindings

`ffi/` is generated by jextract 22 from `src/main/native/fvec_capi.h` and committed; normal
builds never run jextract. Regenerate when the header changes, for example when the staging
script warns that the staged header differs:

```bash
export JEXTRACT_BIN=/path/to/jextract-22/bin/jextract   # must be jextract 22
./mvnw -Pfastervector,fastervector.jextract -pl jvector-fastervector -am compile
```

The script regenerates `ffi/`, adds the license header, and updates
`src/main/native/fvec_capi.h`. Set `FVEC_CAPI_H` to generate from a header that isn't
staged. Commit `ffi/` and the header together, and update `FASTERVECTOR_SOURCE`. Don't use a
jextract newer than 22: its output calls `SymbolLookup.findOrThrow`, which doesn't exist
before JDK 23.

## Updating from fastervector

1. Pin the new fastervector commit in a worktree and run `scripts/stage-fastervector-native.sh`.
2. If it warns that the header changed, regenerate the bindings (above).
3. Copy over any changed Java code from fastervector, keeping this module's package names
   and its edits to `FastervectorLibrary` and `IvfAshIndex`.
4. Update `FASTERVECTOR_SOURCE`, then run the build and tests with `requireNative=true`.

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

## Environment variables

The commands below use these, set to the paths on your machine:

```bash
export JVECTOR_HOME=/path/to/jvector                       # this jvector checkout
export FASTERVECTOR_HOME=/path/to/FasterVector             # fastervector checkout (only read)
export FASTERVECTOR_PINNED="$FASTERVECTOR_HOME-pinned"     # pinned worktree of it, with submodules initialized
export ONEAPI_SETVARS=/opt/intel/oneapi/setvars.sh         # ~/intel/oneapi/setvars.sh for a user install
export JEXTRACT_BIN=/path/to/jextract-22/bin/jextract      # only to regenerate the FFI bindings
```

Create the pinned worktree with
`git -C "$FASTERVECTOR_HOME" worktree add --detach "$FASTERVECTOR_PINNED" origin/java-bindings`, then
`git -C "$FASTERVECTOR_PINNED" submodule update --init --recursive`.

## Build and test

Everything runs from the jvector repository root, `$JVECTOR_HOME`. This module and
`jvector-fastervector-bench` are in the parent's opt-in `fastervector` profile, not its default
modules, so Maven only builds them with `-Pfastervector`:

| Command (from `$JVECTOR_HOME`) | Builds the fastervector modules? |
|---|---|
| `./mvnw clean package` | No. They aren't in the parent's default `<modules>`. |
| `./mvnw -Pfastervector clean package` | Yes, both, including the bench module's `jar-with-dependencies`. The normal way to build them. |
| `./mvnw -Pfastervector -pl <module> -am package` | Yes, that module and what it depends on. A shortcut for working on one module; `-am` is required because the parent skips `install`, so `jvector-base` only resolves from the same build. |

`package` runs this module's tests; add `-DskipTests` to skip them. `clean` doesn't remove the
staged libraries in `native/`.

```bash
cd "$JVECTOR_HOME"
source "$ONEAPI_SETVARS" >/dev/null 2>&1 || true

# 1. Build fastervector's native libraries and stage them into native/linux-x86_64/.
#    --no-build stages an existing build.
jvector-fastervector/scripts/stage-fastervector-native.sh "$FASTERVECTOR_PINNED"

# 2. Build, and run all tests. The native tests skip when the libraries can't load;
#    requireNative makes that a failure instead.
./mvnw -Pfastervector -pl jvector-fastervector -am verify \
    -Djvector.fastervector.test.requireNative=true

# 3. Run the example
./mvnw -Pfastervector -pl jvector-fastervector -am test-compile exec:exec@example
```

To run the example with `java` directly, write out the classpath first. `IvfAshExample` is test
code, so it is in `target/test-classes`, not the module jar:

```bash
cd "$JVECTOR_HOME"
./mvnw -q -Pfastervector -pl jvector-fastervector -am test-compile dependency:build-classpath \
    -Dmdep.outputFile=target/classpath.txt -Dmdep.includeScope=test

java --enable-native-access=ALL-UNNAMED \
     -Djvector.enable_native_vectorization=false \
     -Djvector.fastervector.native.dir="$JVECTOR_HOME/jvector-fastervector/native/linux-x86_64" \
     -cp "jvector-fastervector/target/test-classes:jvector-fastervector/target/classes:$(cat jvector-fastervector/target/classpath.txt)" \
     io.github.jbellis.jvector.fastervector.example.IvfAshExample
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
- **Thread pools.** Without a pool, a build uses all CPUs and a searcher uses 1 thread, as
  `fvec_capi.h` documents. (The pinned C API treats a `NULL` searcher pool as all CPUs, so
  `IvfAshIndex` passes a 1-thread pool instead.) A searcher reads only the pool's thread count,
  so one pool can serve many searchers and be closed once they're built.
- **Concurrency.** Search a batch of queries per call. At the pinned fastervector commit, each
  search call starts its thread count of new I/O threads on io_uring rings `0..n-1`, which every
  searcher shares behind a lock, and scores on a process-wide scan pool of that size. Concurrent
  single-query calls therefore barely scale, whatever their thread counts. To answer single
  queries from many threads, use `IvfAshBatchingSearcher` (below).
- **Lifetime.** Close searchers and indexes (both are `AutoCloseable`). Closing an index
  closes its searchers; using either after `close()` throws `IllegalStateException`.

### Single queries from many threads

`IvfAshBatchingSearcher` combines concurrent single queries into batched searches on one
searcher. Callers submit one query each; a dispatcher thread searches whatever has queued, up to
`maxBatch` queries, in one call, so queries that arrive during a search form the next batch.

```java
try (IvfAshIndexSearcher searcher = index.searcherBuilder()
             .topk(10).nprobe(16).searchThreadPool(pool)   // a pool sized for the machine
             .build();
     IvfAshBatchingSearcher batcher = IvfAshBatchingSearcher.builder(searcher)
             .maxBatch(1024)                               // the default
             .maxDelay(Duration.ZERO)                      // the default: don't wait for more
             .build()) {
    List<NodeScore> top = batcher.search(query);           // from any thread; or searchAsync(query)
}
```

- It doesn't own the searcher: close the batcher first, as above. Closing it finishes the batch
  in progress and fails queued queries with `IllegalStateException`.
- `maxDelay` makes a batch wait for more queries before it searches, adding up to that much
  latency per query. Under load, batches fill without waiting.
- Futures complete on the dispatcher thread, so keep stages chained onto them without an
  executor short.
- `batchCount()` and `queryCount()` give the mean batch size.

## Regenerating the FFI bindings

`ffi/` is generated by jextract 22 from `src/main/native/fvec_capi.h` and committed; normal
builds never run jextract. Regenerate when the header changes, for example when the staging
script warns that the staged header differs:

```bash
cd "$JVECTOR_HOME"
"$JEXTRACT_BIN" --version     # must report 22
./mvnw -Pfastervector,fastervector.jextract -pl jvector-fastervector -am compile
```

The script regenerates `ffi/`, adds the license header, and updates
`src/main/native/fvec_capi.h`. To generate from a header that isn't staged, set `FVEC_CAPI_H`,
for example `export FVEC_CAPI_H="$FASTERVECTOR_PINNED/fvec_capi.h"`. Commit `ffi/` and the header together, and update `FASTERVECTOR_SOURCE`. Don't use a
jextract newer than 22: its output calls `SymbolLookup.findOrThrow`, which doesn't exist
before JDK 23.

## Updating from fastervector

1. Point `$FASTERVECTOR_PINNED` at the new fastervector commit and run
   `jvector-fastervector/scripts/stage-fastervector-native.sh "$FASTERVECTOR_PINNED"`.
2. If it warns that the header changed, regenerate the bindings (above).
3. Copy over any changed Java code from fastervector, keeping this module's package names,
   its edits to `FastervectorLibrary`, `IvfAshIndex` and `IvfAshIndexSearcher`, and
   `IvfAshBatchingSearcher`, which is only here.
4. Update `FASTERVECTOR_SOURCE`, then run the build and tests with `requireNative=true`.

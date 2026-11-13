# JVector

JVector is a Java library for approximate nearest neighbor (ANN) vector search. It builds graph
indexes that combine an HNSW-style hierarchy with Vamana (the algorithm behind DiskANN) in every
layer, can be built concurrently and incrementally, and can be searched from disk with compressed
vectors. JVector targets Java 11 and uses SIMD acceleration on newer JVMs.

- [Why graph indexes](#why-graph-indexes)
- [How JVector searches](#how-jvector-searches)
- [Getting started](#getting-started)
- [Choosing a quantizer](#choosing-a-quantizer)
- [Benchmarking with BenchYAML](#benchmarking-with-benchyaml)
- [SIMD and native acceleration](#simd-and-native-acceleration)
- [Research](#research)
- [Building and testing](#building-and-testing)

## Why graph indexes

Exact nearest neighbor search is prohibitively expensive at high dimension: methods that partition
space well in two or three dimensions, such as k-d trees, degrade to linear scans as dimension
grows ("[the curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)"). For
large datasets, an approximate answer in logarithmic time is usually more useful than an exact
answer in linear time.

ANN indexes are either partition-based, like [LSH or IVF](https://www.datastax.com/guides/what-is-a-vector-index)
and [ScaNN](https://github.com/google-research/google-research/tree/master/scann), or graph-based,
like [HNSW](https://arxiv.org/abs/1603.09320) and
[DiskANN](https://www.microsoft.com/en-us/research/project/project-akupara-approximate-nearest-neighbor-search-for-large-scale-semantic-search/).
Graph indexes tend to be faster and, more importantly, can be built and updated incrementally, which
makes them a better fit for a general-purpose index than partitioning schemes that need the whole
dataset up front.

## How JVector searches

JVector's graph has an in-memory hierarchy of upper layers for fast navigation and a base layer that
can live on disk. Construction uses nonblocking concurrency control and scales linearly with cores:

![JVector scales linearly as thread count increases](https://github.com/jbellis/jvector/assets/42158/f0127bfc-6c45-48b9-96ea-95b2120da0d9)

Searches run in two passes. The first pass traverses the graph using compact, approximate scores
from a quantizer; the second reranks the best candidates with a more accurate representation stored
in the index:

| Pass | Options |
| --- | --- |
| Traversal | Product quantization (PQ), optionally with anisotropic weighting; binary quantization (BQ); Asymmetric Scalar Hashing (ASH); or full-precision vectors |
| Fused traversal | Fused PQ or fused ASH: each node's adjacency list carries its neighbors' codes, so the codes are read with the graph instead of held in memory |
| Reranking | Full-precision (FP32) vectors or NVQ, a high-accuracy non-uniform quantization |

[This two-pass design reduces memory use and latency while preserving accuracy](https://thenewstack.io/why-vector-size-matters/).
JVector can also use two-pass searches while *building* an index, so indexes larger than memory can
be constructed, and searches stay logarithmic within a single index instead of merging results
from many:

![Much larger indexes](https://github.com/jbellis/jvector/assets/42158/34cb8094-68fa-4dc3-b3ce-4582fdbd77e1)

## Getting started

JVector is published to Maven Central as `io.github.jbellis:jvector`. Run with these JVM options to
enable SIMD acceleration (see [SIMD and native acceleration](#simd-and-native-acceleration)):

```text
--add-modules jdk.incubator.vector
--enable-native-access=ALL-UNNAMED
```

The fastest way in is the hello-world programs in [jvector-examples](./jvector-examples). They download
small public datasets on first use. Build them once from the repository root:

```bash
mvn -pl jvector-examples -am -DskipTests package -Dnative.onednn=true
JAR=$(ls jvector-examples/target/jvector-examples-*-jar-with-dependencies.jar)
```

| Program | What it shows |
| --- | --- |
| [HelloVectorWorld](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/HelloVectorWorld.java) | Build an in-memory graph and search it, in about twenty lines |
| [HelloASHWorldFp16](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/HelloASHWorldFp16.java) | Train ASH, build a graph, write a fused ASH index with NVQ reranking, reopen it, and search synchronously and through the queued API |
| [HelloASHWorldInt8](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/HelloASHWorldInt8.java) | The same with INT8 query projection, for processors with VNNI or AMX |
| [HelloBatchedVectors](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/HelloBatchedVectors.java) and [HelloBatchedQuantization](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/HelloBatchedQuantization.java) | Batched access to file-backed vectors, and encoding them with PQ and NVQ |

```bash
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp "$JAR" io.github.jbellis.jvector.example.HelloVectorWorld
```

The [examples README](./jvector-examples/README.md) covers each program, and the
[tutorials](./docs/tutorials/) walk through the API step by step, starting with
[the introduction](./docs/tutorials/1-intro-tutorial.md).

Integrators should also read:

- [Batched input vector access](./docs/batched-vector-access.md) for file-backed input datasets.
- [Deferred batch serialization](./docs/deferred-batch-writer.md) for writing completed graphs.
- [UPGRADING.md](./UPGRADING.md) for API and on-disk format changes between releases.

## Choosing a quantizer

| Quantizer | Use it for | Notes |
| --- | --- | --- |
| **ASH** | Graph traversal at high compression | Most accurate at a given code size; fused or in memory; dot product only |
| **PQ** | Graph traversal at moderate compression | Fused or in memory; any similarity; optional anisotropic weighting |
| **BQ** | Very fast, coarse traversal | One bit per dimension |
| **NVQ** | Reranking | Stored in the graph; high accuracy at about a quarter of FP32's size |

### ASH

ASH (Asymmetric Scalar Hashing) reduces the dimension of database vectors with a learned orthonormal
projection, then scalar-quantizes each projected dimension to a few bits, while keeping queries in
their original form. At the same code size it is more accurate than additive and scalar quantizers,
trains and encodes quickly, and scores with SIMD lookups. JVector is the first graph index to support
it. On 6M and 10M datasets at matched code sizes, fused ASH built faster than fused PQ, was more
accurate in every case, and was faster in nearly every case; see [ASH performance](./docs/ash-performance.md).

Start with the HelloASHWorld programs above, then:

- [ASH configuration and defaults](./docs/ash-configuration.md): model parameters, construction
  scoring, precision and JVM settings.
- [Integrating ASH queries into a database](./docs/ash-query-integration.md): concurrency, bounded
  admission and batched query projection.
- [Choosing ASH query-projection precision](./docs/ash-projection-precision.md): FP16, FP32 or INT8
  by processor.
- [Building and serving larger-than-memory indexes](./docs/larger-than-memory-indexes.md): sizing,
  streamed construction, page-aligned records and query settings for indexes served from disk.

ASH scores dot product only, so use `DOT_PRODUCT` with normalized vectors for cosine similarity.
Fused ASH requires on-disk format version 7, which earlier JVector releases cannot read. Training
uses the system BLAS and LAPACK (MKL or OpenBLAS) when installed, with a slower Java fallback.

## Benchmarking with BenchYAML

BenchYAML builds and measures indexes from YAML descriptions of datasets, index parameters and
measurements. For a first run, benchmark one small public dataset from the repository root. It downloads
on first use (about 430 MB), then builds and searches 100,000 vectors in about a minute and prints one table:

```bash
mvn compile exec:exec@bench -pl jvector-examples -am -DbenchArgs=cohere-english-v3-100k
```

Without `-DbenchArgs`, the same command runs the default set of small public datasets.

Select datasets by name with `-DbenchArgs="cohere ada002"`, or run one index configuration by path,
such as an [example pipeline](./jvector-examples/yaml-configs/index-parameters/README.md). Compressed
datasets use fused ASH by default.

- [How to run benchmarks](./docs/benchmarking.md): datasets, parameters, and output.
- [BenchYAML Guide](./docs/benchyaml-guide.md): every construction and search field.
- [Benchmark launcher](./jvector-examples/bench-profiles/README.md): rebuild and run one experiment
  in one JVM from the command line or IntelliJ.
- [Reproducing ASH versus PQ](./docs/ash-pq-benchmark.md): the published comparison.

## SIMD and native acceleration

On Java 22 and later, JVector prefers its native provider, built on
[Google Highway](https://github.com/google/highway), when the native library is available. Native
libraries are bundled for Linux x86-64 (SSE4.2 or newer) and Linux AArch64. Java 20 and 21 use the
Panama Vector API, and older runtimes use a scalar Java implementation.

| JVM property | Effect |
| --- | --- |
| None | Prefer native on Java 22+, else Panama, else scalar |
| `-Djvector.enable_native_vectorization=false` | Disable automatic native selection |
| `-Djvector.vectorization_provider=panama` | Select Panama |
| `-Djvector.vectorization_provider=native` | Require native; fail if it cannot load |
| `-Djvector.vectorization_provider=default` | Select the scalar provider |

If native access is denied, automatic selection falls back to Panama. For named-module deployments,
enable native access for the module containing JVector's native code. Native ISA selection is
automatic and can be capped with `JVECTOR_MAX_ISA`.

The native provider stores vectors as `MemorySegmentVectorFloat`. Use the `VectorFloat` interface
rather than casting to `ArrayVectorFloat`; `FloatArray.array()` gives array access, and native
operations need vectors created by the selected provider.

ASH uses native kernels for query projection and scoring, and an optional add-on built on
[oneDNN](https://github.com/uxlfoundation/oneDNN) for FP16 and INT8 query projection and batch
encoding. Each falls back to a portable implementation when unavailable.

## Research

- Foundations: the [HNSW](https://ieeexplore.ieee.org/abstract/document/8594636) and
  [DiskANN](https://suhasjs.github.io/files/diskann_neurips19.pdf) papers, and
  [a higher-level explainer](https://www.datastax.com/guides/hierarchical-navigable-small-worlds)
- [ASH: Asymmetric Scalar Hashing With Learned Dimensionality Reduction for High-Fidelity Vector Quantization](https://arxiv.org/abs/2606.07870) (CIKM 2026)
- [NVQ](https://arxiv.org/abs/2509.18471)
- [Anisotropic PQ](https://arxiv.org/abs/1908.10396)
- [Quicker ADC](https://arxiv.org/abs/1812.09162)

## Building and testing

JVector is a [multimodule Maven build](https://maven.apache.org/guides/mini/guide-multiple-modules.html)
that produces a multirelease JAR usable from Java 11:

| Module | Contents |
| --- | --- |
| [jvector-base](./jvector-base) | Core library, compiled for Java 11 |
| [jvector-twenty](./jvector-twenty) | Panama Vector API provider, compiled for Java 20 |
| [jvector-native](./jvector-native) | Native provider and libraries, Java 22 |
| [jvector-multirelease](./jvector-multirelease) | Packages the three modules above as the published `jvector` JAR |
| [jvector-examples](./jvector-examples) | Examples, BenchYAML and benchmarks |
| [jvector-tests](./jvector-tests) | Tests, runnable on Java 11 and newer |

Build with JDK 22 or newer. The native library is built with [Meson](https://mesonbuild.com/) and
[Ninja](https://ninja-build.org/) and needs g++ 11 or newer; Meson fetches the pinned Google Highway
source on first build. The optional oneDNN add-on for ASH query projection is a separate Meson
project; it also needs CMake 3.13 or newer and takes a few minutes to build, so Maven builds it only
with `-Dnative.onednn=true` (the `release` profile and CI turn it on). The [native library README](./jvector-native/src/main/native/README.md) covers build
options, ISA dispatch, adding kernels and
[ABI compatibility](./jvector-native/src/main/native/README.md#native-abi-compatibility).
On Ubuntu, `bash build_native_lib.sh --auto-install-deps` (in `jvector-native/src/main/native/src`)
installs the toolchain and builds the library.

Run the tests with:

```bash
mvn test                                       # all modules, with the SIMD providers on JDK 20+
mvn -DskipScalar=false test                    # also run the scalar provider tests
mvn -Pjdk11 test                               # against Java 11
mvn -Dsurefire.failIfNoSpecifiedTests=false -Dtest=TestNeighborArray test   # one class
```

`-Dtest` also accepts methods and patterns, such as `TestNeighborArray#testRetain*`.
`failIfNoSpecifiedTests=false` is needed because Surefire otherwise requires a match in every module.

To release, configure `~/.m2/settings.xml` for OSSRH and run `mvn -Prelease clean deploy`.

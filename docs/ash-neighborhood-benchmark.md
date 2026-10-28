# ASH neighborhood scoring and isolated full-scan benchmarks

The graph decoder and `DistancesASH` use the same `ASHNeighborhoodScorer`.
It prepares a query once, then scores complete serialized neighborhoods, including
binary16 headers, landmark correction, and DOT_PRODUCT similarity conversion.
The graph path adds neighborhood I/O and traversal around this scorer. The
benchmark invokes the scorer directly, so those graph operations are untimed.

## Code organization and ownership

| Component | Responsibility |
| --- | --- |
| `ASHNeighborhoodScorer` | Immutable projection/LUT state for one query; selects scalar, Panama, or native execution |
| `FusedASHDecoder` | Thread-confined graph I/O, upper-layer source scoring, reusable scratch, and origin cache |
| `FusedASHLayout` | Shared packed layout, checked byte-size calculations, and canonical-to-fused conversion |
| `NativeVectorUtilSupport` | Native provider capability and delegation |
| `NativeASHKernels` | Package-private, checked FFM binding and native memory lifetime |
| `ash_neighborhood-inl.h` | Highway lookup, accumulation, header decoding, and score conversion |
| `ASHNeighborhoodBenchmark` | DistancesASH options, diagnostics, and timing/reporting |
| `ASHNeighborhoodFixture` | Untimed input loading, model/encoding creation, serialization, and replay |
| `ASHNeighborhoodValidator` | Untimed double-precision decoded-component oracle |
| `ASHNeighborhoodPipeline` | Persistent scoring workers and preparation/scoring overlap |

The prepared scorer can be shared across threads if inputs remain immutable and
outputs are separate. A graph decoder owns mutable scratch and a cached origin,
so each decoder belongs to one graph view/search thread. Upper-layer scoring and
L0 neighborhoods share one query projection. A failed neighborhood read or score
invalidates the decoder's cached origin before any output can be reused.

For one-bit ASH, canonical scoring uses `A(q - mu)` while fused scoring uses `Aq`.
`FusedASHLayout.packForFused` adjusts the one-bit offset accordingly. Two/four-bit
canonical offsets already contain that correction. The writer and fixture builder
use the shared helper rather than maintaining separate scoring formulas.

## Native C=1 execution

With `jvector.vectorization_provider=native` and
`jvector.ash.blockKernel=auto` or `simd`, C=1 neighborhoods use Highway. The native
path supports 1/2/4 bits per projected dimension, block widths 8/16/32, partial
blocks, and neighborhoods containing multiple blocks. C>1 retains the existing
scorer. Selecting the Panama provider retains the Panama kernel; selecting
`blockKernel=scalar` disables acceleration for the LUT scoring path.

The unit passed across FFM is **one entire neighborhood**. Its native entry,
`ash_score_neighborhood_c1_f32`, reads packed codes and the existing float LUT,
decodes headers, applies landmark correction and similarity conversion, and writes
all configured lanes. It retains the disk format and float scoring formula.
No additional byte-LUT quantization is introduced.

The AVX-512 B32 path shares each pair of 16-float register lookup tables across
both 16-lane halves and uses four independent accumulators. Headers are stored as
big-endian binary16 arrays and decoded with vector byte swaps and half-to-float
conversion. AVX2 uses indexed float LUT loads; the SSE software conversion includes
explicit handling for NaN and infinity. AVX3_DL and AVX3_SPR inherit the AVX3 ASH
slot, so an `avx3_spr` dispatch report currently executes that AVX-512 implementation.

### FFM safety and lifetime

`NativeASHKernels` lives outside the jextract-generated `cnative` package and
exposes no public raw-pointer interface. The provider detects the optional ASH
symbol when its ASH capability is queried, after library loading. Libraries without
that symbol retain the existing Java LUT path; unrelated native operations do not
acquire a dependency on the new kernel. An explicitly requested native benchmark
fails if that capability is absent, preventing a mislabeled fallback measurement.
The binding validates dimensions, lengths, and LUT/output non-aliasing before C++. Native output is unspecified on error. Status values are part
of the ABI: 0 means success, 1 an invalid C=1 landmark, and 2 an invalid shape.

Normal graph neighborhoods use a short critical downcall with direct heap-array
access. No copies, Java callbacks, pointer retention, or native allocation occur
on that path. Critical work is bounded at 65,536 charged lane/group contributions;
charging includes full serialized blocks and at least 16 SIMD slots per block.
This bounds arithmetic work even for a tiny output with an enormous LUT. The
budget is independent of CPU model and dataset; it is a conservative safety limit,
not a measured wall-clock guarantee or an optimized copying crossover. For example,
B32 / 512 projected dimensions uses at most 16,384 contributions.

Larger legal workloads use one ordinary downcall with temporary confined native
buffers. Their copy/allocation cost remains part of scoring, and the JVM can reach
safepoints during the longer call. Both paths retain the one-neighborhood-per-call
contract. Enable `jextract.trace.downcalls=true` only for untimed diagnosis.

Native float vectors are heap-backed. Their Vector API stores write directly to
the backing float array, which avoids the HotSpot 22 `intoMemorySegment` compiler
crash encountered during development. Compiler exclusions are not required by the
reviewed implementation. The regression suite exercises compiled stores at every
float vector width and nonzero offsets.

## Build and launch with JDK 22

The reported experiments use JDK 22. Runtime and toolchain selection belong to the
launcher and the repository's module requirements; the benchmark imposes no exact
runtime-version check. The examples module retains the repository's Java 16
source/API compatibility; the native module uses the repository's JDK 22 FFM API.

The native build uses the repository's existing Meson targets and compiler policy.
The pre-existing Sapphire Rapids target documents GCC 12+ or Clang 14+ requirements;
this ASH change adds no ISA target, compiler selection, or higher compiler floor.
For the reported benchmark server, the launcher sets `CXX=/usr/bin/g++-12`.
Production toolchains continue to use Meson's existing selection behavior.

The build script retains the original dependency installation, command tracing,
version check, and resource-publication behavior, including deleting the old
resource library before compilation. Its sole change is to omit `--wipe` when no
configured Meson tree exists, supporting first builds on older Meson releases.
Configured trees still wipe exactly as before. A failed build continues to stop
Maven and leaves no old resource library to package.

In IntelliJ, run `io.github.jbellis.jvector.example.DistancesASH` with JDK 22 and
these VM options, changing the fixture path as appropriate:

```text
--add-modules=jdk.incubator.vector
--enable-native-access=ALL-UNNAMED
-Xms2g -Xmx2g
-XX:ActiveProcessorCount=1
-XX:UseAVX=3
-Djvector.vectorization_provider=native
-Djvector.ash.blockKernel=simd
-Djvector.bench.neighborhood-scoring=true
-Djvector.bench.neighborhood-scaling=true
-Djvector.bench.scoringThreads=1
-Djvector.bench.neighborhoodAccess=shuffled
-Djvector.bench.neighborhoodFixture=/path/to/cohere-c1-b32-2bit.ashn
-Djvector.bench.neighborhoodSize=32
-Djvector.bench.blockSize=32
-Djvector.ash.bitsPerDimension=2
-Djvector.ash.quantizedDimensions=512
-Djvector.ash.landmarkCount=1
-Djvector.ash.optimizer=random
-Djvector.bench.maxVectors=99685
-Djvector.bench.maxQueries=4
-Djvector.bench.scoringPasses=5
-Djvector.bench.scoringMinSeconds=0.75
-Djvector.bench.scoringWarmupSeconds=2
```

To create a fixture, supply `base.fvecs query.fvecs` as program arguments. An
optional third ground-truth argument is accepted for launcher compatibility and
is unused. Input loading reads only the requested prefixes; queries are normalized
and base vectors retain their original values. Models, encoded vectors, and packed
neighborhoods are created outside timing. Replaying an existing fixture needs no
program arguments and performs no training, encoding, or packing.

Fixtures are published only after a complete write; existing destinations are
preserved. Separate names are required for different bit widths or shapes.
Replay checks version, lengths, model dimensions, query values, payload sizes, and
truncation/trailing data. The printed SHA-256 identifies the exact stored inputs.
Fixture format version 1 and its existing serialized bytes remain compatible.

For command-line launches, build the reactor and resolve a classpath with the
repository's Maven tooling. Set CPU affinity explicitly, for example
`taskset -c 2 "$JAVA_HOME/bin/java" ...`, using the same VM options and project
classpath as IntelliJ. `ActiveProcessorCount` changes JVM ergonomics; affinity
sets the physical CPU budget.

## Measurement contract

One full scan prepares the selected saved query exactly once, then scores every
saved neighborhood exactly once. Headers, score correction, similarity conversion,
and checksum consumption are included. The last neighborhood is padded to capacity
and fully scored. Output reports both real base-vector throughput and scored-lane
throughput, so padding is visible.

With `neighborhood-scaling=true`, a fixed two-scan pipeline prepares subsequent
queries while persistent workers score earlier scans. Every scan owns independent
prepared state. Pool threads own their output scratch. Timing begins before the
first preparation and ends after all submitted scans complete: preparation, task
submission, completion, checksums, pipeline fill, and drain are included.

`ns_per_scan` in this mode is total elapsed time divided by completed full scans,
a throughput metric. `ns_preparation_per_scan` is an overlapping diagnostic from
inside that same interval and must not be added again. The same selected saved
query is repeated across scans; no multi-query SIMD batching is used. The regular
non-scaling mode measures serial full scans on the calling thread instead.

The preparation coordinator, scoring workers, and JVM helper threads share the
stated CPU affinity. One-core experiments have no additional preparation core.
For scaling, select one hardware thread per physical core and keep NUMA placement
consistent. A full scan partitions the same neighborhood sequence at each worker
count; input work does not grow with worker count. Actual preparation bottlenecks,
contention, and scheduling overhead stay visible in the measured result.

Validation is independent of the LUT kernel and projects queries in double
precision. It checks serialized header effects and uses the unchanged tolerance
`3e-5 + 3e-5 * abs(expected)`. Both expected and actual scores must be finite.
Validation, fixture I/O, and worker creation are outside the timed scan stream.

### Comparison controls

Use fresh JVMs for `panama/scalar`, `panama/simd`, and `native/simd`. The scalar
comparator selects the scalar LUT kernel; its query preparation still uses the
Panama provider. Keep fixture hash, query index, bit width, block shape, access
order, JDK, heap, warmup, and affinity identical. The `Scorer:` line identifies the
actual execution path; the native provider's kernel description identifies its ISA.

The initial Cohere fixtures fit in the server's shared last-level cache. These
measurements describe repeated in-memory scoring, without disk I/O. Results are
specific to the tested workload and hardware. Pipelined full-scan throughput does
not by itself describe adaptive graph traversal latency or graph-search scaling.

| Property suffix under `jvector.bench.` | Default | Meaning |
| --- | --- | --- |
| `neighborhoodSize` | 32 | Scored capacity when creating a fixture |
| `blockSize` | Chosen from capacity | Serialized width: 8/16/32 |
| `neighborhoodAccess` | sequential | Base order; `shuffled` uses a reproducible permutation |
| `queryIndex` | 0 | Saved query to repeat; validation uses all saved queries |
| `maxVectors` | 8192 | Base input prefix used during fixture creation |
| `maxQueries` | 8 | Query input prefix saved during fixture creation |
| `scoringThreads` | 1 | Persistent scoring workers in pipelined mode |
| `scoringPasses` | 5 | Samples per JVM |
| `scoringMinSeconds` | 0.5 | Minimum sample duration before complete drain |
| `scoringWarmupSeconds` | 1.0 | Warmup duration |
| `validationNeighborhoods` | 1024 | Evenly spaced neighborhoods checked per saved query |

The former `neighborhoodsPerQuery` setting is rejected: a full scan always covers
the complete fixture. Malformed integer options fail rather than selecting a
silent fallback value.

## Regression checks

Run the ASH regression classes with native, Panama, and the default scalar
provider. The focused set is:

```text
NativeVectorStoreTest
TestASHNeighborhoodScorer
TestASHNativeScoring
TestASHNeighborhoodBenchmark
TestASHLutScoring
TestASHStandaloneScoring
TestASHStandaloneVsFusedScoring
TestASHGraphSearch
TestFusedASHBlockPacking
TestASHScoringDispatch
```

For example, with JDK 22 and an appropriate `CXX`:

```bash
mvn -B -pl jvector-tests -am test \
  -DskipSIMD=false -DskipScalar=true \
  -Djvector.vectorization_provider=native \
  -Dtest=NativeVectorStoreTest,TestASHNeighborhoodScorer,TestASHNativeScoring,TestASHNeighborhoodBenchmark,TestASHLutScoring,TestASHStandaloneScoring,TestASHStandaloneVsFusedScoring,TestASHGraphSearch,TestFusedASHBlockPacking,TestASHScoringDispatch \
  -Dsurefire.failIfNoSpecifiedTests=false
```

Native-only ABI checks are skipped when another provider is selected. To test
lower native dispatch tiers on a capable host, use fresh JVMs with
`JVECTOR_MAX_ISA=avx2` or `JVECTOR_MAX_ISA=sse42`. The tests cover critical and ordinary
FFM calls, tails, multiple blocks, invalid shapes, aliasing, canaries, IEEE headers,
query lifetimes, failed workers, interruption, fixture replay, and real graph use.

## Full-scan recall@10

Use the same DistancesASH neighborhood entry point with `jvector.bench.recall=true`.
Set `jvector.bench.recall.groundTruth=/path/to/matching.ivecs` and optionally
`jvector.bench.recall.output=/path/to/new-results.tsv`. Output files are not overwritten.
`scoringThreads` now controls parallel independent queries; each query is prepared
once and scans every stored neighborhood. This mode returns before throughput timing.
It uses all saved queries rather than `queryIndex`; set `maxQueries` at fixture
creation to retain the complete query set. Use the entire base set for recall.

The top ten are selected directly from approximate scores without exact reranking.
Padding is scored but excluded from top-k, and equal scores prefer the smaller
original base ordinal in every backend. Recall is the intersection with the first
ten unique ground-truth IDs divided by ten, averaged over queries. Little-endian
ivecs rows are validated against the real base cardinality. The optional TSV
retains each query's hit count, ranked IDs and scores, enabling comparison beyond
the aggregate recall. An independent decoded-component check precedes evaluation
on up to four queries; recall itself uses every query and every real base vector.

Create each fixture once, then replay the identical SHA-256 under scalar, Panama
and native providers. In particular, the legacy DistancesASH canonical-vector
recall mode does not exercise the native whole-neighborhood Highway entry. The
neighborhood recall mode is required to validate that implementation.

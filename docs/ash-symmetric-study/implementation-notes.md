# ASH symmetric construction scoring

Implementation branch: `ash-symmetric-scorer`, based on `ash-dev` commit
`67f532a282c72b66e97c4b074242bea406c0eaa0`. No PR #522 deduplication changes.

## Derivation and storage

The final ASH paper (2606.07870v2), Appendix B equations B.2–B.4, gives

    dot(x,y) ~= sx sy dot(code_x,code_y) + ox + oy + ||mu||²

where `sx = ||x-mu|| / ||code_x||` and `ox = <x,mu> - ||mu||²`.
For the Java/C++ 2/4-bit projection representation, the stored offset is
`ox - sx <Wmu,code_x>`. The new scorer restores the latter term once per
encoded vector. Both calculations use stored FP16-rounded headers; recovering
`ox` consequently retains FP16 error. Graph construction remains restricted to C=1. The symmetric scorer does not change the encoded payload layout. Standalone scoring now supports up to 256 landmarks; this expands the prior reader/encoder range guard, so standalone C>64 compressor files will be rejected by older implementations even though the layout is unchanged.

The packed code layouts and offset convention follow
`tlwillke/ivf-ash-fastscan` revision `56d78d0`, particularly
`ash_projection_code.h` and the projection query code in `ash_query.cpp`, as
well as the existing Java `AsymmetricHashing` encoder and `ASHScorer` decoder.
The C++ IVF implementation provides the packed layout and header arithmetic;
it is not evidence for a graph-construction symmetric kernel.

The binary shortcut is `d - 2*popcount(xor(sign_x,sign_y))`, with masked tail
bits. This follows B.3/B.6 directly. The printed B.7 has an apparent constant
term error (`d^-1` instead of 1 after dividing by d); its prose also confuses
the AND dot product of 0/1 bits with XNOR agreement. The implementation uses
the algebraic identity and independent decoder tests.

For 2/4-bit codes, doubled component values are odd signed integers. The 2-bit
single kernels use weighted popcounts (scalar word or SIMD); the 4-bit scalar
fallback uses a shared 512-byte nibble-product table, while SIMD unpacks to
short lanes and multiplies. Integer sums are divided by four. Widths 3 and 5–9
use the existing generic sign-plus-extra layout. Binary range scoring applies
direct XOR/popcount pairs. C=1 multibit block scoring builds exact short-valued
lookup tables from the encoded query, then uses packed paired-nibble lookups
and bounded short accumulation followed by widening. This borrows the C++
fastscan implementation approach without adding another quantization step.
There is no per-pair raw-vector projection or SIMD gather.

The graph adapter applies `max(0,(1+dot)/2)` exactly once, matching asymmetric
ASH. The prior symmetric adapter returned raw dots while insertion search
used transformed similarities. That mixed incompatible scales during pruning.

## Defects in the previous scorer

For 2/4-bit ASH, `binaryVector` is empty and the payload is in `extraBits`.
The previous loop read no payload, yielding the same code-dot term `-d` for
every pair before header corrections. It also treated the projection-mode
offset as the unadjusted offset. For one-bit tails it counted complement
padding as matches. These defects predate the duplicate-neighbor work.

## Comparison modes

- Baseline: untouched ash-dev jar, including the broken construction scorer.
- Symmetric (default candidate): corrected symmetric node-to-node scoring;
  insertion search still uses the incoming raw vector asymmetrically.
- Asymmetric: benchmark adapter uses the existing query scorer and original
  source vectors for node-ID construction calls, including diversity checks.
- Symmetric-all: corrected symmetric scorer also used for insertion search.
- Asymmetric-cached: identical existing asymmetric math, with immutable query
  state precomputed once per node in parallel; retains the arrays for the build.

All modes use the existing graph algorithm, including its current duplicate
behavior. Query-time scoring remains the same asymmetric fused scorer.
Symmetric construction adds one float per encoded vector for 2/4-bit landmark
corrections. Asymmetric construction needs access to original vectors and
repeats the current scorer's projection/setup on each node-ID provider call.
The uncached and cached controls separate pairwise scoring cost from setup cost.

Run-level settings: 32X, 2 bits, C=1, ITQ, M=64, efConstruction=200,
overflow=1.2, hierarchy/refine/fused enabled, NVQ reranking, k=10 at 1x,
pruning disabled, adaptive termination at the existing default gamma=0.01. New graph build each run; existing identical compressors
are reused to isolate graph construction. JDK23, vector module, SIMD kernels,
128GB heap, RAID10 temp directory. Two repetitions use reversed mode order.
Provider setup is reported separately because the legacy graph-build timer
starts after provider initialization; the tables add it to measured graph time.
Writer/builder initialization outside that timer and initial ASH encoding are
not included; these are graph-build comparisons, not end-to-end indexing time.
The old provider setup was not separately timed and is treated as zero (it
only computed the landmark norm).

## Asymmetric construction profile

A 30-second JFR profile of the first Ada002 asymmetric construction run contains
14,333 execution samples: 96.29% of samples include `ASHScorer.precomputeQuery`,
and 86.90% have `PanamaVectorUtilSupport.ashDotRow` as the leaf method. Only
1.97% include `ashProjectionDot`. Inclusive percentages overlap and must not
be added. The first Ada002 asymmetric timing includes this profile overhead;
the reversed-order repetition is unprofiled.

This establishes repeated projection/setup as the main cost of the uncached
asymmetric adapter. It does not establish that asymmetric pairwise arithmetic
is intrinsically slower by the observed build-time ratio. A separate cached
adapter retains each immutable existing asymmetric scorer once per node to
measure the speed/memory tradeoff without changing its numerical algorithm.
For C=1 the existing scorer retains two d-length float arrays plus two scalar
float arrays per node, as well as object/array overhead. This is far larger
than the symmetric scorer's single cached float per node.

Existing on-disk graphs keep their existing topology. Rebuild to obtain the
new construction behavior; merely changing the reader does not repair edges
selected by the old scorer. The asymmetric query scorer and encoded payload
format are unchanged.

The proposed default remains a mixed construction strategy: accurate asymmetric search
for a newly supplied raw vector, compact symmetric scoring for existing-node
pruning and refinement. Fully symmetric insertion and fully asymmetric node-ID
scoring are benchmark controls, not additional production defaults.

The paper writes multibit components as odd integers (e.g. -3,-1,1,3 at two
bits). Java and the C++ projection decoder use half of those values
(-1.5,-0.5,0.5,1.5). The stored scale is correspondingly twice as large.
The reconstruction and symmetric product are unchanged by this convention;
the nibble table uses doubled integer components internally and divides its
product sum by four.

Score-quality probes are sampled database vectors, with self matches excluded,
and their reference neighbors are obtained by an exact input-space scan.
They diagnose construction-space score quality. Graph recall uses the dataset's
separate benchmark queries and ground truth; the two recall numbers should
not be interpreted as a shared recall ceiling.

## Validation and reproduction

Initial correctness fix: `baa1027c`. Expanded encoded kernels and flexibility: `cc5d0d1b`. Correct timing and isolated benchmark modes: `62bc6f6d`. Graph-context SIMD allocation fix: `1338d46f`. The expanded regression tests cover all widths 1–9,
dimensions 1–749 including every tail alignment, independent unpacking,
scalar/SIMD cross-checks, exact exchange symmetry, Appendix B reconstruction,
existing asymmetric scoring of reconstructed sources, FP16 header round trips,
zero residuals, negative raw scores, the graph C=1 guard, standalone C=2/64/256 across widths 1–9, and graph construction.
The focused suite also includes existing `TestASHGraphSearch` (2 tests) and
`TestASHScoringDispatch` (4 tests). Scalar and SIMD Maven executions passed, with both the preferred 512-bit SIMD shape and an additional forced-AVX2 256-bit check.

Build/test command (JDK23):

```sh
mvn -pl jvector-tests,jvector-examples -am \
  -Dtest=TestASHSymmetricScorer,TestASHGraphSearch,TestASHScoringDispatch,TestASHStandaloneVsFusedScoring \
  -Dsurefire.failIfNoSpecifiedTests=false \
  -DskipScalar=false -DskipSIMD=false -Dmaven.javadoc.skip=true package
```

Benchmark-only mode selection is
`-Djvector.bench.ashConstruction=symmetric|symmetric-all|asymmetric|asymmetric-cached`.
The library default has no new configuration switch. The example benchmark
adapter changes construction only; query scoring remains asymmetric.

JVM options used for every benchmark:

```text
--add-modules=jdk.incubator.vector
--enable-native-access=ALL-UNNAMED
-Xmx128g
-Djvector.ash.singleKernel=simd
-Djvector.ash.blockKernel=simd
-Djvector.ash.lut.accumulators=2
-Djvector.ash.lut.unroll=1
-Djvector.ash.projection.accumulators=1
-Djvector.ash.projection.unroll=1
-Djava.io.tmpdir=/mnt/raid10/jvector-bench/tmp
```

The main class is `io.github.jbellis.jvector.example.BenchYAML`; pass a YAML
with the settings above as its argument. Score diagnostics use
`-Djvector.ash.debugSymmetric=true`, with `.pairs=5000`, `.subsetN=100000`,
and `.topK=10` appended to the same property prefix. Diagnostics run before
graph construction; those extra diagnostic-run builds are excluded from the
comparison tables. Their single-thread scoring measurements exclude setup
and use the median of nine scans after five warmups. These are supporting
microbenchmarks, not graph throughput predictions or a replacement for JMH.

The remote evidence directory is
`/mnt/raid10/jvector-bench/ted_willke/ash-symmetric-study`.
It retains frozen baseline/candidate jars, per-run logs, the input-space score
study, JFR profile, and extracted `results.csv`. The baseline jar was compiled
from `67f532a2` before the scorer change. The first candidate jar is preserved
as `candidate-v1.jar`; the later jar adds cache and diagnostic controls with
identical production scorer arithmetic. No compressor training was retimed.

## DistancesASH prepared-query comparison

Commit `f8e64213` adds `ASHSymmetricScorer.scoreFunctionFor(QuantizedVector)`
and the benchmark mode `-Djvector.bench.symmetric-scoring=true`. Queries must
be encoded with the same trained compressor as the base vectors. This API
recovers the query landmark offset once, before returning the scoring function;
its timed kernel is the same `codeDot` implementation used by node-ID scoring.

In this mode DistancesASH prepares all compressed queries and both sets of
scoring functions before measurement. Query encoding and projection setup are
reported separately and excluded. Both scorers then use the same full-scan
loop, executor, query subset, checksum accumulation, and five measured passes
with alternating order. It reports median throughput and min/max elapsed time.
Warmup, independent decoded-oracle validation, and held-out-query recall are
outside all throughput timers. The ordinary historical benchmark mode remains
available; its timing includes scorer setup and must not be mixed with these
prepared-query results without labeling that distinction.

Example flags in addition to the shared JVM flags:

```text
-Djvector.bench.symmetric-scoring=true
-Djvector.ash.optimizer=itq
-Djvector.ash.landmarkCount=1
-Djvector.ash.bitsPerDimension=2
-Djvector.ash.quantizedDimensions=748
-Djvector.bench.recall=true
-Djvector.bench.recall.k=10
-Djvector.bench.maxQueries=1000
-Djvector.bench.scoringQueries=128
-Djvector.bench.scoringPasses=5
-Djvector.bench.scoringWarmupQueries=32
-Djvector.bench.scoringThreads=1
```

Call `io.github.jbellis.jvector.example.DistancesASH` with base `.fvecs`,
query `.fvecs`, and ground-truth `.ivecs` paths. At 32X with two bits, the
quantized dimension is `(originalDimension - 40) / 2`: 748 for Ada002 and
492 for E5-large/Cohere. The 40-bit header is included in the compression
budget. Generic bit widths require `singleKernel=scalar` for the asymmetric
control; that restriction belongs to the existing asymmetric implementation.
The binary symmetric path is popcount, 2/4-bit is packed nibble arithmetic,
and the remaining widths use the generic reference representation.

The tiny end-to-end fixture uses 129 base vectors and nine distinct held-out
queries at original dimension 64 / quantized dimension 31. All widths 1–9
pass the independent decoder validation, ground-truth recall plumbing, and
timing loops. The fixture timings are not used as performance evidence.


## Encoded-input contract and measurement

Symmetric distance starts from two encoded vectors. Prior projection and encoding
of both operands are excluded. Query-specific state derived from the encoded
query (such as LUT creation or folded-header recovery) is included in timing.
No symmetric scorer multiplies a raw vector by the projection matrix.
Asymmetric distance starts from a raw query and an encoded target: its query
projection and scorer setup are timed once per query, then reused over that
query's candidates. Recall selection is outside throughput timing.

The initial PREPARED_* results are kernel-only measurements with scorer setup
excluded for both methods. They are retained as historical data, not the final
end-to-end scoring comparison. SCORING_* results use the contract above.
Final microbenchmark modes should run in independent JVMs to avoid cross-mode
JIT profile effects. DistancesASH supports `jvector.bench.scoringModes` for this.
Warmup lasts at least one second per selected mode, and each timed pass lasts
at least 0.25 seconds by repeating query batches when necessary. Raw duration,
repetitions, and normalized scan duration are recorded separately. A 64-candidate
run measures setup amortization and is not graph QPS or a full scan.

## Multiple landmarks

Appendix B presents C=1. For standalone C>1, the implementation generalizes its
header-calibrated reconstruction rather than claiming an additional paper result:

    xhat = mu_a + sx W^T code_x
    delta_x = ox - sx <Wmu_a, code_x>
    score(x,y) = <xhat,yhat> + delta_x + delta_y

This expands into symmetric code-dot, cross-centroid, and centroid-dot terms and
reduces algebraically to Appendix B at C=1. Encoded-query state can be fed to the
existing asymmetric single/block kernels without a raw-vector projection. Tests
compare direct decoded reconstruction, exchanged operands, and scalar/SIMD paths.
Centroid IDs must be treated as unsigned bytes, including IDs 128–255. Serialization
round trips allow float-matrix rounding error; packed integer kernel parity is exact.


## Graph-context SIMD compilation

A fast isolated scorer was not sufficient evidence for a production build path.
The first combined SIMD dispatch body allocated boxed ByteVector and mask objects
when called from graph diversity pruning. A profiled Ada002 100k build took
24.7 seconds, versus 18.2 seconds for the earlier corrected scalar scorer.
Splitting the 1/2/4-bit Vector API kernels into small compilation units and
making matched vector species static constants removed that allocation pattern.
The first profiled split-SIMD build took 12.8 seconds; the improved scalar
word-popcount control took 15.5 seconds. Whole-run GC counts were 137 before
the split and 30 afterward. These are diagnostic runs; final tables use fresh
unprofiled runs. This is why the final recommendation is based on actual graph
construction as well as isolated full scans.


## Sustained graph-query timing

The original graph throughput harness measured each 10,000-query batch only
once. At hundreds of thousands of QPS those samples lasted only tens of
milliseconds, producing large variability and an unreproduced Cohere slowdown.
`f98fcf92` repeats complete query batches until each sample lasts at least one
second and records completed query count plus elapsed time. Synthetic warmup
queries remain unchanged. Both graph branches use the identical benchmark-only
overlay; the baseline library JAR remains byte-for-byte unchanged. The overlay
contains only `ThroughputBenchmark.class`, no graph, quantization, or SIMD classes.
Final graph results use this sustained timing protocol. Earlier short-sample
results and the construction-boxing experiment are archived separately.

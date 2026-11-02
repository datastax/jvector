# Native ASH projection and symmetric construction scoring

ASH models prepare an immutable provider-specific projection when their Stiefel
transform is constructed or loaded. The default implementation keeps the
existing row-dot implementation, including Panama SIMD. The native provider
copies the row-major float matrix once into automatically managed shared native
memory. The extra matrix storage is included in `ramBytesUsed()` and is released
when its model is no longer reachable. It is not serialized; the model and graph
formats are unchanged.

`ash_project_f32` computes the entire matrix-vector product in one Highway
dispatch. Four rows share query loads, with independent FMA chains and masked
tails. Every graph query still pays for its own projection. There is no batch of
unrelated queries, BLAS thread pool, per-row FFM call, or per-query matrix copy.
Small bounded products use a synchronous critical downcall with heap query and
output arrays. Larger products use an ordinary downcall with temporary native
query/output buffers so the JVM can reach safepoints during the dense multiply.
The matrix is shared read-only; native staging buffers are reused per calling
thread and grow only when needed.

`ash_symmetric_dot` evaluates encoded pairs used by graph construction: 1-bit
XOR/popcount; 2-bit weighted popcounts over packed sign/magnitude fields; 4-bit
unsigned-magnitude by signed-magnitude byte dot products. Integer accumulation
precedes a single factor of one quarter for 2/4-bit codes. Tail padding is ignored.
The Ice Lake vtable selects the same Highway source compiled with native
population-count and VNNI support, and Sapphire Rapids inherits that slot.
AVX-512 without those instructions, AVX2, scalar and ARM retain the portable
Highway implementations. The Java scorer still combines scales, centered
offsets, landmark terms and the existing similarity conversion.

`ash_symmetric_lut_i16` handles byte-interleaved ranges in one call, retaining
the existing exact signed-short LUT. It uses 32-bit partial sums and bounded
chunks accumulated into 64-bit totals, rather than letting a short or integer
sum overflow. The graph builder's pair-oriented API uses the pair kernel;
callers of `ASHSymmetricScorer.blockScorerFor` use the native LUT kernel. Existing
C>1 formulas and the graph-neighborhood scorer are unchanged. The graph study
uses C=1 throughout.

Projection and symmetric symbols are optional and looked up independently. Loading an older
compatible library preserves Java projection and symmetric implementations.
Java validates shapes, array bounds and output aliasing before passing pointers
to native code. Native kernels do not allocate, retain pointers or call Java.
Scorer diagnostics identify whether each operation is Highway or Java.

## Base-vector encoding

`encode()` and `encodeTo()` normalize the centered residual, project it through
`StiefelTransform.project()` in FP32, and apply the existing code quantizer and
header adjustment. This uses the prepared Highway projection with the native
provider and Panama SIMD with the Panama provider. Encoding never requests the
query-only FP16 view. Header scale and offset retain their existing FP16 storage
format; database projection precision remains FP32.

`encodeAll()` replaces per-vector projection with bounded batches on each worker
of the caller's `ForkJoinPool`. Each worker gathers up to 32 non-null vectors,
normalizes them immediately into reusable buffers, calls `projectBatch()` once,
and quantizes/packs the projected outputs in their original ordinal positions.
Immediate gathering is necessary because a vector source may lend the same
mutable vector object on each read. Missing vectors keep the existing all-zero
encoded representation. Empty sources and partial final batches are supported.
Every submitted task is joined before returning or propagating a worker error.

The Highway `ash_project_batch_f32` kernel is a blocked matrix multiply with
FP32 operands and FP32 accumulators. Its SIMD microkernel shares matrix weights
across four inputs; output tiles are reused over the full caller batch. It has
no internal threads. A second native FP32 matrix, tiled by up to 32 output
coordinates, is packed lazily once per model and shared read-only. The final
tile has no padding, so this backing is exactly the original matrix size and
is included in `ramBytesUsed()`. Single-vector queries do not allocate it.
Input and output staging is reused per calling thread. Bulk calls use ordinary
FFM to allow JVM safepoints during the multiply.

The worker batch size of 32 is an internal working-set bound, not an API or ISA
limit. `AshProjection.projectBatch()` accepts any nonnegative count that fits
the supplied row arrays. The default provider implementation calls its own
FP32 single-vector projector; the normal worker batch needs no temporary
validation collection or projection arrays. This preserves Panama fallback. An older native library without the batch symbol uses its available
single-vector projection. There are no leader groups, arrival queues, timers,
public tuning controls, or new dependencies.

Changing SIMD reduction order can produce small FP32 differences near code
boundaries; FP32 does not imply bit-identical arithmetic across providers.
Regression tests use a double-precision projection oracle and exercise native
and Panama batch tails, input/alias rejection, concurrent models, and FP32
operand precision. Encoding tests cover missing/borrowed vectors, supported bit
widths, multiple landmarks, zero residuals, destination reuse, source failure,
and independence from the query precision setting. Existing model and graph
formats are unchanged.

## Training sample preparation

For a single landmark (`C=1`), `AsymmetricHashing.initialize()` computes the mean,
PCA inputs and ITQ training inputs from one uniform reservoir sample of
`min(N, 20 * D)` vectors, where `D` is the original dimension. The sample size
is independent of the projected dimension and encoded bit width. Selection uses
the existing deterministic seed, 103, and selected ordinals are sorted before
reading. Only selected source vectors are fetched, once each; encoding performs
the full pass over the source vectors afterward.

Each selected vector is copied into one reusable normalization buffer before
another source read, so borrowed source storage is never retained or modified.
ITQ retains normalized sample values in a flat FP32 array. The mean is accumulated
in FP64 and stored as the existing FP32 landmark. Centered residuals are normalized
in FP64 and copied in bounded tiles directly into the column-major training
matrix, avoiding an array of per-vector objects and a second full-size FP64
matrix conversion. Preparation retains bounded serial transpose tiles.
The existing BLAS/LAPACK backend is retained. PCA uses checked
FP64 covariance eigendecomposition, with guarded direct SVD fallback, as described
in the [PCA validation contract](ash-pca-native-svd.md).

The `RANDOM` optimizer estimates its single landmark from the same deterministic
sample, using an independent sampler so the Gaussian projection's existing random
sequence is preserved. Multi-landmark training retains its clustering and full
input-materialization path.

Empty sources, invalid dimensions/configurations, and training matrices that
cannot fit a Java array are rejected before gathering vectors. Selected inputs
must be non-null, have the declared dimension, and have a finite norm under the
provider's FP32 normalization. Non-finite values and FP32 norm overflow are
rejected with the offending ordinal. Zero and small-norm vectors retain the
existing normalization behavior. Source read failures propagate to the caller.
Training validates selected vectors only; it does not fetch unsampled vectors
for validation. Allocation failures are not swallowed or retried with a different
training sample.

Saved model and graph formats are unchanged. Loading an existing model preserves
its stored landmark and projection. Retrain the compressor and rebuild its graph
cache to adopt sampled-mean training; the new mean is used consistently by
encoding and scoring.

## Reference implementation review

Reviewed IVF ASH Fastscan at commit
[`56d78d0ff3aaaa61a02b94c6f49a97789706a717`](https://github.com/tlwillke/ivf-ash-fastscan/tree/56d78d0ff3aaaa61a02b94c6f49a97789706a717).
Its [`AshProjectionModel::transform_residual_batch`](https://github.com/tlwillke/ivf-ash-fastscan/blob/56d78d0ff3aaaa61a02b94c6f49a97789706a717/ash_transform.cpp#L727)
uses a contiguous input-major projection and BLAS matrix multiplication. Its
[`compute_lut_uint8`](https://github.com/tlwillke/ivf-ash-fastscan/blob/56d78d0ff3aaaa61a02b94c6f49a97789706a717/ash_query.cpp#L347)
path batches projections before building query-local LUTs. The
[`AVX-512 scan`](https://github.com/tlwillke/ivf-ash-fastscan/blob/56d78d0ff3aaaa61a02b94c6f49a97789706a717/ash_fastscan_avx512.cpp)
uses packed nibbles, byte shuffles and independent integer accumulators.

The transferable principles here are preparing contiguous immutable weights,
amortizing native dispatch over a complete operation, reusing query loads, and
accumulating compact codes without materializing decoded vectors. This port
implements a single-query Highway matrix-vector operation because graph query
latency must include preparation for each query. The IVF implementation does
not provide a graph-construction symmetric scorer. Its quantized uint8 query
LUTs are not substituted for the graph's existing exact symmetric integer LUTs.
The kernels in this change are independent implementations of these operations.

## Validation and measurement

Targeted regression checks compare projection with a double-precision oracle,
packed pair dots with an independent decoder, and integer LUT ranges with
64-bit scalar accumulation. They cover dirty tail bits, irregular dimensions,
both FFM modes, concurrent projections, bounds/alias rejection and absent native
symbols. Both JDK 22 and JDK 25 are checked with normal dispatch and the existing
AVX3_DL, AVX3, AVX2 and SSE/scalar ISA caps. The generic kernel translation unit
also compiles for ARM NEON on Apple Silicon. SVE/SVE2 execution is not validated
on this host.

Performance collection uses unchanged standard BenchYAML, existing inclusive
timers, normal query preparation, deferred writes and NVQ reranking. Before and
after query runs load the same saved model and graph, retain stock warmup and
samples, and use 1 or 8 scoring threads pinned to physical cores. Traced small
BenchYAML preflights confirm actual native calls; tracing is disabled for all
performance measurements. Separate fresh full-data builds verify construction.
The focused cases are MPNet at the 2-bit ASH budget and OpenAI3072 at the 4-bit
ASH budget on both JDKs. They are not a replacement for the earlier full PQ/ASH
matrix. Full results and measurement scope accompany the saved report.

## FP32 encodeAll measurements (JDK 22, October 31, 2026)

These are standard BenchYAML `construction.index_quant_time_s.ASH.encoding_time_s`
measurements for four 1M-vector datasets, ITQ, C=1 and D/2. They time the complete
`encodeAll()` operation, including normalization, projection, quantization and
result packing. They are not timings of individual `encode()` calls. Training
and graph/write are separate metrics. The first native packing cost is included.

Both runs use physical CPUs 0–47 with memory interleaved across NUMA nodes 0 and
1. The before version is `16273e24` (per-vector Panama FP32 encoding); the after
code is `83e49aac` (Highway FP32 worker-local batches). Both use fresh model and
index caches, the native scoring provider, deferred parallel writes, NVQ
reranking, and FP16 query projection. All eight serialized trained models are
byte-identical between the two runs. Each cell is one full construction run;
these are not repeated-run averages.

| Dataset | ASH case | Before (s) | After (s) | Speedup |
| --- | --- | ---: | ---: | ---: |
| MPNet | 2-bit | 1.140 | 0.800 | 1.43× |
| MPNet | 4-bit | 1.205 | 0.986 | 1.22× |
| Cohere | 2-bit | 1.699 | 0.960 | 1.77× |
| Cohere | 4-bit | 2.049 | 1.264 | 1.62× |
| Ada002 | 2-bit | 5.793 | 1.344 | 4.31× |
| Ada002 | 4-bit | 6.057 | 1.896 | 3.19× |
| OpenAI3072 | 2-bit | 20.621 | 3.641 | 5.66× |
| OpenAI3072 | 4-bit | 25.240 | 4.712 | 5.36× |

The current code was checked on JDK 22 with native and Panama providers, and
native batch projection with the AVX2 and SSE/scalar ISA caps. The native module
passed 12 projection/FP16 tests; the general ASH suite passed 41 tests with two
provider-specific skips under native, and 39 with four skips under Panama.
Each lower-ISA projection pass ran four tests. The optional standalone Meson
C++ test targets were unavailable on this host; the Java native tests invoke
the actual dispatched C++ kernels and require the new batch symbol.

Fresh graph recall was also recorded, but these independent parallel graph
builds do not isolate an encoding-only recall effect. Model and graph formats,
query FP16 arithmetic, and scoring kernels were not changed by this extension.

## Preferred oneDNN FP32 encoding projection (November 1, 2026)

The native provider automatically prefers the existing sequential oneDNN add-on
for FP32 `projectBatch()` when its new entry point is available. It executes row-major
SGEMM as `[B,D] * transpose([d,D]) -> [B,d]`, with FP32 inputs, weights and outputs.
The adapter shares the immutable Highway matrix, avoiding a second packed model
copy, and reuses the existing calling-thread native input/output buffers. Its
batch API accepts arbitrary positive counts; no eight-vector cap or padding
controls were introduced. The library's existing `encodeAll()` bound remains 32.
The sequential-runtime build assertion prevents nested oneDNN thread pools.

Single-vector `encode()` retains Highway FP32 projection. The wrapper delegates
query precision selection to the same original projection: FP16 queries remain
FP16 in automatic/required modes, with their existing FP32 fallback. Database
encoding never converts to FP16. Java code selection and landmark correction
remain unchanged. The slower experimental native quantizer was archived outside
the production tree after repeated MPNet 2-bit regressions; it is not packaged.

The optional library retains its existing package/resource name for compatibility.
Both FP32 and FP16 loaders validate ABI 1, handle missing/denied/unsupported
libraries, and retain Highway when optional support is absent. An older add-on
with ABI 1 but without the FP32 symbol also keeps Highway. Unsupported oneDNN
FP32 execution falls back to Highway batch projection; invalid arguments,
allocation failures and unexpected runtime errors propagate. FP16 additionally
checks representable range, and explicit `fp16` selection requires support.
No production selection flag, benchmark-specific path or BenchYAML change was
added. The existing `-Dnative.onednn=true` build packages both operations.

BLAS kernels can use different FP32 reduction orders for different batch shapes.
The 9-bit reconstruction check therefore uses its existing per-component code
step error bound for worker/batch comparisons as well as singleton comparisons;
sign bits, offsets and the full 1/2/3/4-bit code/header checks remain exact in
those tests. Separate tests compare oneDNN output to a double-precision oracle,
cover FP32 values outside FP16 precision/range, caller aliasing and invalid
shapes, batch tails/counts above 32, immutable model copies and concurrent model
switching. A direct ABI test requires successful oneDNN SGEMM, preventing a
Highway fallback from passing as a oneDNN test.

### Repeated BenchYAML measurements

Each case used three alternating matched control/candidate pairs with JDK 22,
CPUs 0-47 and NUMA interleave across nodes 0 and 1. Both variants include the same
backing-array residual-normalization improvement and Java code generator. Both
use C=1, ITQ, D/2, deferred parallel writing and NVQ reranking. Each process builds
fresh model/index caches; input/output copies, library/JIT first use and scratch
allocation are included in BenchYAML's existing `encoding_time_s` field. These
are `encodeAll()` times, not standalone `encode()` or query-throughput results.

| Case | Highway median (range), s | oneDNN median (range), s | Ratio of medians |
| --- | ---: | ---: | ---: |
| mpnet-1m, 2-bit | 0.665 (0.622-0.665) | 0.589 (0.550-0.604) | 1.13x |
| openai-3072-1m, 4-bit | 4.055 (3.996-4.092) | 3.926 (3.913-3.955) | 1.03x |

oneDNN was faster in each of the three matched pairs for both datasets. The
OpenAI3072 gain is modest; these measurements do not establish improvements for
all other datasets, code widths or hardware. Median graph recall@10 was 81.901%
versus 81.844% for MPNet and 79.280% versus 79.310% for OpenAI3072. Independent
parallel graph rebuilds vary and do not isolate an encoding-only recall effect.
Training and graph-write timers/recall for every run are retained in the raw
CSVs; they are not manually added to inclusive harness timers. Original CSVs,
commands, class/source snapshots and live placement observations are preserved
under `benchmark-runs/ash-quantization-20261101` on the development server.

## ITQ training workspaces

ITQ keeps the residual and PCA-reduced inputs in column-major FP64 arrays for
BLAS/LAPACK. Its iterative projection writes contiguous training vectors directly:
`(Xld * R)^T = R^T * Xld^T`, evaluated by one DGEMM. The result is viewed as
row-major `[N,d]` by the direction encoder. Quantization replaces each projected
row in place; the unquantized result is not needed by the update or loss. A second
DGEMM consumes this layout to overwrite the existing update matrix with
`Xld^T * Xenc`. No explicit transpose pass or second `[N,d]` output is required.

Each trainer owns an `ASHFloatOrthogonalizer` with one thin-SGESDD workspace for
repeated polar steps. It rounds the small update matrix to FP32 into reusable
scratch, decomposes it, multiplies `U * VT` with SGEMM, and widens the rotation into
the trainer's FP64 buffer. Its workspace is queried once; singular vectors,
values and native work arrays are reused. The destroyed SVD input also holds
the polar product, avoiding another output array.

After the last update, the trainer computes a fresh FP64 polar factor with
`ASHOrthogonalizer` before forming the final projection. This preserves the
final transform's orthogonality contract. Random projection initialization also
retains FP64. Both implementations use the existing portable BLAS/LAPACK provider;
there is no new native dependency, vendor selection or precision control.

The direction encoder retains scratch for its logical workers across iterations.
Workers operate on disjoint rows in the shared physical-core executor, without
the previous fixed 16-worker cap. Each trainer owns its direction and polar
workspaces; scaling-event buffers remain thread-local. Parallel task submission
still occurs each iteration; this is not an allocation-free training API.

PCA forms the upper triangle of `X^T X` with FP64 DSYRK and obtains eigenvectors
with DSYEVD. Spectral energy, retained-tail conditioning, the cutoff gap and
selected eigenpair residuals are checked before accepting the result. Uncertain
or underdetermined inputs use the guarded FP64 DGESDD fallback, including its
complete null-space basis for wide matrices. No full left-singular-vector matrix
is computed in the common covariance path.

Large ITQ projection/update products, norms, update storage, loss and the stopping
criterion remain FP64. The scaling-factor search retains its existing FP32
arithmetic and event ordering. The 1-bit sign rule and 2–9-bit direction quantization are
unchanged; tests compare the new in-place encoder to the former column-major
algorithm bit for bit on identical projected inputs.

Different DGEMM orientations can change reduction order, so retraining need not
produce a byte-identical matrix. Model serialization, database encoding and query
projection precision are unchanged. Existing saved models remain compatible.
This structural change does not select a different BLAS/LAPACK backend.

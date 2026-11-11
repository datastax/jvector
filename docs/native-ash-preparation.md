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
C>1 formulas and the graph-neighborhood scorer are unchanged.

Projection and symmetric symbols are optional and looked up independently. Loading an older
compatible library preserves Java projection and symmetric implementations.
Java validates shapes, array bounds and output aliasing before passing pointers
to native code. Native kernels do not allocate, retain pointers or call Java.
Scorer diagnostics identify whether each operation is Highway or Java.

## Query-projection precision

See [Choosing ASH query-projection precision](ash-projection-precision.md) for hardware recommendations and selection behavior. Explicit INT8 uses the existing immutable query-view interface, batch validation, caller-owned scheduling and Java thread-local staging. Its model-owned native context holds quantized weights, scales and distinct oneDNN packed layouts; compatible batch shapes share packed weights. Calling-thread scratch retains no model.

The optional oneDNN ABI and status structure are shared with FP16. INT8 checks ABI compatibility before binding, distinguishes unsupported hardware, invalid operands/ranges, allocation failures and runtime failures, and never silently substitutes another precision. Full-range integer dot products require a qualified VNNI or AMX implementation; legacy intermediate saturation is rejected. Scales, quantization and output rescaling retain the prototype’s FP32 arithmetic. Nonfinite operands and a nonzero operand whose scale rounds to zero are rejected before integer conversion. Output rescaling retains ordinary FP32 overflow behavior. INT8 has no unconditional native console output. The original FP16 and FP32 implementations and automatic-selection policy are unchanged.

## Direct-node scoring and query scratch

Fused 2-bit direct-node scoring uses `ash_canonical_lut_dot_f32` and reuses the
neighborhood scorer's FP32 nibble LUT. One lookup combines two projected
coordinates. For 4-bit codes each nibble represents only one coordinate, so its
LUT is 16 times larger than the projected query. The preferred native
`ash_projection_dot_4bit_f32` instead decodes packed nibbles in registers,
loads contiguous query coordinates and performs one final SIMD reduction. This
also supplies the native provider's existing 4-bit projection-code dot API.
Neither operation materializes decoded vectors. Java applies the existing
scale, landmark correction, offset and graph similarity conversion.

Normal codes use one bounded synchronous native call per direct node with heap
access, no native copy and no output array. Unusually large legal codes use an
ordinary call with native staging so the JVM can reach safepoints. Both symbols
are optional independently: older native libraries and other providers retain
the previous projection-code scorer. Forced scalar scoring remains an
independent reference. The 4-bit choice follows the representation's working
set and does not depend on dataset names, dimensions or benchmark settings.

Each on-disk graph view lazily owns its ASH input, projection, landmark terms,
LUT, neighborhood bytes and output arrays. Subsequent queries refill this storage.
Projection codes skip the generic centered-query arrays and sums, which their
scoring formula does not consume. Query projection precision and model/index
formats are unchanged. Separate views have independent scratch even on the same
thread. A view is confined to one active search, including any `resume()` calls;
starting another query invalidates the previous decoder, whose scoring methods
then throw `IllegalStateException` instead of reading overwritten query data.
Public standalone scorers retain independent prepared state and can be retained
across subsequent queries. The reusable path still allocates small scorer objects;
it does not claim an allocation-free search API.

## Base-vector encoding

`encode()` and `encodeTo()` normalize the centered residual, project it through
`StiefelTransform.project()` in FP32, and apply the existing code quantizer and
header adjustment. This uses the prepared Highway projection with the native
provider and Panama SIMD with the Panama provider. Encoding never requests the
query-only FP16 or INT8 view. Header scale and offset retain their existing FP16 storage
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
accumulating compact codes without materializing decoded vectors. JVector
implements a single-query Highway matrix-vector operation because graph query
latency must include preparation for each query. The IVF implementation does
not provide a graph-construction symmetric scorer. Its quantized uint8 query
LUTs are not substituted for the graph's existing exact symmetric integer LUTs.
JVector's kernels are independent implementations of these operations.

## Validation

Targeted regression checks compare projection with a double-precision oracle,
packed pair dots with an independent decoder, and integer LUT ranges with
64-bit scalar accumulation. They cover dirty tail bits, irregular dimensions,
both FFM modes, concurrent projections, bounds/alias rejection and absent native
symbols. Both JDK 22 and JDK 25 are checked with normal dispatch and the existing
AVX3_DL, AVX3, AVX2 and SSE/scalar ISA caps. The generic kernel translation unit
also compiles for ARM NEON on Apple Silicon. SVE/SVE2 execution is not yet validated.

## Batched FP32 encoding

`encodeAll()` projects worker-local batches with Highway FP32 instead of
projecting one vector at a time. On four 1M-vector datasets (ITQ, C=1, D/2,
48 physical cores) this reduced BenchYAML's ASH encoding time by 1.2x to 5.7x,
with the largest gains at higher dimensions, and produced byte-identical trained
models. Training and graph writing are measured separately.

## oneDNN FP32 encoding projection

The native provider automatically prefers the existing sequential oneDNN add-on
for FP32 `projectBatch()` when its entry point is available. It executes row-major
SGEMM as `[B,D] * transpose([d,D]) -> [B,d]`, with FP32 inputs, weights and outputs.
The adapter shares the immutable Highway matrix, avoiding a second packed model
copy, and reuses the existing calling-thread native input/output buffers. Its
batch API accepts arbitrary positive counts; no eight-vector cap or padding
controls were introduced. The library's existing `encodeAll()` bound remains 32.
The sequential-runtime build assertion prevents nested oneDNN thread pools.

Single-vector `encode()` retains Highway FP32 projection. The wrapper delegates
query precision selection to the same original projection: automatic mode prefers
INT8, then FP16 with its existing FP32 fallback. Database
encoding never converts to FP16. Code selection and landmark correction stay in
Java.

The optional library retains its existing package/resource name for compatibility.
Both FP32 and FP16 loaders validate ABI 1, handle missing/denied/unsupported
libraries, and retain Highway when optional support is absent. An older add-on
with ABI 1 but without the FP32 symbol also keeps Highway. Unsupported oneDNN
FP32 execution falls back to Highway batch projection; invalid arguments,
allocation failures and unexpected runtime errors propagate. FP16 additionally
checks representable range, and explicit `fp16` selection requires support.
There is no selection flag; the `-Dnative.onednn=true` build packages both
operations.

BLAS kernels can use different FP32 reduction orders for different batch shapes.
The 9-bit reconstruction check therefore uses its existing per-component code
step error bound for worker/batch comparisons as well as singleton comparisons,
and compares the code-independent centered offset within FP16 rounding; the full
1/2/3/4-bit code/header checks remain exact in those tests. Separate tests compare oneDNN output to a double-precision oracle,
cover FP32 values outside FP16 precision/range, caller aliasing and invalid
shapes, batch tails/counts above 32, immutable model copies and concurrent model
switching. A direct ABI test requires successful oneDNN SGEMM, preventing a
Highway fallback from passing as a oneDNN test.

In repeated BenchYAML construction runs, oneDNN FP32 encoding was 1.13x faster
than Highway for MPNet 1M at 2 bits and 1.03x faster for OpenAI-3072 1M at
4 bits, with equivalent graph recall.

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


## Retained-memory accounting

`AsymmetricHashing.ramBytesUsed()` counts the retained model: landmarks and
auxiliary landmark arrays, the Stiefel matrices (including row/outer arrays),
model objects, provider-owned native FP32 backing and its adapter/delegate objects,
optional Highway batch backing,
and a distinct FP16 or INT8 query representation when initialized. Immutable Java sizes
are cached during construction; reading the estimate does not scan matrices or
create them. FP32 query selection and automatic FP16 fallback preserve the
original projection view's identity, so shared FP32 backing is charged once.
Dense row and block matrices use their corresponding storage layouts; other
`RealMatrix` implementations retain a dense storage estimate without calling
`getData()`, which could materialize a matrix copy.

`ASHVectors.ramBytesUsed()` includes its compressor once, its compact code
pages and decoded scale/offset/landmark arrays, and any retained packed or
symmetric-scoring caches. It uses heap array layout estimates, not serialized
code sizes. Code storage is sized when the pages are allocated. Packed-block
totals are charged when a layout is successfully created and include a small
conservative allowance for map bookkeeping. Polling the estimate is
constant-time with respect to vector count, even after packed caches are
created. Vectors passed to the public constructor are copied, so repeated
ordinals sharing one input vector are charged per ordinal. The model arrays must remain immutable after construction, as required by scoring.

Cached fused ASH sources report their owned source object and quantized vector,
including its primitive arrays, without charging the shared compressor. Shared
graph-cache bookkeeping is accounted by `OnDiskGraphIndex`, not by the ASH source.
Callers such as Cassandra add the compressor or full compressed collection
separately. This ownership split avoids charging the same model for every source.

As with PQ, these retained model/collection estimates exclude worker-owned query
buffers, native staging/oneDNN execution scratch, training temporaries and OS file
cache. Search concurrency and training memory require separate budgeting. They
are estimates of retained ownership, not an allocator reservation or a total
process-RSS bound. VM/native-runtime bookkeeping is approximate. Formats,
compression byte budgets and scoring arithmetic are unchanged.

The accounting regression tests derive storage expectations from the allocated
arrays and the running JVM's layout, rather than fixed dataset-specific byte
totals. They cover code widths 1, 2, 3, 4 and 9, empty collections, padded capacities,
multiple landmarks, lazy query/batch/symmetric representations, repeated and
concurrent cache creation, and allocation-free polling. When ASH storage or cache
ownership changes, update its allocation-site totals and the corresponding tests
together. Shared JVM layout and graph bookkeeping remain centralized in their
common implementations.

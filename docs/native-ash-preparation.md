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

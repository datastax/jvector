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
The matrix is shared read-only; temporary storage is invocation-local.

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

All three symbols are optional and looked up independently. Loading an older
compatible library preserves Java projection and symmetric implementations.
Java validates shapes, array bounds and output aliasing before passing pointers
to native code. Native kernels do not allocate, retain pointers or call Java.
Scorer diagnostics identify whether each operation is Highway or Java.

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

# Native ASH projection and symmetric construction scoring

With the native vectorization provider, ASH runs its dense math in Highway
kernels from the native library. The Panama provider keeps its Java SIMD
implementations. Model and graph formats are the same for both providers.

## What runs natively

- **Query projection.** One matrix-vector product per query. Precision is chosen
  as described in [Choosing ASH query-projection precision](ash-projection-precision.md).
- **Symmetric construction scoring.** Code-to-code scores used while building a
  graph, for 1-, 2- and 4-bit codes.
- **Direct-node scoring.** Scores between a projected query and one node's code,
  used for upper layers and direct lookups.
- **Base-vector encoding.** `encodeAll()` projects vectors in batches on the
  caller's `ForkJoinPool`. Encoding always uses FP32.

The native provider copies each model's projection matrix once into native
memory. That copy is counted in `ramBytesUsed()` and released with the model.
It is not serialized.

## Compatibility

Each native entry point is optional. With an older native library that lacks
one, JVector uses the Java implementation for that operation. Native kernels do
not allocate or call back into Java, and Java validates shapes and bounds before
each call.

Different providers can sum in different orders, so FP32 results may differ in
the last bits near a code boundary. Retraining a model is not guaranteed to
produce a byte-identical matrix. Saved models load unchanged.

## Training sample

With one landmark, `AsymmetricHashing.initialize()` trains the mean, PCA and ITQ
on a deterministic uniform sample of `min(N, 20 * D)` vectors, where `D` is the
original dimension. Encoding then reads every vector. Non-finite inputs are
rejected with the offending ordinal. PCA checks are described in
[ASH PCA correctness](ash-pca-native-svd.md).

## oneDNN FP32 encoding projection

When the optional oneDNN add-on is present, batched FP32 encoding uses its
sequential SGEMM and shares the model's native matrix. Without it, or when it
reports the operation unsupported, encoding uses Highway. On four 1M-vector
datasets, batched FP32 encoding cut BenchYAML's ASH encoding time by 1.2x to
5.7x over one-at-a-time projection, with byte-identical trained models;
oneDNN added up to 1.13x over Highway.

## Retained-memory accounting

`ramBytesUsed()` includes native memory, and `nativeBytesUsed()` reports that
part. Per-thread query buffers, native staging, oneDNN scratch, training
temporaries and the OS page cache are not counted; budget search concurrency and
training separately.

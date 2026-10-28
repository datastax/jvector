# DistancesPQ full-scan recall at exact code sizes

`DistancesPQ` supports an accuracy-only mode for comparison with the collected
`DistancesASH` neighborhood full-scan recall. It trains and encodes standard PQ,
prepares one production `PQDecoder.DotProductDecoder` lookup table per query,
scans every encoded base ordinal, and compares its approximate top ten with the
first ten IDs in the matching ground-truth file. It performs no graph search or
exact reranking. Independent queries can run concurrently; the run duration is
not a throughput benchmark.

Enable `jvector.bench.recall=true` and `jvector.bench.recall-only=true`. Queries
are normalized in this mode to match the ASH neighborhood fixture policy. Base
vectors retain their input values. Exact score ties prefer smaller base IDs.
The existing throughput mode and its query policy remain the default.

Set `jvector.pq.subspaces=M` for an exact M-byte code: PQ uses 256 centroids per
subspace, so each subspace requires one byte. PQ already supports uneven
subvector sizes. This setting and the legacy `jvector.pq.mFactor` are mutually
exclusive; absent either option the legacy factor of eight is preserved.

For ASH with C=1, projected dimension D/2, and 2 or 4 bits per projected
coordinate, the packed per-vector budget is `5 + (D/2)*bits/8` bytes. The five
bytes cover the ASH vector header. The corresponding PQ budgets are:

| Dataset dimension | ASH 2-bit / PQ bytes | ASH 4-bit / PQ bytes |
| ---: | ---: | ---: |
| 768 | 101 | 197 |
| 1024 | 133 | 261 |
| 1536 | 197 | 389 |
| 3072 | 389 | 773 |

This matches per-vector code bytes, not total serialized index size. Shared
ASH transforms and PQ codebooks must be reported separately. PQ uses its normal
training policy (up to 128,000 sampled vectors, six clustering rounds, no global
centering, no anisotropic weighting). Centroid initialization is stochastic;
preserve the trained artifact to identify the actual model measured.

Example, with dataset-specific paths and an existing output directory:

```sh
java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -Djvector.vectorization_provider=panama \
  -Djvector.pq.subspaces=101 \
  -Djvector.bench.recall=true -Djvector.bench.recall-only=true \
  -Djvector.bench.recall.k=10 \
  -Djvector.bench.trainingThreads=48 -Djvector.bench.scoringThreads=48 \
  -Djvector.bench.recall.output=/results/mpnet-pq-101.tsv \
  -Djvector.bench.pq.artifact=/results/mpnet-pq-101.pqv \
  -cp "$JVECTOR_CLASSPATH" io.github.jbellis.jvector.example.DistancesPQ \
  /data/base.fvecs /data/query.fvecs /data/ground-truth.ivecs
```

Choose thread counts and JVM memory for the machine and dataset. No query limit
is needed: the default uses all queries. `jvector.bench.maxQueries` can restrict
the query prefix when explicitly desired. Use the same complete base and query
sets as the ASH measurement for a direct comparison.

The log identifies the provider, ADC implementation, normalized-query SHA-256,
shared codebook size, and exact hit numerator/denominator. A small independent
reconstruction check validates ADC scores before the recall scan. The TSV stores
per-query IDs, scores and hits; the optional PQ artifact stores the trained
compressor and encoded base set. Output paths must not exist. No scalar
comparison is implied by the accuracy-only flag: SIMD selection is controlled
by the provider and available vector width.

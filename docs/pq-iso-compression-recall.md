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

For a comparison of learned quantizers, run ASH with
`jvector.ash.optimizer=itq`. This uses the existing PCA/ITQ training defaults:
`min(N,20D)` sampled base vectors, up to 25 iterations, and early stopping.
Keep C=1, d=D/2 and the same bit width when reproducing the table below.
`DistancesASH` defaults to the RANDOM transform for its isolated kernel runs;
those results are an untrained control rather than a trained-ASH accuracy claim.
Use a fresh neighborhood fixture path for ITQ: an existing fixture replays its
stored model, so changing the optimizer property does not retrain that fixture.
Preserve the training log and verify the serialized optimizer and matrix hash.

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

## Measured trained-ASH comparison

JDK 22.0.2+9, Panama SIMD (512-bit), complete base sets and all queries,
C=1, d=D/2, 32-vector neighborhoods, no graph traversal or reranking.
ASH uses the unchanged ITQ training defaults described above. PQ uses its
normal K=256 training policy. Each case retains its trained model and rankings.

| Dataset | ASH bits | Bytes/vector | ASH recall@10 | PQ recall@10 | ASH − PQ (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mpnet-1M | 2 | 101 | 88.011% | 82.040% | +5.971 |
| mpnet-1M | 4 | 197 | 95.569% | 89.478% | +6.091 |
| cohere-1M | 2 | 133 | 77.200% | 74.416% | +2.784 |
| cohere-1M | 4 | 261 | 91.421% | 84.496% | +6.925 |
| ada002-1M | 2 | 197 | 86.449% | 66.772% | +19.677 |
| ada002-1M | 4 | 389 | 94.217% | 81.731% | +12.486 |
| openai-3072-1M | 2 | 389 | 90.570% | 85.600% | +4.970 |
| openai-3072-1M | 4 | 773 | 96.170% | 92.130% | +4.040 |

MPNet, Cohere and Ada002 use 10,000 queries each; OpenAI-3072 uses 1,000.
Cleaned base counts are respectively 999,812; 1,000,000; 982,790; and 999,000.
All eight cases favor trained ASH in this measurement. Shared model storage is
excluded from the matched code sizes: ASH models are approximately 1.18/2.10/4.72/18.89 MB
for these datasets, versus PQ's 0.79/1.05/1.57/3.15 MB. This is not equal total
serialized storage and does not compare with OPQ or Faiss 4-bit FastScan.

The original collected ASH results used RANDOM transforms and remain controls.
For example, Cohere RANDOM recall was 59.898%/65.499%; trained recall is
77.200%/91.421%. No model or parameter was selected using recall. PQ has one
stochastic fresh fit per case, saved for reproduction.

Source for the measured runs: `c136c5411a4fa207823d0b8f9cdc2ae780bcf6cc`.
Evidence on the benchmark host: `/home/ted_willke/benchmark-runs/pq-iso-ash-full-scan-recall-jdk22-20261028`.
Its `summary.csv` includes exact hit numerators, denominators and model hashes;
`validation.json` records independent ground-truth/ranking checks. All prepared
query hashes agree bit for bit. Each trained ASH fixture declares ITQ and has a
different matrix from its RANDOM counterpart. Logs and rankings are retained
for both methods, with PQ artifacts in `artifacts/` and ASH fixtures in
`ash-itq/fixtures/`. Source, protocol and outputs are also attached to the local
comparison report. These are accuracy runs with 48 independent query workers;
no throughput conclusion is drawn from their durations.

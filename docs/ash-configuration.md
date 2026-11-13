# ASH configuration and defaults

ASH has independent settings for stored codes, construction, query projection,
and query execution. This reference covers library parameters and BenchYAML
controls. The [BenchYAML Guide](benchyaml-guide.md) defines the index YAML
fields. Recipes specify their own graph and compression values.

Set JVM properties before the affected model, provider or executor is initialized.
Query views and pools are cached; changing a property later does not reconfigure
existing objects. CPU-count properties do not pin threads.

## Stored codes and training

BenchYAML puts these under `compression[].parameters` with `type: ASH`.
Construction compression controls graph scoring. Search compression selects
the codes stored with a fused ASH graph.

| Parameter | Default / requirement | Optional values and definition |
| --- | --- | --- |
| `optimizer` | Required | `ITQ` (`1`): iterative-quantization transform. `RANDOM` (`2`): random transform. |
| `encodedBits` | Exactly one of this and `compressionRatio` | Total stored bits including the 40-bit / five-byte header. The positive payload `encodedBits - 40` must be divisible by `bitsPerDimension`. |
| `compressionRatio` | Alternative to `encodedBits` | BenchYAML accepts `8`, `16`, `32`, `64`, `128`, `256`. Resolves a payload near the requested FP32-to-code ratio, with a minimum 64-bit payload and whole projected dimensions; inspect the printed resolved size. |
| `bitsPerDimension` (`b`) | `1` | Integer `1` through `9`, all supported by fused ASH. `1`, `2` and `4` have the most optimized kernels (see below). Stored-code precision is independent of query-projection arithmetic. |
| `landmarkCount` (`C`) | `1` | Integer `1` through `256` for the quantizer. Symmetric graph-construction scoring currently requires `C=1` and dot-product similarity. |
| Projected dimension (`d`) | Derived, not a separate YAML field | `d = (encodedBits - 40) / b`, with `1 <= d <= D` for input dimension `D`. Increasing `d` or `b` increases code size. |

For an exact payload, use `encodedBits = 40 + d*b`. For example, a 256-D
input projected to 128 dimensions at four bits uses `encodedBits: 552`.
Include the header when comparing memory budgets with other quantizers.

### Choosing the projected dimension for speed

For the fastest construction and search, use `b` = 1, 2 or 4 and make `d` a
multiple of 64. The native kernels then process whole 64-bit words and whole
SIMD vectors, with no leftover dimensions:

- Construction compares stored codes with each other. At 1 bit, the kernel
  works in words of 64 dimensions. At 2 bits, it works in words of 32
  dimensions. At 4 bits, it works in vectors of up to 128 dimensions.
- Search builds per-query lookup tables in 16-byte chunks of code and projects
  queries 32 output dimensions at a time.

Other values of `d` are correct and handled in one extra masked step, so the
penalty is small. In the measurements below, a 2-bit code at `d = 380` costs
about 20-30% more per construction comparison than at `d = 384`, and search
QPS differs by about 1.5%. When a byte budget lands just short of an aligned
`d`, prefer the aligned value. For example, with a 384-D input and a
100-byte budget, `d = 384` at 2 bits (101 bytes, no dimension reduction) is a
better choice than `d = 380` (100 bytes).

The other widths, `b` = 3 and 5 through 9, store each dimension as an unsigned
`b`-bit field. Search scores their graph neighborhoods with the projected query
rounded to INT8 and integer dot products (VNNI on x86 processors that have it),
so make `d` a multiple of 8. Their other asymmetric scoring uses the same
4-bit query tables as 2 and 4 bits. Construction still compares their codes
with a portable per-dimension path, so building is slower than at 1, 2 or 4
bits; native code-to-code kernels for these widths are left for future work.

Construction comparison cost, in ns per code pair, on one pinned core with
the native provider, on an Intel Xeon 6985P-C and on an AWS m6id.4xlarge
(Ice Lake).

| `b` | `d` | Xeon 6985P-C | m6id.4xlarge |
| --- | --- | --- | --- |
| 2 | 192 | 9.3 | 16.7 |
| 2 | 380 | 16.6 | 30.0 |
| 2 | 384 | 13.8 | 22.5 |
| 4 | 190 | 15.2 | 24.0 |
| 4 | 192 | 14.5 | 24.1 |
| 4 | 384 | 18.4 | 28.5 |

Training and base encoding retain their FP32 projection path; some training
statistics and factorizations use FP64 internally. Query precision does not
change the trained model, base encoding, stored-code layout or index format.

## Graph construction

BenchYAML selects construction scoring through `compression[].scoring` on
an ASH construction entry. Omission uses the JVM policy, which defaults to
`symmetric-all`. For example, this JVM property selects mixed scoring:

```text
-Djvector.bench.ashConstruction=symmetric
```

| Property value | Insertion search | Existing-node / diversity scoring |
| --- | --- | --- |
| `symmetric-all` **(default)** | Encoded vector against encoded vectors; no dense insertion-query projection | Symmetric encoded scoring |
| `symmetric` | Original vector used as an asymmetric query | Symmetric encoded scoring |
| `symmetric-scalar` | Original vector used as an asymmetric query | Scalar symmetric scoring; diagnostic control |
| `asymmetric` | Original vector used as an asymmetric query | Original vector used as an asymmetric query; diagnostic control |
| `asymmetric-cached` | Original vector used as an asymmetric query | Cached asymmetric query state per node; diagnostic control with extra memory use |

`jvector.bench.ashConstruction` controls **BenchYAML/Grid**, not every library
builder. For library integration, obtain a provider using
`BuildScoreProvider.ashBuildScoreProvider(similarity, encoded)`. Use
`builder.addGraphNode(node, provider.searchProviderFor(node))` for fully symmetric
insertion. The overload taking an original vector retains asymmetric query
preparation. Both policies use the same stored ASH vectors; application queries
continue to use asymmetric scoring.

Construction policy can change the graph, recall and QPS. Use fresh indexes
when comparing modes: the graph cache key does not include this JVM property,
so changing it does not invalidate an existing cached graph.

### In-memory code layout

`ASHVectors` keeps one copy of the codes, in compact pages. Each vector's code
body sits back to back with the others: sign words at 1 bit, sign-magnitude
projection codes at 2 and 4 bits, and offset-binary fields at the other widths
(dimension `j` is the unsigned field at bits `[j*b, (j+1)*b)`, with value
`field - (2^b - 1)/2`). Every stored offset includes the landmark term
`-scale * <A mu_c, code>`, so asymmetric scoring needs only `Aq`. Bodies have
no per-vector object, array or padding. Scale, offset and landmark are decoded
once into three parallel arrays. Every scorer, symmetric and
asymmetric, reads the codes in place, and the encoder writes straight into the
pages.

Retained memory per vector is the serialized size plus 4 bytes, because the two
binary16 header fields are held as floats: for example 101 bytes on disk and
105 bytes in memory for 768-D at 32x. Pages hold up to 256 KiB, which keeps
them below G1's smallest humongous-object threshold so the collector never
rounds them up to whole regions. `ASHVectors.get(i)` returns an allocated copy;
`copyTo(i, dest)` fills reusable storage. Optional caches built for block
scoring (`blockScorerFor` with a block size) hold a second, interleaved copy and
are charged in `ramBytesUsed()` when created; graph construction and search do
not build them.

## Query projection and scoring

| JVM property | Default | Optional values and definition |
| --- | --- | --- |
| `jvector.ash.queryProjection` | `auto` | `auto`: INT8 where an integer dot-product backend exists, then optimized FP16, falling back to FP32. `fp32`: FP32. `fp16`: require optimized FP16 with FP32 accumulation/output. `int8`: require INT8 projection with integer accumulation and FP32 output. Explicit unsupported precisions fail. Default for models without query options set in code. |
| `jvector.ash.singleKernel` | `auto` | `auto`, `scalar`, `simd`; single-vector ASH scoring kernels, subject to provider/format support. |
| `jvector.ash.blockKernel` | `auto` | `auto`, `scalar`, `simd`; blocked/fused ASH scoring kernels, subject to provider/format support. |
| `jvector.ash.queryBatchSize` | `8` | Positive integer below `Integer.MAX_VALUE`; maximum opportunistic batch for model-owned synchronous query preparation; `1` disables combining. Separate from the queued batch limit below. Default for models without query options set in code. |

To choose these per model in code, call `setQueryOptions` before the model
projects its first query, for example right after opening an index. Later calls
throw `IllegalStateException`, because the prepared projection is fixed for the
model's lifetime.

```java
var ash = ((FusedASH) index.getFeatures().get(FeatureId.FUSED_ASH)).getASH();
ash.setQueryOptions(ASHQueryProjection.AUTO, 1);  // precision, maximum batch
```

See [Choosing ASH query-projection precision](ash-projection-precision.md) for
processor guidance and FP16/INT8 requirements. Under `symmetric-all`, query
precision affects application queries, not insertion search. Under mixed or
asymmetric construction, it can affect insertion too.

### BenchYAML precision selection

Set `queryProjection` in the `parameters` map of an ASH compression entry.
See [ASH query parameters](benchyaml-guide.md#ash-query-parameters) for values
and JVM precedence.
ASH CSV rows record `ash_query_projection_requested` and
`ash_query_projection_effective`; the latter is read from the prepared view,
with `fp16+fp32` indicating observed automatic FP32 range fallback. Non-ASH
rows have blank precision columns; an unused projection has blank effective
precision. See the [precision guide](ash-projection-precision.md#benchyaml-runs).

## Providers, CPU budgets and serialization

| Setting | Default | Optional values and definition |
| --- | --- | --- |
| `jvector.vectorization_provider` | Automatic selection | `native`, `panama`, `default` (scalar). Supported Java 22+ runtimes prefer native, with existing fallbacks. Explicit `native` requires successful loading. |
| `jvector.enable_native_vectorization` | `true` | `true` / `false`; automatic native selection. Explicit provider selection overrides it. Deprecated `jvector.experimental.enable_native_vectorization` is a lower-precedence alias. |
| `jvector.physical_core_count` | OS topology detection bounded by available processors; fallback heuristic | Positive thread budget for JVector's bounded pool and queued workers. Construction and queries can use separate budgets/processes; see [Thread budgets](#thread-budgets). |
| `-XX:ActiveProcessorCount` | JVM/container detection | Positive processor count exposed to the JVM; coordinate with allocated physical cores. Does not set affinity. |
| `--add-modules=jdk.incubator.vector` | Not added by the application | Enables the Vector API module needed by SIMD providers on supported runtimes. |
| `--enable-native-access=ALL-UNNAMED` | JVM native-access policy | Native access for classpath deployments. Named-module deployments should enable the relevant module. |
| `-Xms`, `-Xmx` | JVM ergonomics | Initial / maximum heap; account for resident input, model and construction memory. |
| `-ea`, `-da` | Assertions normally disabled | Enable / disable Java assertions; independent of numerical precision. |

### Thread budgets

Graph insertion mostly waits on memory, so a second hardware thread per core
fills those stalls: give construction every hardware thread, for example
`-Djvector.physical_core_count=16 -XX:ActiveProcessorCount=16` on 8 cores with
two threads each, or pass a pool of that size to the builder. Builds of millions
of vectors gain the most; recall and the built graph's search speed do not
change. Compute-bound search runs best with one worker per physical core. Search
that waits on storage needs more queries in flight than cores; see
[Building and serving larger-than-memory indexes](larger-than-memory-indexes.md#serving-queries).

### BLAS threads

ASH training is the only BLAS and LAPACK user. JVector sets the native BLAS
thread count for its own calls, so a host does not need to configure MKL or
OpenBLAS: training uses one BLAS thread, or as many as the executor passed to
`AsymmetricHashing.initialize(..., ForkJoinPool executor)`. Queries never call
BLAS. With the native provider, the count is applied per calling thread through
MKL's `MKL_Set_Num_Threads_Local` or OpenBLAS's `openblas_set_num_threads_local`;
OpenBLAS releases before 0.3.27 only have a process-wide setting, which JVector
sets for the duration of training. Other providers leave BLAS threading to
`MKL_NUM_THREADS`, `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS`.
`io.github.jbellis.jvector.util.BlasThreads` applies the same limit to a host's
own BLAS work.

BenchYAML trains in its construction pool, so BLAS uses `construction.workers`
threads (all physical cores by default). Set `MKL_DYNAMIC=FALSE` so MKL uses the
count it is given. The ASH versus PQ study scripts also set the thread variables,
to one for queries and the core budget for construction, as in the published
runs. OS affinity (`numactl --physcpubind`) and memory placement (`--membind`,
`--interleave`) are deployment choices.

## Graph and search parameters in BenchYAML

Use the [BenchYAML Guide](benchyaml-guide.md) for index type, graph parameters,
writer selection, compression, and reranking.

## Query delivery and measurement controls

Set BenchYAML query delivery in the [search section](benchyaml-guide.md#search)
(`queryExecution`, `requestsPerWorker`) and the ASH batch limit in the
[ASH search parameters](benchyaml-guide.md#ash-query-parameters). Database
integrations select their own admission policy and executors. See
[Integrating ASH queries into a database](ash-query-integration.md).

| Setting | Default | Optional values / definition |
| --- | --- | --- |
| `jvector.bench.queryThreads` | `0` | Direct caller count: `0` uses common pool, `1` is serial, larger positive values use a dedicated caller pool. Does not set queued worker count. |
| `jvector.bench.verifyConcurrentResults` | `false` | `true` / `false`; validate concurrent results against reference results. |
| `jvector.bench.runConfig` | `jvector-examples/yaml-configs/run-config.yml` | Run YAML path for measured benchmarks, console and CSV fields. |
| `jvector.bench.sourceRef` | Unset | Source identity for provenance; no numerical effect. |

Run YAML `benchmarks`, `console` and `logging` select computed/reported throughput,
recall, latency and visit counts, and `measurement` sets warmup, timed passes and
the page cache state; see [Run configuration](benchyaml-guide.md#run-configuration).
Construction separately reports quantizer training, encoding, NVQ training and
graph/write. The throughput benchmark can also report per-query response times,
page reads per query and, for queued execution, mean batch size. Unloaded latency and loaded
response time describe different measurements. Independent build/query JVM
repetitions are launch choices, not model defaults; preserve individual CSVs.

The ASH versus PQ case script (`run-ash-pq-benchmark.sh`) accepts `JVECTOR_BENCH_DATA_DIR` (required local input
directory), `JVECTOR_BENCH_CATALOG` (`catalog-1m.yml`; optionally
`catalog-large.yml`), `JVECTOR_BENCH_HEAP` (`128g`), and
`JVECTOR_BENCH_NUMA_POLICY` (`bind:0`; alternatively `interleave:<nodes>`).
These defaults belong to the script, not the library. See the
[Reproducing ASH versus PQ](ash-pq-benchmark.md) for launch and artifact details.

## Diagnostics

`jvector.ash.debugSymmetric` defaults to `false`. Its optional construction
check is bounded by `jvector.ash.debugSymmetric.pairs` (`2000`),
`jvector.ash.debugSymmetric.subsetN` (`50000`) and
`jvector.ash.debugSymmetric.topK` (`100`). Keep diagnostics off for normal timing.

Implementation references: [YAML compression parsing](../jvector-examples/src/main/java/io/github/jbellis/jvector/example/yaml/Compression.java),
[ASH model](../jvector-base/src/main/java/io/github/jbellis/jvector/quantization/AsymmetricHashing.java),
[construction modes](../jvector-examples/src/main/java/io/github/jbellis/jvector/example/diagnostics/ASHBuildScoreExperiment.java),
and [run configuration](../jvector-examples/src/main/java/io/github/jbellis/jvector/example/yaml/RunConfig.java).

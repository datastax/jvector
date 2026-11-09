<!--
Copyright DataStax, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

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
| `bitsPerDimension` (`b`) | `1` | Integer `1` through `9` in core ASH. Fused ASH supports `1`, `2`, `4`; optimized multibit formats use `2` or `4`. Stored-code precision is independent of query-projection arithmetic. |
| `landmarkCount` (`C`) | `1` | Integer `1` through `256` for the quantizer. Symmetric graph-construction scoring currently requires `C=1` and dot-product similarity. |
| Projected dimension (`d`) | Derived, not a separate YAML field | `d = (encodedBits - 40) / b`, with `1 <= d <= D` for input dimension `D`. Increasing `d` or `b` increases code size. |

For an exact payload, use `encodedBits = 40 + d*b`. For example, a 256-D
input projected to 128 dimensions at four bits uses `encodedBits: 552`.
Include the header when comparing memory budgets with other quantizers.

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
body (sign words, projection code or extra bits) sits back to back with the
others, with no per-vector object, array or padding. Scale, offset and landmark
are decoded once into three parallel arrays. Every scorer, symmetric and
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
| `jvector.ash.queryProjection` | `auto` | `auto`: prefer optimized FP16, falling back to FP32. `fp32`: FP32. `fp16`: require optimized FP16 with FP32 accumulation/output. `int8`: require INT8 projection with integer accumulation and FP32 output. Explicit unsupported precisions fail; `auto` never selects INT8. |
| `jvector.ash.singleKernel` | `auto` | `auto`, `scalar`, `simd`; single-vector ASH scoring kernels, subject to provider/format support. |
| `jvector.ash.blockKernel` | `auto` | `auto`, `scalar`, `simd`; blocked/fused ASH scoring kernels, subject to provider/format support. |
| `jvector.ash.queryBatchSize` | `8` | Positive integer below `Integer.MAX_VALUE`; maximum opportunistic batch for model-owned synchronous query preparation. Separate from the queued batch limit below. |

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
| `jvector.physical_core_count` | OS topology detection bounded by available processors; fallback heuristic | Positive physical-core budget for JVector's bounded pool and queued workers. Construction and queries can use separate budgets/processes. |
| `-XX:ActiveProcessorCount` | JVM/container detection | Positive processor count exposed to the JVM; coordinate with allocated physical cores. Does not set affinity. |
| `--add-modules=jdk.incubator.vector` | Not added by the application | Enables the Vector API module needed by SIMD providers on supported runtimes. |
| `--enable-native-access=ALL-UNNAMED` | JVM native-access policy | Native access for classpath deployments. Named-module deployments should enable the relevant module. |
| `-Xms`, `-Xmx` | JVM ergonomics | Initial / maximum heap; account for resident input, model and construction memory. |
| `-ea`, `-da` | Assertions normally disabled | Enable / disable Java assertions; independent of numerical precision. |

`MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`GOTO_NUM_THREADS` are native-runtime environment controls, not ASH model
parameters. Avoid nested native query pools. The ASH versus PQ study scripts set them
to one for queries and the core budget for construction; it also sets
`MKL_DYNAMIC=FALSE` and `ONEDNN_VERBOSE=0`. OS affinity
(`numactl --physcpubind`) and memory placement (`--membind`, `--interleave`)
are deployment choices.

## Graph and search parameters in BenchYAML

Use the [BenchYAML Guide](benchyaml-guide.md) for index type, graph parameters,
writer selection, compression, and reranking.

## Query delivery and measurement controls

Set BenchYAML query delivery and batch limits in the
[ASH search parameters](benchyaml-guide.md#ash-query-parameters). Database
integrations select their own admission policy and executors. See
[Integrating ASH queries into a database](ash-query-integration.md).

| Setting | Default | Optional values / definition |
| --- | --- | --- |
| `jvector.bench.queryThreads` | `0` | Direct caller count: `0` uses common pool, `1` is serial, larger positive values use a dedicated caller pool. Does not set queued worker count. |
| `jvector.bench.verifyConcurrentResults` | `false` | `true` / `false`; validate concurrent results against reference results. |
| `jvector.bench.throughputMinSeconds` | `1.0` | Minimum throughput sample duration in seconds. |
| `jvector.bench.runConfig` | `jvector-examples/yaml-configs/run-config.yml` | Run YAML path for measured benchmarks, console and CSV fields. |
| `jvector.bench.sourceRef` | Unset | Source identity for provenance; no numerical effect. |

Run YAML `benchmarks`, `console` and `logging` select computed/reported throughput,
recall, latency and visit counts. Construction separately reports quantizer
training, encoding, NVQ training and graph/write. Queued execution can also
report loaded response times and mean batch size. Unloaded latency and loaded
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

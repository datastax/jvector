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

# BenchYAML Guide

BenchYAML uses dataset configuration to build an index and measure search.
Each case produces one index. Construction scoring determines the graph
topology, search scoring determines how queries traverse that graph, and
reranking optionally rescores the candidates.

## Configuration files

- [Index parameters](../jvector-examples/yaml-configs/index-parameters) describe
  construction and search for a dataset. A dataset-specific file takes
  precedence over `default.yml`.
- [Dataset catalogs](../jvector-examples/yaml-configs/dataset-catalogs) locate
  the base vectors, query vectors, and ground truth.
  [Dataset metadata](../jvector-examples/yaml-configs/dataset-metadata.yml)
  supplies the similarity function. Vector types come from the dataset.
- [Run configuration](../jvector-examples/yaml-configs/run-config.yml) selects
  which measurements to compute and which to display or log.

See [How to run benchmarks](benchmarking.md) for dataset setup and commands.

## Construction

Choose `inMemory` or `onDisk` for the case. To measure both index forms, use
separate cases with matching graph settings. The graph settings and
`fusedGraph` are lists of sweep values. `useSavedIndexIfExists` is a scalar.
Each `compression` list entry has a `type` and its own settings.

| Field | Values | Description |
| --- | --- | --- |
| `type` | `inMemory`, `onDisk` | `inMemory` keeps the graph in memory for search. `onDisk` serializes the completed graph and reopens it for search. |
| `writerMode` | `sequential`, `parallel` | Selects the writer after graph cleanup for `onDisk`. Defaults to `parallel`. |
| `useSavedIndexIfExists` | `Yes`, `No` | Controls reuse of a compatible `onDisk` index. A cache hit skips fresh construction and serialization measurements. |
| `outDegree` | Positive integers | Maximum final neighborhood degree, also called M. |
| `efConstruction` | Positive integers | Search width used to find candidates during insertion. |
| `neighborOverflow` | Finite numbers ≥ 1.0 | Temporary neighborhood capacity relative to final degree. |
| `addHierarchy` | `Yes`, `No` | Builds upper graph layers. |
| `refineFinalGraph` | `Yes`, `No` | Refines the graph during final cleanup. |
| `fusedGraph` | `Yes`, `No` | Applies to `onDisk`. `Yes` stores ASH or PQ neighbor codes alongside the graph records. `search.compression` selects the codec. |
| `compression[].type` | `None`, `ASH`, `PQ`, `BQ` | Selects the representation for construction scoring. `None` uses source vectors. |
| `compression[].parameters` | [Quantizer parameters](#quantizer-parameters) | Supplies training and encoding settings for this entry. |
| `compression[].scoring` | `symmetric`, `asymmetric`, `mixed` | Optional for ASH construction. `symmetric` scores encoded vectors during insertion and pruning. `asymmetric` scores source queries against codes for both operations. `mixed` uses asymmetric insertion and symmetric pruning. |
| `reranking.type` | `None`, `FP32`, `NVQ` | `FP32` rescores candidates with source vectors. `NVQ` rescores them with NVQ codes. `None` or omission skips reranking where the scorer and layout allow it. |

ASH construction uses symmetric scoring by default. When `scoring` is omitted,
the configured ASH construction policy applies. See
[ASH configuration and defaults](ash-configuration.md#graph-construction)
for the JVM setting.

## Search

Search uses the index selected by `construction.type`. Changes to stored features
define separate index cases. Search sweeps such as `topKOverquery` use the index
for their case.

| Field | Values | Description |
| --- | --- | --- |
| `topKOverquery` | Map of positive integer K to lists of finite multipliers ≥ 1 | Required for search. The integer part of K × multiplier is the candidate budget. Return up to K results, with a second scoring pass when configured. |
| `useSearchPruning` | List of `Yes` and `No` values | Enables traversal pruning for supported scorers and similarity functions. |
| `compression[].type` | `None`, `ASH`, `PQ`, `BQ` | Selects the traversal representation. `None` scores source vectors. |
| `compression[].parameters` | [Quantizer parameters](#quantizer-parameters) | Supplies encoding settings for traversal. Reuse matching construction models and codes where possible. |

For a fused disk case, Grid writes the selected search codes over the completed
topology. The same ASH model can serve construction and search when their
quantizer settings match.

### ASH query parameters

Place `queryProjection` in the `parameters` map of an ASH compression entry.
Place `queryExecution` and `maxBatchSize` in that map on an ASH **search** entry.

| Field | Values | Description |
| --- | --- | --- |
| `queryProjection` | `auto`, `fp32`, `fp16`, `int8` | Optional. `auto` is the default and selects FP16 where supported, with FP32 fallback. `fp32` selects FP32 projection. `fp16` and `int8` require their selected paths. The mode applies to ASH search and asymmetric construction scoring. All ASH entries in one JVM must resolve to one mode. The `jvector.ash.queryProjection` JVM property overrides YAML. |
| `queryExecution` | `direct`, `queued` | `direct` runs searches on the calling threads. `queued` sends searches to workers and batches query preparation when supported. Defaults to `direct`. |
| `maxBatchSize` | Positive integer | Caps the number of queries prepared together in queued mode. Defaults to 8. The value must fit the executor limits at the selected worker count. |

FP16 projection uses FP32 accumulation and output. INT8 projection uses integer
accumulation and FP32 output. Fully symmetric construction scores encoded
vectors directly. See the [ASH precision guide](ash-projection-precision.md)
for provider and hardware requirements.

### Validate the index and scorer

Grid validates the selected scorer, layout, similarity function, and precision
before training or construction. Disk writer, cache, and fused layout settings
apply to `onDisk` cases. Fused disk search requires ASH or PQ search compression.

With `reranking.type: None` or an omitted reranker, search returns up to K results
ordered by traversal score, including compressed scores where supported. If the
selected scorer or layout requires a reranker, Grid throws an exception during
configuration validation. The fused ASH and PQ scorer setup requires an FP32 or
NVQ scoring feature.

Grid rejects settings that do not apply to the case and reranking that repeats
the traversal scorer with the same vectors and arithmetic. Explicit precision
requests require support on the selected provider and machine.

## Quantizer parameters

The fields below belong to the `parameters` map of their compression entry.
D is the dataset dimension.

| Quantizer and field | Values | Description |
| --- | --- | --- |
| ASH `optimizer` | `ITQ`, `RANDOM` | Training transform. Required for a new model. |
| ASH `compressionRatio` | 8, 16, 32, 64, 128, 256 | Nominal FP32-to-code ratio. The parser rounds to whole projected dimensions and uses a minimum payload. Specify either `compressionRatio` or `encodedBits`. |
| ASH `encodedBits` | Positive integer bit budget | Includes a 40-bit header. The payload must be divisible by `bitsPerDimension` and yield 1 through D projected dimensions. |
| ASH `bitsPerDimension` | 1–9 in core ASH. 1, 2, or 4 for fused ASH. | Bits per projected dimension in the stored codes. Defaults to 1. |
| ASH `landmarkCount` | 1–256 | Number of landmarks. Defaults to 1. ASH graph scoring uses dot-product similarity. Symmetric and mixed construction require one landmark. |
| PQ `m` | Integer from 1 through D | Number of subspaces. Specify either `m` or `mFactor`. |
| PQ `mFactor` | Positive integer yielding a valid `m` | Chooses the subspace count by integer division of D by `mFactor`. |
| PQ `k` | Integer from 1 through 256 | Centroids per subspace. Defaults to 256. Training requires at least k vectors. Fused PQ requires 256. |
| PQ `centerData` | `"Yes"`, `"No"` | Centers data before encoding. Omission enables centering for Euclidean similarity only. Quote these values in the YAML parameter map. |
| PQ `anisotropicThreshold` | −1.0 ≤ value < 1.0 | Controls anisotropic training on unit-normalized input. The default −1 disables anisotropic weighting. |
| BQ parameters | Omitted | Uses sign codes and Hamming scoring to approximate cosine similarity. |

NVQ reranking uses the built-in encoder settings. BenchYAML chooses the
subvector count and uses eight bits per dimension.

## Example index parameters

These sections describe source-vector graph construction followed by fused ASH
disk search with NVQ reranking on a 768-dimensional dot-product dataset.

```yaml
construction:
  type: onDisk
  writerMode: parallel
  useSavedIndexIfExists: No
  outDegree: [32]
  efConstruction: [100]
  neighborOverflow: [1.2]
  addHierarchy: [Yes]
  refineFinalGraph: [Yes]
  fusedGraph: [Yes]
  compression:
    - type: None
  reranking:
    type: NVQ

search:
  topKOverquery:
    10: [1.0, 2.0]
  useSearchPruning: [No]
  compression:
    - type: ASH
      parameters:
        optimizer: ITQ
        encodedBits: 808
        bitsPerDimension: 2
        landmarkCount: 1
        queryProjection: fp16
        queryExecution: queued
        maxBatchSize: 8
```

For a source-vector memory case, select `inMemory`, use `None` compression in
both sections, and omit the disk settings. Use `reranking.type: None` to
measure search with traversal scores alone.

## Related guides

- [How to run benchmarks](benchmarking.md) covers dataset setup and invocation.
- [Fused PQ and ASH comparison](ash-pq-benchmark.md) specifies the matched
  compression budgets, machine setup, and measurement procedure.
- [ASH configuration and defaults](ash-configuration.md) covers model and
  runtime settings.
- [Deferred batch serialization](deferred-batch-writer.md) explains writer
  batching and feature production.

# BenchYAML Guide

BenchYAML uses dataset configuration to build an index and measure search.
Each case produces one index. Construction scoring determines the graph
topology, search scoring determines how queries traverse that graph, and
reranking optionally rescores the candidates.

## Configuration files

- [Index parameters](../jvector-examples/yaml-configs/index-parameters) describe
  construction and search for a dataset. Running BenchYAML with a dataset name
  loads `index-parameters/<dataset>.yml`, named exactly like the dataset and
  declaring it, or `default.yml` when the dataset has no file. Subdirectories
  hold [example pipelines](../jvector-examples/yaml-configs/index-parameters/README.md),
  such as in-memory graphs and in-memory builds flushed to fused on-disk graphs,
  and published studies. Run them by passing their paths to BenchYAML.
- [Dataset catalogs](../jvector-examples/yaml-configs/dataset-catalogs) locate
  the base vectors, query vectors, and ground truth.
  [Dataset metadata](../jvector-examples/yaml-configs/dataset-metadata.yml)
  supplies the similarity function. Vector types come from the dataset.
- [Run configuration](../jvector-examples/yaml-configs/run-config.yml) selects
  which measurements to compute, how to measure them, and which to display or
  log; see [Run configuration](#run-configuration).

Each index-parameter file sets `yamlSchemaVersion: 3`, the current
`onDiskIndexVersion`, and its `dataset` name. An optional top-level
`useDatasetPrefetching: true` streams base vectors in order during construction
instead of loading them all into memory; see
[Batched input vector access](batched-vector-access.md#dataset-streaming-and-caches).

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
| `pageAlignment` | Power of two from 512 to 1048576 | Optional, `onDisk` only. Places base-layer node records so none crosses a page boundary of this size, typically `4096`; see [Page-aligned records](larger-than-memory-indexes.md#page-aligned-records). Omitted, records are packed back to back. Part of the saved-index identity. |
| `useSavedIndexIfExists` | `Yes`, `No` | Controls reuse of a compatible `onDisk` index. `Yes` saves each built index in the working directory's `index_cache` and reuses it on later runs; a cache hit skips fresh construction and serialization measurements. Saved indexes are never deleted automatically, so starting-point configurations use `No`; recipes that query an index from a later JVM, such as the ASH/PQ comparison, use `Yes`. |
| `workers` | Positive integer or `max` | Optional. Threads for quantizer training, encoding, graph insertion and the parallel writer. `max` is the platform's physical core count, which is also the default. |
| `outDegree` | Positive integers | Maximum final neighborhood degree, also called M. |
| `efConstruction` | Positive integers | Search width used to find candidates during insertion. |
| `neighborOverflow` | Finite numbers ≥ 1.0 | Temporary neighborhood capacity relative to final degree. Java float suffixes such as `1.2f` are accepted. |
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

For `onDisk` builds, three construction metrics split the build: `graph_build_time_s`
covers graph construction and cleanup, `flush_time_s` covers writing the completed
graph and its features, and `reopen_time_s` covers reopening the written files.
Graph build plus flush equals `index_build_time_s`; quantizer training and encoding
are reported separately under `index_quant_time_s`. Select them under
`metrics.construction` in the run YAML. A cached index reports none of them.

## Search

Search uses the index selected by `construction.type`. Changes to stored features
define separate index cases. Search sweeps such as `topKOverquery` use the index
for their case. When CSV logging is enabled, the row also records the index
type, writer mode, reranker, ASH construction mode, and query delivery settings
including the number of callers or workers. A run with mixed delivery policies
reports thread counts per case in the CSV.

| Field | Values | Description |
| --- | --- | --- |
| `topKOverquery` | Map of positive integer K to lists of finite multipliers ≥ 1 | Required for search. The integer part of K × multiplier is the candidate budget. Return up to K results, with a second scoring pass when configured. |
| `useSearchPruning` | List of `Yes` and `No` values | Enables traversal pruning for supported scorers and similarity functions. |
| `workers` | List of positive integers or `max` | Optional sweep. Direct search uses this many calling threads; queued search uses this many workers to prepare and execute queries. Each value queries the same index, so `[1, 8]` reports 1 and 8 workers from one build. Omitted, the JVM-wide settings apply. |
| `queryExecution` | `direct`, `queued` | `direct` runs each search on a calling thread. `queued` sends searches to a fixed pool of workers through a bounded queue and, for ASH, prepares waiting queries together. Applies to every compression entry. Defaults to `direct`. |
| `requestsPerWorker` | Positive integer | Optional, `queued` only. Queries admitted per worker, counting the one it is searching; the queue holds this many times the worker count. Defaults to the larger of 8 and the ASH `maxBatchSize`. Use `1` when search waits on storage, so queries do not queue behind busy workers. |
| `compression[].type` | `None`, `ASH`, `PQ`, `BQ` | Selects the traversal representation. `None` scores source vectors. |
| `compression[].parameters` | [Quantizer parameters](#quantizer-parameters) | Supplies encoding settings for traversal. Reuse matching construction models and codes where possible. |

For a fused disk case, Grid writes the selected search codes over the completed
topology. Matching ASH or PQ settings reuse the construction model and codes.
With a non-fused layout, compressed traversal uses a separate in-memory code
store. With source-vector disk search and no reranker, the inline source
vectors are the sole vector feature.

### ASH query parameters

Place `queryProjection` in the `parameters` map of an ASH compression entry.
Place `maxBatchSize` in that map on an ASH **search** entry; it requires
`search.queryExecution: queued`.

| Field | Values | Description |
| --- | --- | --- |
| `queryProjection` | `auto`, `fp32`, `fp16`, `int8` | Optional. `auto` is the default and selects INT8 where an integer dot-product backend exists, then FP16 where supported, with FP32 fallback. `fp32` selects FP32 projection. `fp16` and `int8` require their selected paths. The mode applies to ASH search and asymmetric construction scoring. Explicit ASH settings in one JVM must agree. Entries that omit it inherit the JVM mode, which defaults to `auto`. The `jvector.ash.queryProjection` JVM property overrides YAML. |
| `maxBatchSize` | Positive integer | Caps the number of queued queries whose projections are prepared together. Defaults to 8. Batching saves CPU when search is compute-bound; when search waits on storage, `1` avoids holding queries for a batch. |

FP16 projection uses FP32 accumulation and output. INT8 projection uses integer
accumulation and FP32 output. Fully symmetric construction scores encoded
vectors directly. See the [ASH precision guide](ash-projection-precision.md)
for provider and hardware requirements.

### Validate the index and scorer

Grid validates the selected scorer, layout, similarity function, and precision
before training or construction. Disk writer, page alignment, cache, and fused
layout settings apply to `onDisk` cases. Fused disk search requires ASH or PQ search compression.

With `reranking.type: None` or an omitted reranker, search returns up to K results
ordered by traversal score, including compressed scores where supported. If the
selected scorer or layout requires a reranker, Grid throws an exception during
configuration validation. The fused ASH and PQ scorer setup requires an FP32 or
NVQ scoring feature.

Grid rejects settings that do not apply to the case. Search type `None` scores
with full-precision vectors, so it requires `reranking.type: None`. Explicit precision
requests require support on the selected provider and machine.

## Run configuration

The run YAML applies to every index configuration in an invocation. Select it
with `-Djvector.bench.runConfig=<path>`. Its sections, in order:

| Section | Purpose |
| --- | --- |
| `benchmarks` | The benchmarks to compute and the stats from each: `throughput` (`AVG`, `MEDIAN`, `MAX` queries per second), `latency` (`AVG`, `STD`, `P999` for one query at a time), `count` (`visited`, `expanded`, `expanded base layer` nodes per query) and `accuracy` (`recall`, `MAP`). |
| `measurement` | Optional. How the benchmarks run; see below. |
| `console` | The benchmark stats and named metrics to print. |
| `logging` | Optional. The directory, run identifier, stats and metrics for `experiments.csv`. |

`measurement` settings, each optional:

| Field | Values | Description |
| --- | --- | --- |
| `evictIndex` | `true`, `false` | Cold index: drops each on-disk index file from the page cache before its first search, so search reads it from storage. Pages the open index has already read stay cached. Linux only. Defaults to `false`. |
| `throughput.warmup` | `random`, `base`, `none` | Warmup queries. `random` uses random vectors; `base` uses base vectors sampled from the dataset, which follow the data without favoring the measured queries. Defaults to `random`. |
| `throughput.warmupPasses` | Integer ≥ 0 | Warmup passes over the query count. Defaults to 3. |
| `throughput.untilCompiled` | `true`, `false` | Continues warming, up to 10 passes, until a pass adds little JIT compilation, so timed passes run compiled code. Defaults to `false`. |
| `throughput.passes` | Integer ≥ 1 | Timed passes; the throughput stats summarize them. With `1` and `minSeconds: 0`, each held-out query runs once (cold queries) in the first search case. Defaults to 3. |
| `throughput.minSeconds` | Number ≥ 0 | A timed pass repeats the queries until it has run this long; `0` runs them once. Defaults to 1.0. |

Named metrics under `console.metrics` and `logging.metrics` come in categories:
`system` (heap use), `disk` (index files), `construction` (build phases) and
`execution`. The `execution` metrics describe the throughput benchmark's timed
passes: `response_mean_ms` and the `response_p50_ms`, `response_p95_ms`,
`response_p99_ms` and `response_p999_ms` percentiles of each query's time (with
queued execution, from submission to completion; with direct execution, the
search itself); `mean_batch_size` for queued ASH preparation; and, on Linux,
`page_reads_per_query`, the process's major page faults per query, which counts
pages read from storage.

[The default run configuration](../jvector-examples/yaml-configs/run-config.yml)
prints one short table for a first run.
[The larger-than-memory run configuration](../jvector-examples/yaml-configs/run-config-larger-than-memory.yml)
measures queries that read the index from storage.

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
  queryExecution: queued
  compression:
    - type: ASH
      parameters:
        optimizer: ITQ
        encodedBits: 808
        bitsPerDimension: 2
        landmarkCount: 1
        queryProjection: fp16
        maxBatchSize: 8
```

For a source-vector memory case, select `inMemory`, use `None` compression in
both sections, and omit the disk settings. Use `reranking.type: None` to
measure search with traversal scores alone. For an index larger than memory,
see the [larger-than-memory example](../jvector-examples/yaml-configs/index-parameters/larger-than-memory).

## Related guides

- [How to run benchmarks](benchmarking.md) covers dataset setup and invocation.
- [Fused PQ and ASH comparison](ash-pq-benchmark.md) specifies the matched
  compression budgets, machine setup, and measurement procedure.
- [ASH configuration and defaults](ash-configuration.md) covers model and
  runtime settings.
- [Building and serving larger-than-memory indexes](larger-than-memory-indexes.md)
  covers sizing, page alignment and query settings for indexes served from disk.
- [Deferred batch serialization](deferred-batch-writer.md) explains writer
  batching and feature production.

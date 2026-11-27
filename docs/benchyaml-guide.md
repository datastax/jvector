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
  A dataset marked `is_int8: true` holds signed int8 vectors (unsigned `.bvecs` data such as
  SIFT1B must be converted first). Its cases use `type: onDisk`, compression `None`,
  `reranking.type: None` and `fusedGraph: [No]`, as the int8 files in `index-parameters` do.
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
| `streaming` | `Yes`, `No` | Optional scalar. `Yes` builds the index as a live index is built, inserting vectors as they arrive; see [Streaming builds](#streaming-builds). Defaults to `No`. |
| `coldStartSize` | Integers ≥ 0 | Required with `streaming: Yes` and ASH construction, which it sweeps. `0` starts the index with the construction compressor's model. A positive count starts it cold; see [Streaming builds](#streaming-builds). PQ and `None` streaming builds omit it or use `[0]`. |
| `trainWhileStreaming` | `Yes`, `No` | Optional with `streaming: Yes` and ASH construction, swept. `Yes` lets a cold index train its first model, with the construction compressor's optimizer, once its sample holds 20 × the dimension vectors. `No` fixes a random projection after `coldStartSize` vectors instead. Defaults to `No`. |
| `writerMode` | `sequential`, `parallel` | Selects the writer after graph cleanup for `onDisk`. Defaults to `parallel`. |
| `pageAlignment` | Power of two from 512 to 1048576 | Optional, `onDisk` only. Places base-layer node records so none crosses a page boundary of this size, typically `4096`; see [Page-aligned records](larger-than-memory-indexes.md#page-aligned-records). Omitted, records are packed back to back. Part of the saved-index identity. |
| `useSavedIndexIfExists` | `Yes`, `No` | Controls reuse of a compatible index, `inMemory` or `onDisk`, batch or streaming. `Yes` saves each built index, with the traversal codes it stores, in the working directory's `index_cache` and reuses it on later runs; an `inMemory` index is reloaded into memory. A cache hit skips fresh construction and serialization measurements. Saved indexes are never deleted automatically, so starting-point configurations use `No`; recipes that query an index from a later JVM, such as the ASH/PQ comparison, use `Yes`. |
| `workers` | Positive integer or `max` | Optional. Threads for quantizer training, encoding, graph insertion and the parallel writer. `max` is the platform's physical core count, which is also the default. |
| `outDegree` | Positive integers | Maximum final neighborhood degree, also called M. |
| `upperLayerDegree` | Positive integers | Optional, schema 3, with `addHierarchy: [Yes]`. Maximum degree of the layers above the base layer; with it, `outDegree` is the base layer's degree. Omit for `outDegree` on every layer. Each value is one more build. |
| `efConstruction` | Positive integers | Search width used to find candidates during insertion. |
| `neighborOverflow` | Finite numbers ≥ 1.0 | Temporary neighborhood capacity relative to final degree. Java float suffixes such as `1.2f` are accepted. |
| `addHierarchy` | `Yes`, `No` | Builds upper graph layers. |
| `refineFinalGraph` | `Yes`, `No` | Refines the graph during final cleanup. |
| `fusedGraph` | `Yes`, `No` | Applies to `onDisk`. `Yes` stores ASH or PQ neighbor codes alongside the graph records. `search.compression` selects the codec. |
| `compression[].type` | `None`, `ASH`, `PQ`, `BQ` | Selects the representation for construction scoring. `None` uses source vectors. |
| `compression[].parameters` | [Quantizer parameters](#quantizer-parameters) | Supplies training and encoding settings for this entry. |
| `compression[].scoring` | `symmetric`, `asymmetric`, `mixed` | Optional for ASH construction. `symmetric` scores encoded vectors during insertion and pruning. `asymmetric` scores source queries against codes for both operations. `mixed` uses asymmetric insertion and symmetric pruning. |
| `reranking.type` | `None`, `FP32`, `NVQ` | `FP32` rescores candidates with source vectors. `NVQ` rescores them with NVQ codes. `None` or omission skips reranking where the scorer and layout allow it. |

### Streaming builds

A streaming case adds the base vectors one at a time, in file order, from one
thread per construction worker, to an index that keeps no copy of the vectors,
as a database's live index would. The case needs a single `compression`
entry of type `ASH`, `PQ` or `None`. Each defines how the index starts and what
an insert does. Searches of an `inMemory` index use the codes it built,
whatever `search.compression` says, with the search case's query settings.

- **ASH warm start.** With `coldStartSize: [0]` the index starts with the
  construction compressor's model and encodes each vector as it arrives. Train
  that model on a window with the ASH `trainingStart` and `trainingCount`
  parameters to stand in for a model inherited from an earlier index.
- **ASH cold start.** With a positive `coldStartSize` the index has no graph at
  first: it records the vectors that arrive and searches them with exact
  scores. When its first model is ready, trained with
  `trainWhileStreaming: [Yes]` or a random projection without it, the index
  encodes the recorded vectors and inserts them into a new graph with ASH
  scores while vectors keep arriving, then continues with ASH. While the first
  model trains, inserts run on at most half the logical CPUs, so the training
  gets the CPU it needs.
- **PQ.** The construction compressor's PQ model is trained before the first
  vector arrives. Each vector is then encoded into mutable PQ codes and added
  to the graph, which scores with those codes, as an index under construction
  in a database does. PQ has no cold start.
- **None.** Each vector is added to the graph with exact scores.
- **The flush.** At the end, the build readies the index as a flush does: an
  ASH index ends every vector on its newest model.

`inMemory` streaming indexes rerank from the source vectors, so `reranking` is
`FP32` or `None`. `onDisk` streaming indexes are then written and reopened for
search, with `fusedGraph: [No]`. An ASH index's codes are stored beside it and
searched after reloading; PQ and `None` indexes are searched with
`search.compression`, as other `onDisk` builds are. `NVQ` reranking is centered
with the mean the index kept as vectors arrived, so writing reads each vector
once. The console
reports the insertion rate and the memory the index holds; for ASH, also the
streaming timeline and how long fixing the first model held inserts. Build time covers insertion,
the flush preparation and graph cleanup, plus writing for `onDisk`. With
`useSavedIndexIfExists: Yes`, a saved streaming index is reused, with its
codes, only by a case with the same streaming settings and insert threads, so
later JVMs can query it, for example with fewer cores. An `inMemory` PQ
streaming index is always built fresh, since saved indexes keep only ASH codes.

ASH construction uses symmetric scoring by default. When `scoring` is omitted,
the configured ASH construction policy applies. See
[ASH configuration and defaults](ash-configuration.md#graph-construction)
for the JVM setting.

An `onDisk` index that does not fuse its ASH codes (`fusedGraph: [No]`) writes the build's
codes beside the index file during the flush, as a deployed index would, and a search with the
same ASH settings loads them when it opens the index instead of encoding the base vectors again.
Indexes cached before codes were stored, PQ and BQ codes, and searches with other ASH settings
still encode. [Results](#results) defines the build-phase and code timings.

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
| `workers` | List of positive integers or `max` | Optional sweep. Direct search uses this many calling threads; queued search uses this many workers to prepare and execute queries. Each value queries the same index, so `[1, 8]` reports 1 and 8 workers from one build. Later search cases repeat queries that earlier cases already ran, so for cold-query measurements use one search case per JVM. Omitted, the JVM-wide settings apply. |
| `queryExecution` | `direct`, `queued` | `direct` runs each search on a calling thread. `queued` sends searches to a fixed pool of workers through a bounded queue and, for ASH, prepares waiting queries together, with fused codes or with codes stored beside the index or held in memory. Applies to every compression entry. Defaults to `direct`. |
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
| `evictIndex` | `true`, `false` | Cold index: drops each on-disk index file from the page cache before its first search, so search reads it from storage. Pages the open index has already read stay cached. Linux and JDK 22+ only. Defaults to `false`. |
| `throughput.warmup` | `random`, `base`, `none` | Warmup queries. `random` uses random vectors; `base` uses base vectors sampled from the dataset, which follow the data without favoring the measured queries. Defaults to `random`. |
| `throughput.warmupPasses` | Integer ≥ 0 | Warmup passes over the query count. Defaults to 3. |
| `throughput.untilCompiled` | `true`, `false` | Continues warming, up to 10 passes, until a pass adds little JIT compilation, so timed passes run compiled code. Defaults to `false`. |
| `throughput.passes` | Integer ≥ 1 | Timed passes; the throughput stats summarize them. With `1` and `minSeconds: 0`, each held-out query runs once (cold queries) in the first search case. Defaults to 3. |
| `throughput.minSeconds` | Number ≥ 0 | A timed pass repeats the queries until it has run this long; `0` runs them once. Defaults to 1.0. |

Named metrics under `console.metrics` and `logging.metrics` come in the categories
`system`, `disk`, `construction` and `execution`. [Results](#results) defines each one.

[The default run configuration](../jvector-examples/yaml-configs/run-config.yml)
prints one short table for a first run.
[The larger-than-memory run configuration](../jvector-examples/yaml-configs/run-config-larger-than-memory.yml)
measures queries that read the index from storage.

## Results

Each search case prints one console row per overquery and, with `logging`, appends one row to
`experiments.csv` in the run directory. The console shows the selected fields under their
headers; the CSV starts with fixed columns, then one column per selected field, named by its key.
A field a case does not produce is left blank.

### Fixed CSV columns

| Column | Meaning |
| --- | --- |
| `schema_version` | Version of the CSV layout. |
| `run_id`, `run_uuid` | The run's directory name (from `logging.runId`) and a unique id for it. |
| `system_id` | A hash of the machine and JVM description in the run's system information, for joining runs from one system. |
| `dataset_name` | The dataset. |
| `M`, `efConstruction`, `neighborOverflow`, `addHierarchy`, `refineFinalGraph` | The graph construction parameters. `M` is `outDegree`. |
| `feature_set` | The features stored in the index, such as `[NVQ_VECTORS, FUSED_ASH]`. |
| `build_compressor`, `search_compressor` | The compressors used for construction and for search traversal. |
| `usePruning`, `topK`, `overquery` | The search settings of the row. |
| `rerankK` | Candidates kept for reranking, `topK × overquery` rounded down. |
| `ash_query_projection_requested`, `ash_query_projection_effective` | The requested ASH query projection precision and the one in effect; blank without ASH search. |
| `index_type`, `writer_mode`, `reranking_type`, `ash_construction_mode` | Schema 3 only: `construction.type`, `writerMode`, `reranking.type` and the ASH construction scoring mode. |
| `query_execution`, `query_threads`, `max_batch_size` | Schema 3 only: `direct` or `queued`, the query workers (queued) or caller threads (direct), and the ASH query batch limit. |

### Benchmark fields

Selected under `benchmarks`. In the console, `(of N)` under a throughput header is the number of
timed passes.

| Selector | CSV key | Console header | Meaning |
| --- | --- | --- | --- |
| `throughput: AVG` | `search.throughput.avg_qps` | Avg QPS | Mean queries per second over the timed passes. |
| | `search.throughput.stddev_qps`, `search.throughput.cv_pct` | ± Std Dev, CV % | Spread of QPS across passes, as a standard deviation and as a percentage of the mean. Only with more than one pass. |
| `throughput: MEDIAN` | `search.throughput.median_qps` | Median QPS | Median QPS over the timed passes. |
| `throughput: MAX` | `search.throughput.max_qps`, `search.throughput.min_qps` | Max QPS, Min QPS | Fastest and slowest timed pass. |
| `latency: AVG`, `STD`, `P999` | `search.latency.mean_ms`, `std_ms`, `p999_ms` | Mean, STD, p999 Latency (ms) | Time of one query run on its own, one at a time: mean, standard deviation and 99.9th percentile. |
| `count: visited` | `search.count.avg_visited` | Avg Visited | Nodes scored per query. |
| `count: expanded` | `search.count.avg_expanded` | Avg Expanded | Nodes whose neighbors were explored, per query. |
| `count: expanded base layer` | `search.count.avg_expanded_base_layer` | Avg Expanded Base Layer | The same, in the base layer only. |
| `accuracy: recall` | `search.accuracy.recall_at_{topK}` | Recall@{topK} | Fraction of each query's true `topK` neighbors among its `topK` results, averaged over queries. |
| `accuracy: MAP` | `search.accuracy.map_at_{topK}` | MAP@{topK} | Mean average precision of the `topK` results. |

### Named metrics

Selected under `metrics`, by category.

| Category: field | CSV key | Console header | Meaning |
| --- | --- | --- | --- |
| `system: max_heap_mb` | `search.system.max_heap_mb` | Max heap usage (MB) | The larger of the heap in use before and after the case's queries; not a sampled peak. |
| `system: max_offheap_mb` | `search.system.max_offheap_mb` | Max offheap usage (MB) | The same for direct and mapped buffers. |
| `disk: file_count`, `total_file_size_mb` | `search.disk.file_count`, `search.disk.total_file_size_mb` | File count, Total file size (MB) | Files in the work and saved-index directories after the queries, and their total size. |
| `construction: index_build_time_s` | `construction.index_build_time_s` | Index build time (s) | Building and writing the index; blank when every index came from the index cache. |
| `construction: graph_build_time_s` | `construction.graph_build_time_s` | Graph build time (s) | `onDisk`: graph construction and cleanup. |
| `construction: flush_time_s` | `construction.flush_time_s` | Flush time (s) | `onDisk`: writing the completed graph, its features and any stored codes. Graph build plus flush equals the index build time. |
| `construction: reopen_time_s` | `construction.reopen_time_s` | Reopen time (s) | `onDisk`: opening the written index files. |
| `construction: codes_write_time_s` | `construction.codes_write_time_s` | Codes write time (s) | The part of the flush spent writing the ASH codes stored beside a non-fused index. |
| `construction: codes_file_size_mb` | `construction.codes_file_size_mb` | Codes file size (MB) | Size of the stored codes file, when the search uses it. |
| `construction: index_quant_time_s` | `construction.index_quant_time_s.{type}.compute_time_s`, `.encoding_time_s` | Idx {type} compute (s), Idx {type} encoding (s) | Training the quantizer and encoding the vectors for construction, per quantizer type: PQ, BQ, ASH or NVQ. Compute time is blank when the model was loaded from the compressor cache. |
| `construction: search_quant_time_s` | `search.search_quant_time_s.{type}.compute_time_s`, `.encoding_time_s` | Search {type} compute (s), Search {type} encoding (s) | The same for the search traversal codes of an index that does not fuse them. |
| | `search.search_quant_time_s.ASH.load_time_s` | Search ASH load (s) | Loading the ASH codes stored with the index, in place of training and encoding; a row has this or the encoding time, not both. |
| `execution: response_mean_ms`, `response_p50_ms`, `response_p95_ms`, `response_p99_ms`, `response_p999_ms` | `search.execution.response_*` | Loaded mean, p50, p95, p99, p99.9 (ms) | Each query's time in the throughput benchmark's timed passes: with queued execution from submission to completion, with direct execution the search itself. |
| `execution: mean_batch_size` | `search.execution.mean_batch_size` | Mean batch | Queries prepared together, on average, by queued ASH execution. |
| `execution: page_reads_per_query` | `search.execution.page_reads_per_query` | Page reads (per query) | Linux only: the process's major page faults per query in the timed passes, which counts pages read from storage. |

## Quantizer parameters

The fields below belong to the `parameters` map of their compression entry.
D is the dataset dimension.

| Quantizer and field | Values | Description |
| --- | --- | --- |
| ASH `optimizer` | `ITQ`, `RANDOM` | Training transform. Required for a new model. |
| ASH `compressionRatio` | 8, 16, 32, 64, 128, 256 | Nominal FP32-to-code ratio. The parser rounds to whole projected dimensions and uses a minimum payload. Specify either `compressionRatio` or `encodedBits`. |
| ASH `encodedBits` | Positive integer bit budget | Includes a 40-bit header. The payload must be divisible by `bitsPerDimension` and yield 1 through D projected dimensions. |
| ASH `bitsPerDimension` | 1–9, in memory and fused. 1, 2 and 4 have the fastest kernels. | Bits per projected dimension in the stored codes. Defaults to 1. |
| ASH `landmarkCount` | 1–256 | Number of landmarks. Defaults to 1. The model is trained for the dataset's similarity. Symmetric and mixed construction require one landmark. |
| ASH `trainingStart`, `trainingCount` | Base ordinals inside the dataset | Optional. Trains the model only on `trainingCount` base vectors starting at `trainingStart` (default 0), as an index that has seen only its first vectors would. All base vectors are still encoded. Omitted, training samples the whole set. A windowed model gets its own cache entries, and its codes are always re-encoded rather than loaded. |
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

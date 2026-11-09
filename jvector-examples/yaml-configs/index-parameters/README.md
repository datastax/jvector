# Index parameters

## Dataset defaults

Each `<dataset>.yml` in this directory is that dataset's default. Running BenchYAML with a dataset
name loads `index-parameters/<dataset>.yml`. The file name must match the dataset name exactly,
including case, and the file must declare that dataset. A dataset without its own file uses
`default.yml`. Subdirectories are never searched for defaults.

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML mpnet-1m
```

## Examples

Each subdirectory holds complete configurations for one pipeline. Run them by path. BenchYAML
prints the pipeline each configuration selects before it runs.

| Folder | Pipeline | Use case |
| --- | --- | --- |
| [in-memory-fp](in-memory-fp) | inMemory: build None -> search None -> rerank None | Small or frequently queried collections that fit in memory, and lower-dimensional data where quantization saves little. |
| [in-memory-ash](in-memory-ash) | inMemory: build None -> search ASH -> rerank FP32 | Larger in-memory collections that need faster traversal while keeping exact final scores. |
| [in-memory-bq](in-memory-bq) | inMemory: build BQ -> search BQ -> rerank FP32 | Very high-dimensional embeddings, where sign codes keep construction and traversal cheap. |
| [flush-fused-ash-nvq](flush-fused-ash-nvq) | onDisk: build None -> search fused ASH -> rerank NVQ | The database write path: an in-memory index (for example a memtable) is flushed to an on-disk index. |
| [compressed-build-fused-ash-nvq](compressed-build-fused-ash-nvq) | onDisk: build ASH symmetric -> search fused ASH -> rerank NVQ | Ingest under a memory limit: construction scores compact codes instead of source vectors. |
| [flush-fused-pq-nvq](flush-fused-pq-nvq) | onDisk: build None -> search fused PQ -> rerank NVQ | The same write path as flush-fused-ash-nvq with PQ codes, for comparison and compatibility. |
| [on-disk-ash-fp32](on-disk-ash-fp32) | onDisk: build ASH -> search ASH -> rerank FP32 | Indexes too large for memory, where compact codes stay resident and full vectors are read only for rescoring. |
| [on-disk-pq-fp32](on-disk-pq-fp32) | onDisk: build PQ -> search PQ -> rerank FP32 | The same layout as on-disk-ash-fp32 with PQ codes. |
| [on-disk-fp](on-disk-fp) | onDisk: build None -> search None -> rerank None | Lower-dimensional data stored on disk, and tuning graph degree, build width, hierarchy and pruning. |
| [ash-options](ash-options) | onDisk: build ASH -> search fused ASH -> rerank NVQ | Choosing ASH settings for a deployment. Each file changes one option from the folder's shared setup. |
| [studies](studies) | Varies | Published comparisons, run with their own scripts |

A pipeline combines the index type, construction scoring, fused codes, search scoring and
reranking. See [the BenchYAML guide](../../../docs/benchyaml-guide.md) for every field.

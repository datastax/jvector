# On-disk graph with separate PQ codes

Builds with PQ scoring and writes the graph with the source vectors. Searches traverse with PQ codes held in memory and rescore from disk.

**Use case:** The same layout as on-disk-ash-fp32 with PQ codes.

**Pipeline:** onDisk: build PQ -> search PQ -> rerank FP32

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/on-disk-pq-fp32/*.yml
```

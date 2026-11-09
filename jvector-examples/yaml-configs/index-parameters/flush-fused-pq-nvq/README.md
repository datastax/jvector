# Build in memory, flush a fused PQ graph

Builds the graph with the source vectors, then writes fused PQ traversal codes and NVQ reranking vectors to disk.

**Use case:** The same write path as flush-fused-ash-nvq with PQ codes, for comparison and compatibility.

**Pipeline:** onDisk · build None · search fused PQ · rerank NVQ

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/flush-fused-pq-nvq/*.yml
```

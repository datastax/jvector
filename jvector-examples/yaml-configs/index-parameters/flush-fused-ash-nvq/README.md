# Build in memory, flush a fused ASH graph

Builds the graph with the source vectors, then writes fused ASH traversal codes and NVQ reranking vectors to disk.

**Use case:** The database write path: an in-memory index (for example a memtable) is flushed to an on-disk index.

**Pipeline:** onDisk: build None -> search fused ASH -> rerank NVQ

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset. ASH requires dot-product datasets.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/flush-fused-ash-nvq/*.yml
```

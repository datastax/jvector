# Build with ASH codes, flush a fused ASH graph

Builds the graph with symmetric ASH scoring, then writes fused ASH traversal codes and NVQ reranking vectors in parallel.

**Use case:** Ingest under a memory limit: construction scores compact codes instead of source vectors.

**Pipeline:** onDisk · build ASH symmetric · search fused ASH · rerank NVQ

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset. ASH requires dot-product datasets.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/compressed-build-fused-ash-nvq/*.yml
```

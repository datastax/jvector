# In-memory graph built and searched with BQ

Builds and searches an in-memory graph with 1-bit BQ codes, then rescores with the source vectors.

**Use case:** Very high-dimensional embeddings, where sign codes keep construction and traversal cheap.

**Pipeline:** inMemory: build BQ -> search BQ -> rerank FP32

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/in-memory-bq/*.yml
```

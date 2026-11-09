# In-memory graph with ASH search

Builds an in-memory graph with the source vectors. Searches traverse with ASH codes and rescore the candidates with the source vectors.

**Use case:** Larger in-memory collections that need faster traversal while keeping exact final scores.

**Pipeline:** inMemory · build None · search ASH · rerank FP32

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset. ASH requires dot-product datasets.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/in-memory-ash/*.yml
```

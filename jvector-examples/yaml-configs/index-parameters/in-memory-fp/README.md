# In-memory graph, full-precision vectors

Builds and searches a graph held entirely in memory with the source vectors. Nothing is written to disk and nothing is quantized.

**Use case:** Small or frequently queried collections that fit in memory, and lower-dimensional data where quantization saves little.

**Pipeline:** inMemory · build None · search None · rerank None

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/in-memory-fp/*.yml
```

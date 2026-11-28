# Streaming build with full-precision vectors

Builds an in-memory graph as vectors arrive, in file order, from one insert thread per construction worker. Each vector is added to the graph with exact scores; nothing is quantized, and searches use full-precision vectors.

**Use case:** The full-precision baseline for streaming builds, as a database builds an index in memory before it flushes it. Compare with [streaming-pq](../streaming-pq), which builds the same graph from PQ codes.

**Pipeline:** inMemory streaming: build None -> search None -> rerank None

See [Streaming builds](../../../../docs/benchyaml-guide.md#streaming-builds). Compare streaming times with other streaming runs, not with batch builds, whose insert order and threads differ.

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset.

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/streaming-fp/*.yml
```

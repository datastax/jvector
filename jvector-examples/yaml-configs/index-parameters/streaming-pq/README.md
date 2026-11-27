# Streaming build with PQ codes

Builds an in-memory graph as vectors arrive, in file order, from one insert thread per construction worker. The PQ model is trained first; each vector is then encoded into mutable PQ codes and added to the graph, which scores with those codes. Searches use the same codes and rescore with full-precision vectors.

**Use case:** Timing graph construction from in-progress PQ codes, as a database builds an index in memory before it flushes it.

**Pipeline:** inMemory streaming: build PQ -> search PQ -> rerank FP32

Set the construction compression to `None` to stream with exact scores, or to `ASH` (with `coldStartSize`) for a streaming ASH index; see [Streaming builds](../../../../docs/benchyaml-guide.md#streaming-builds). Compare streaming times with other streaming runs, not with batch builds, whose insert order and threads differ.

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset.

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/streaming-pq/*.yml
```

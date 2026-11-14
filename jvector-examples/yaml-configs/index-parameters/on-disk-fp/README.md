# On-disk graph, full-precision vectors

Builds and searches an on-disk graph with the source vectors, without quantization. Includes a graph-settings sweep.

**Use case:** Lower-dimensional data stored on disk, and tuning graph degree, build width, hierarchy and pruning.

**Pipeline:** onDisk: build None -> search None -> rerank None

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/on-disk-fp/*.yml
```

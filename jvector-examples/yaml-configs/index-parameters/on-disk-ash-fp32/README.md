# On-disk graph with separate ASH codes

Builds with ASH scoring and writes the graph with the source vectors. Searches traverse with ASH codes held in memory and rescore from disk.

**Use case:** Indexes too large for memory, where compact codes stay resident and full vectors are read only for rescoring.

**Pipeline:** onDisk: build ASH -> search ASH -> rerank FP32

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset. ASH requires dot-product datasets.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/on-disk-ash-fp32/*.yml
```

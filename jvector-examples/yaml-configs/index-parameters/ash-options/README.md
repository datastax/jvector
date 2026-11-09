# ASH options

Fused ASH graphs with NVQ reranking, each changing one ASH option: construction scoring, landmarks, optimizer, query projection or query execution.

**Use case:** Choosing ASH settings for a deployment. Each file changes one option from the folder's shared setup.

**Pipeline:** onDisk: build ASH -> search fused ASH -> rerank NVQ

Files are named for their dataset. Each sets `dataset:`; change it to run the same pipeline on another compatible dataset. ASH requires dot-product datasets.

Run one file, or every file in the folder, by passing paths to BenchYAML from the repository root:

```bash
java -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML jvector-examples/yaml-configs/index-parameters/ash-options/*.yml
```

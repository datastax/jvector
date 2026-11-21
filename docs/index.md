# JVector

JVector is a Java library for approximate nearest neighbor (ANN) vector search. It builds graph
indexes that combine an HNSW-style hierarchy with Vamana (the algorithm behind DiskANN), can be
built concurrently and incrementally, and can be searched from disk with compressed vectors.

- **Fast to build.** Construction scales with cores and can score compact codes instead of full
  vectors, so building needs far less memory than the vectors themselves.
- **Larger than memory.** Indexes can be served from disk, with compressed codes stored next to
  each node's neighbors and accurate vectors used to rerank the best candidates.
- **Choice of quantizers.** Product quantization (PQ), Asymmetric Scalar Hashing (ASH,
  experimental), binary quantization (BQ), and NVQ for reranking.
- **Native SIMD.** On Java 22 and later, distance and scoring kernels run natively on x86-64 and
  AArch64; Java 11 and newer are supported.

## Where to start

| If you want to… | Read |
| --- | --- |
| Add JVector to a project and run a first search | [Quick start](getting-started.md) |
| Learn the API step by step | [Tutorial 1: Introduction](tutorials/1-intro-tutorial.md) |
| Pick a quantizer and index layout | [Choosing an index](choosing-an-index.md) |
| Decide between fused and separated codes and vectors | [Fused and separated layouts](fused-and-separated.md) |
| Trade recall against speed | [Tuning construction and search](tuning.md) |
| Plan memory and disk | [Memory and disk sizing](sizing.md) |
| Serve an index bigger than memory | [Larger-than-memory indexes](larger-than-memory-indexes.md) |
| Measure an index on your data | [Running benchmarks](benchmarking.md) |
| Look up a class or method | [API reference (Javadoc)](https://javadoc.io/doc/io.github.jbellis/jvector) |

## Project

JVector is open source under the Apache 2.0 license. The source and issues are on
[GitHub](https://github.com/datastax/jvector). These pages are built from the repository's
`docs/` folder, so every change to them is reviewed with the code; use the edit icon on any page
to propose a fix.

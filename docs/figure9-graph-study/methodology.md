# Figure 9 graph comparison — protocol

Authoritative host: `ted_willke@34.169.37.132`, GCP c4-standard-96-lssd. ASH source: `/home/ted_willke/jvector-ash`, `ash-symmetric-scorer` commit `922b23e3`, plus benchmark-only explicit concurrency controls, subsequently committed as `638edc37`. PQ: verified GitHub origin/main `1a86718baa2d8b72a0d134711a682875888b9646`, isolated detached worktree. Identical ThroughputBenchmark source on both. No duplicate-ID changes.

## Matrix

Three datasets: Ada002-1M (982,790 base vectors) and CAP-1M (1,000,000), both D=1536; Cohere English v3-1M (1,000,000), D=1024. Each has 10,000 queries. User confirmed d=D/2 to match Figure 9: respectively 768, 768, 512. ASH b=2 and b=4. Graph ASH requires C=1. Production training remains 20D, max25 iterations with early stopping; paper IVF uses C=32 and 10D. The study does not change production training defaults.

PQ uses production JVector K=256, uncentered, anisotropy disabled. The actual fused query path is `FusedPQDecoder.DotProductDecoder`, with float partial-sum tables and SIMD `VectorUtil.assembleAndSum`; it is not Faiss 4-bit FastScan. Match exact stored quantized-vector bytes including ASH's 5-byte header: M=197/389 for Ada002/CAP and M=133/261 for Cohere, with uneven PQ subvector dimensions supported. The nominal 32X/16X paper labels refer to payload only; actual graph code compression is slightly lower once the ASH header is counted. Faiss IVF in the paper uses packed 4-bit PQ; these are different production PQ implementations.

Graph settings: M=64, efConstruction=200, overflow1.2, hierarchy/refinement enabled, fused neighborhoods, NVQ reranking, k=10 and rerankK=10/20 (overquery1/2), search pruning disabled. ASH retains production adaptive termination gamma0.01. PQ uses latest-main behavior, which has no ASH adaptive feature. ASH writes version7, latest-main PQ version6. This is a production-branch comparison, not an isolated quantizer ablation.

## Timing and retention

Two independent fresh compressor/graph builds per configuration (24 total), reversed ASH/PQ order on repetition2. No shared compressor cache across fresh repetitions. Each index is retained and reopened for a 48-query-worker concurrency pass. Explicit serial/concurrent checks compare complete ranked node IDs, scores and visited count for all test queries at each overquery setting. Recall and visited measurements are outside the throughput timer.

Single-threaded QPS is the main result. Three timed samples, each at least3seconds and a complete query batch; raw durations and query counts logged. Existing synthetic warmup remains enabled. Query concurrency uses a dedicated 48-worker pool. Grid submits insertion to PhysicalCoreExecutor with 48 workers; GraphIndexBuilder cleanup uses its default common ForkJoinPool (normally parallelism 95 on this 96-logical-CPU JVM). Both branches use this arrangement. Workers are not explicitly pinned, unlike the paper’s 48 pinned build threads. Asymmetric projection/setup remains inside each query call for ASH. The graph uses corrected symmetric node comparisons and asymmetric raw-vector insertion/query scoring.

JDK23, vector module, native access enabled, Xmx128g, ASH SIMD single/block kernels, production accumulator/unroll settings. Explicit `jvector-twenty` classpath supplies MemorySegmentReader; reader fallback warnings are checked. Temporary files and retained indexes are on RAID10 SSDs. These1M indexes fit RAM; warmed memory-mapped query throughput is not a larger-than-memory I/O result.

Retain graph build/write time separately from training, base encoding and provider setup. Add NVQ compressor setup to these measured phases; their sum is a measured phase subtotal, excluding uninstrumented writer initialization overhead. Process elapsed time additionally includes dataset loading, JVM startup, warmup and queries and is not called construction time.

The paper uses the same GCP c4-standard-96-lssd/Intel Xeon 6985P-C host class, 48 build cores and single-thread queries. Its IVF implementation is C++/AVX-512 with 4096 lists and an nprobe sweep; this graph experiment uses Java/JDK23 and two overquery settings.

No IVF reruns. Published Figure9 panels are extracted from the supplied final PDF and shown separately from measured graph points. Any comparison across index types must retain the landmark/training/PQ implementation and reranking differences above.

Remote evidence and reusable caches: `/mnt/raid10/jvector-bench/ted_willke/figure9-graph/`. Per-case directories contain YAMLs, `index_cache`, ASH `compressor_cache` or latest-main PQ `pq_cache`, logs and completion manifests. Frozen JARs and hashes are stored at the study root.

## Interpretation

The runtime profile confirms that graph construction uses `ImmutablePQVectors.diversityFunctionFor(DOT_PRODUCT)`, which caches centroid-pair partial sums and calls `VectorUtil.assembleAndSumPQ`. The base `PQVectors` implementation is not the active diversity scorer in these runs. Corrected ASH construction compares packed codes. Query scoring remains asymmetric in both configurations.

Figure9 compares published IVF search curves. It does not report complete graph or IVF index-build times. Its ASH/PQ query paths do not use this graph's NVQ reranking configuration, so higher graph terminal recall must not be attributed solely to the quantizer. Graph construction is compared directly between the two measured graph implementations.

The exact same graph is used for each serial/concurrent comparison; independent graph repetitions are used only for build/run variability. Two repetitions provide ranges, not narrow statistical confidence intervals.

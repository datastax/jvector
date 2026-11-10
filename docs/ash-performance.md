# ASH performance

This page compares fused ASH and fused PQ graph indexes on three large datasets,
at matched code sizes: both store the same bytes per vector, including ASH's
five-byte header. [Reproducing ASH versus PQ](ash-pq-benchmark.md) explains how
to run the same study on your own hardware.

## Summary

ASH wins total construction in 6/6 configurations, recall in 12/12 cases and QPS
in 22/24 cases. The sum of all six construction times is 48.5% lower with ASH.
The better value in each comparison is in bold.

### Total construction seconds (lower wins)

| Dataset | Compression | PQ | ASH | ASH change | Faster |
| --- | --- | ---: | ---: | ---: | --- |
| MPNet 10M | 32x (b=2) | 343.7 | **238.1** | −30.7% | ASH |
| MPNet 10M | 16x (b=4) | 568.9 | **301.2** | −47.1% | ASH |
| Cohere 10M | 32x (b=2) | 478.0 | **263.3** | −44.9% | ASH |
| Cohere 10M | 16x (b=4) | 771.2 | **335.8** | −56.5% | ASH |
| CAP 6M | 32x (b=2) | 295.6 | **190.2** | −35.6% | ASH |
| CAP 6M | 16x (b=4) | 630.7 | **261.8** | −58.5% | ASH |
| **Sum: six configurations** | | 3,088.1 | **1,590.5** | −48.5% | ASH |

### Recall and throughput (higher wins)

| Dataset | Compression | Rerank | Recall@10 PQ | Recall@10 ASH | QPS 1 core PQ | QPS 1 core ASH | QPS 8 cores PQ | QPS 8 cores ASH |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MPNet 10M | 32x (b=2) | 1x | 0.54043 | **0.75532** | **9,349** | 9,012 | 77,609 | **81,421** |
| MPNet 10M | 32x (b=2) | 2x | 0.71512 | **0.88903** | **7,540** | 7,266 | 61,759 | **62,913** |
| MPNet 10M | 16x (b=4) | 1x | 0.72452 | **0.81330** | 7,224 | **9,860** | 63,221 | **76,619** |
| MPNet 10M | 16x (b=4) | 2x | 0.86195 | **0.90050** | 5,661 | **7,397** | 48,894 | **57,709** |
| Cohere 10M | 32x (b=2) | 1x | 0.52124 | **0.58682** | 7,642 | **9,067** | 62,520 | **72,500** |
| Cohere 10M | 32x (b=2) | 2x | 0.70611 | **0.76181** | 5,826 | **6,875** | 48,494 | **55,340** |
| Cohere 10M | 16x (b=4) | 1x | 0.59860 | **0.66033** | 6,145 | **8,339** | 50,152 | **65,660** |
| Cohere 10M | 16x (b=4) | 2x | 0.75846 | **0.78864** | 4,684 | **6,122** | 38,369 | **49,100** |
| CAP 6M | 32x (b=2) | 1x | 0.41240 | **0.72119** | 5,815 | **7,295** | 51,746 | **62,544** |
| CAP 6M | 32x (b=2) | 2x | 0.59705 | **0.88574** | 4,748 | **5,756** | 41,022 | **48,482** |
| CAP 6M | 16x (b=4) | 1x | 0.65664 | **0.80468** | 4,436 | **6,295** | 39,786 | **58,394** |
| CAP 6M | 16x (b=4) | 2x | 0.83966 | **0.90749** | 3,550 | **4,774** | 31,393 | **41,777** |

Compression ratios are nominal; ASH and PQ store the same bytes per vector,
including the ASH header. b is ASH bits per projected dimension.

## Setup and method

**Machine.** One Intel Xeon 6985P-C socket at 2.30 GHz base: 48 physical cores,
2 threads per core (96 hardware threads), 504 MiB L3. 354 GiB RAM in two NUMA
nodes of about 177 GiB each. Debian 12, Linux 6.1. Transparent huge pages:
enabled=always, defrag=madvise.

**Storage.** Datasets, model caches and indexes are on local NVMe SSDs in Linux
md RAID 10 (512 KiB chunk, two near-copies), formatted XFS. The page cache was
not dropped between runs.

**JVM.** OpenJDK 22.0.2 (Eclipse Temurin). Flags:
`--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -da -Xms2g -Xmx128g -XX:ActiveProcessorCount=<cores>`.
No collector is selected, so the JVM uses its default: G1 for the 8- and 48-core
runs and Serial for the 1-core runs, because `ActiveProcessorCount=1`.
`JAVA_TOOL_OPTIONS` and `JDK_JAVA_OPTIONS` are empty.

**JVector system properties.** `-Djvector.vectorization_provider=native
-Djvector.parallel_writer=true -Djvector.physical_core_count=<cores>
-Djvector.bench.queryThreads=<4 × cores> -Djvector.bench.verifyConcurrentResults=true
-Djvector.bench.throughputMinSeconds=3.0`. ASH construction scoring is
symmetric-all, the default; no other ASH properties are set.

**Native thread pools.** `MKL_NUM_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `GOTO_NUM_THREADS` equal the core count for
construction and 1 for queries; `MKL_DYNAMIC=FALSE`.

**CPU and memory placement.** Each JVM starts under `numactl`. Construction uses
one hardware thread on each of the 48 physical cores, with memory interleaved
across both NUMA nodes. Queries use 1 or 8 physical cores on node 0, with memory
bound to node 0.

**Datasets.** All use dot-product similarity on normalized vectors, 10,000
queries and exact top-100 ground truth. MPNet 10M: 9,997,697 base vectors × 768
dimensions. Cohere 10M (English v3): 10,000,000 × 1,024. CAP 6M (gte-Qwen2-1.5B
case law): 6,070,488 × 1,536.

**Index parameters.** On-disk index format 7, fused graph, parallel writer.
Degree (M) 32, `efConstruction` 100, neighbor overflow 1.2, alpha 1.2, hierarchy
and final refinement on. ASH: ITQ optimizer, one landmark (C=1), symmetric-all
construction scoring, and b = 2 or 4 bits per projected dimension over D/2
projected dimensions. Code size per vector, including the five-byte header:
MPNet 101 / 197 bytes, Cohere 133 / 261 bytes and CAP 197 / 389 bytes for
b = 2 / 4. PQ uses the same number of bytes as 8-bit subspaces with 256
centroids. Reranking uses NVQ vectors stored in the graph.

**Query parameters.** Top 10 returned. Overquery 1.0 and 2.0 (rerank 1x and 2x)
rescore 10 and 20 candidates with NVQ. Search pruning off. ASH uses queued
execution with a maximum batch of 8 and automatic query projection, which
resolves to FP16 on this CPU: one ASH worker per core (1 or 8), queue capacity 8
per core, fed by 4 × cores caller threads (4 or 32); the mean batch size is 8.0.
PQ uses direct execution with 4 × cores caller threads. Concurrent results are
verified against single-threaded results.

**Measurement protocol.** Each configuration was built twice from empty caches.
Construction time is quantizer training, vector encoding, NVQ training, and
graph construction and writing; dataset loading and setup are excluded. Each
construction figure is the mean of the two builds:

| Configuration | ASH build 1 (s) | ASH build 2 (s) |
| --- | ---: | ---: |
| MPNet 10M 32x (b=2) | 227.3 | 249.0 |
| MPNet 10M 16x (b=4) | 317.6 | 284.8 |
| Cohere 10M 32x (b=2) | 291.2 | 235.3 |
| Cohere 10M 16x (b=4) | 348.2 | 323.4 |
| CAP 6M 32x (b=2) | 192.6 | 187.8 |
| CAP 6M 16x (b=4) | 260.7 | 262.9 |

Query JVMs reuse the first build's index, one JVM for 1 core and one for 8 cores.
QPS is the mean of three timed samples of at least three seconds each; each JVM
runs about 90,000 rerank-1x queries in total. Recall@10 is the fraction of the
exact top 10 returned. Jobs ran one at a time, with no other benchmark process on
the machine. PQ was measured in an earlier run on the same machine, JDK,
placement, input files and graph parameters, with PQ codes sized to the same
stored bytes per vector as ASH.

**Single-core variation.** Single-core QPS varies between graphs built from the
same settings. On an earlier MPNet 10M 32x graph, ASH measured 10,414 / 7,838
QPS on one core (rerank 1x / 2x). Three further runs of the current code and of
the earlier code on the reported graph averaged 9,229 / 7,460 and 9,124 / 7,334
QPS with identical recall, so the code performs the same; on this graph, ASH's
single-core figures for that case are below PQ's 9,349 / 7,540.

**ASH memory.** Encoded ASH vectors are stored back to back in shared pages and
scored in place, without per-vector objects. Measured on 2M MPNet vectors as live
heap with and without the encoded vectors (Serial and G1 agree within 1%), ASH
retains 109 bytes per vector at b = 2 and 205 at b = 4, compared with 189 and 285
for the earlier per-vector storage: 42% and 28% less. The codes occupy 101 and
197 bytes per vector on disk; the remaining 8 bytes hold the header scale and
offset as floats and a per-vector construction term. The library's own
accounting (`ramBytesUsed`) reports the same 109 and 205 bytes.

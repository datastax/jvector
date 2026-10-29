# Native PQ versus trained ASH with graph indexing

Standard BenchYAML with ASH PCA correction `507b74e7ef8a0e12fcfe402deb4bf758c65758d5`. Four complete cleaned datasets, both JDK 22 and JDK 25, two exact code-byte budgets. All 32 final graphs and 64 cached query processes completed. No changes to BenchYAML or its timers were used.

A native PCA correctness failure interrupted the first collection. The QR SVD fast path returned incorrect decompositions on unchanged finite input with 48 MKL threads on both JDKs, sometimes reporting success. ASH PCA now uses a validated divide-and-conquer SVD. All ASH results below were recollected after the fix; earlier ASH graph records are archived as superseded. Nine completed PQ cases from source b44e25b7 were retained: every non-ASH runtime file was verified byte-for-byte identical after the correction. Each raw record retains its actual source SHA. See [the PCA investigation](../../ash-pca-native-svd.md) and [the runtime audit](pca-fix-runtime-audit.json).

Both algorithms use the native Highway provider, format V7, M=32, efConstruction=100, overflow=1.2, hierarchy and final refinement enabled, deferred parallel complete-record writes (batch 64), and NVQ reranking (two 8-bit subvectors). ASH uses C=1, ITQ training with production defaults, projected dimension D/2, and 2 or 4 bits per projected dimension. PQ uses K=256, no centering and anisotropy -1. Search pruning is disabled. Each query count searches the identical saved graph.

The existing configured system opts ASH into adaptive DOT_PRODUCT stopping; PQ uses its existing stopping path. `useSearchPruning: No` does not disable that separate ASH policy. The comparison includes these current construction and stopping policies; it does not isolate quantizer error alone. No policy was changed in response to measured results.

Compression budgets match stored ASH/PQ code bytes, including ASH’s 5-byte header. Shared models, graph structure and NVQ storage are additional. “2-bit/4-bit budget” refers to the ASH setting, not PQ bits per original dimension.

| Dataset | Base vectors | Queries | D → ASH d | Code bytes, 2-bit / 4-bit budget |
|---|---:|---:|---:|---:|
| MPNet | 999,812 | 10,000 | 768 → 384 | 101 / 197 |
| Cohere | 1,000,000 | 10,000 | 1024 → 512 | 133 / 261 |
| Ada002 | 982,790 | 10,000 | 1536 → 768 | 197 / 389 |
| OpenAI3072 | 999,000 | 1,000 | 3072 → 1536 | 389 / 773 |

## Construction — seconds

Each cell is **PQ → ASH**. Training, encoding and graph/write are the harness’s existing CSV measurements. They are reported as measured, without replacing the timers or subtracting work.

**Total process** is externally measured wall time of the fresh construction JVM, including startup, dataset loading, construction, one standard recall@10 validation pass at 1×, and teardown. It includes that accuracy pass; it is not labeled build-only. QPS warmup and measurement run in separate cached JVMs and are excluded. NVQ setup/encoding and other setup costs remain part of the actual process; the displayed phase columns need not sum to total process.

| Dataset | JDK | ASH bit budget | Training | Encoding | Graph/write | Total process incl. validation |
|---|---:|---:|---:|---:|---:|---:|
| MPNet | 22 | 2 | 2.73 → 5.21 | 1.08 → 1.09 | 44.44 → 58.37 | 52.27 → 70.72 |
| MPNet | 22 | 4 | 6.14 → 4.69 | 1.71 → 1.19 | 50.79 → 42.03 | 62.75 → 54.34 |
| MPNet | 25 | 2 | 2.77 → 4.80 | 1.05 → 1.00 | 33.77 → 31.63 | 41.55 → 43.57 |
| MPNet | 25 | 4 | 4.65 → 4.45 | 1.75 → 1.20 | 48.55 → 43.29 | 58.95 → 54.84 |
| Cohere | 22 | 2 | 2.97 → 8.03 | 1.32 → 1.86 | 35.40 → 62.37 | 44.33 → 96.01 |
| Cohere | 22 | 4 | 6.75 → 6.62 | 2.30 → 2.17 | 70.71 → 49.73 | 84.35 → 66.19 |
| Cohere | 25 | 2 | 3.08 → 7.50 | 1.33 → 1.64 | 40.66 → 32.97 | 49.73 → 52.84 |
| Cohere | 25 | 4 | 6.57 → 6.46 | 2.23 → 2.06 | 66.55 → 46.25 | 80.16 → 62.54 |
| Ada002 | 22 | 2 | 3.93 → 13.71 | 2.20 → 7.91 | 48.12 → 49.60 | 60.27 → 81.40 |
| Ada002 | 22 | 4 | 8.62 → 10.95 | 3.61 → 7.43 | 100.92 → 56.91 | 119.42 → 85.25 |
| Ada002 | 25 | 2 | 3.72 → 13.82 | 2.05 → 8.28 | 42.70 → 45.40 | 54.51 → 81.50 |
| Ada002 | 25 | 4 | 8.81 → 13.05 | 3.51 → 7.32 | 106.25 → 60.39 | 125.37 → 90.96 |
| OpenAI3072 | 22 | 2 | 6.51 → 41.12 | 5.51 → 24.10 | 90.50 → 85.71 | 113.08 → 169.55 |
| OpenAI3072 | 22 | 4 | 19.65 → 39.54 | 7.64 → 27.21 | 257.82 → 124.33 | 295.42 → 208.44 |
| OpenAI3072 | 25 | 2 | 5.90 → 39.41 | 5.58 → 24.28 | 113.01 → 87.03 | 135.15 → 170.94 |
| OpenAI3072 | 25 | 4 | 16.62 → 38.63 | 7.78 → 25.96 | 247.53 → 130.57 | 281.56 → 213.53 |

## Queries — recall@10 and QPS

Each cell is **PQ → ASH**. Recall is a percentage. 1× and 2× rerank 10 and 20 candidates respectively to return 10. QPS includes the graph search, normal query preparation and NVQ reranking. Construction uses 48 physical cores; throughput uses 1 or 8 scoring threads, pinned to 1 or 8 physical cores. Standard BenchYAML warmup and three measured samples are retained. No JIT compilation logs, per-call tracing or profiler runs were used for these measurements.

### 2-bit ASH code-byte budget

| Dataset | JDK | Recall 1× (%) | Recall 2× (%) | 1 thread, 1× QPS | 1 thread, 2× QPS | 8 threads, 1× QPS | 8 threads, 2× QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPNet | 22 | 65.13 → 81.94 | 82.13 → 94.06 | 11,930 → 9,611 | 9,441 → 7,822 | 86,215 → 70,990 | 68,760 → 59,039 |
| MPNet | 25 | 65.48 → 82.02 | 82.21 → 93.97 | 11,812 → 10,357 | 9,294 → 8,282 | 76,802 → 73,841 | 71,112 → 60,434 |
| Cohere | 22 | 59.43 → 63.90 | 78.29 → 81.99 | 9,695 → 6,041 | 7,700 → 5,224 | 74,000 → 47,759 | 59,165 → 39,833 |
| Cohere | 25 | 59.13 → 63.98 | 78.45 → 82.33 | 9,412 → 6,427 | 7,191 → 5,258 | 75,638 → 50,148 | 57,863 → 41,902 |
| Ada002 | 22 | 54.99 → 78.88 | 75.35 → 92.69 | 7,065 → 3,301 | 5,685 → 2,987 | 53,847 → 24,911 | 45,513 → 22,128 |
| Ada002 | 25 | 54.59 → 79.06 | 75.04 → 92.72 | 7,271 → 3,437 | 5,712 → 3,097 | 49,407 → 25,149 | 43,938 → 22,643 |
| OpenAI3072 | 22 | 64.37 → 77.92 | 79.85 → 89.80 | 3,329 → 1,031 | 2,954 → 1,068 | 27,544 → 7,190 | 22,785 → 7,037 |
| OpenAI3072 | 25 | 65.10 → 76.76 | 80.32 → 89.17 | 2,170 → 1,003 | 2,068 → 1,094 | 27,772 → 8,050 | 23,214 → 7,893 |

### 4-bit ASH code-byte budget

| Dataset | JDK | Recall 1× (%) | Recall 2× (%) | 1 thread, 1× QPS | 1 thread, 2× QPS | 8 threads, 1× QPS | 8 threads, 2× QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPNet | 22 | 80.73 → 86.64 | 92.75 → 93.97 | 9,643 → 8,630 | 7,456 → 6,976 | 74,893 → 64,520 | 57,187 → 52,279 |
| MPNet | 25 | 80.75 → 86.29 | 92.60 → 94.03 | 10,001 → 9,154 | 7,557 → 7,115 | 74,887 → 67,390 | 56,905 → 54,358 |
| Cohere | 22 | 65.08 → 70.69 | 81.04 → 83.40 | 7,656 → 5,460 | 5,826 → 4,610 | 57,545 → 41,673 | 43,366 → 35,207 |
| Cohere | 25 | 65.53 → 70.59 | 81.36 → 83.70 | 7,858 → 5,662 | 5,921 → 4,636 | 57,671 → 42,689 | 43,805 → 35,645 |
| Ada002 | 22 | 73.56 → 83.23 | 90.05 → 92.44 | 5,341 → 2,976 | 4,200 → 2,699 | 41,039 → 21,871 | 30,913 → 19,259 |
| Ada002 | 25 | 73.57 → 83.53 | 90.15 → 92.47 | 3,665 → 2,959 | 3,884 → 2,650 | 26,980 → 21,983 | 22,436 → 20,076 |
| OpenAI3072 | 22 | 76.91 → 79.03 | 87.70 → 88.34 | 2,144 → 1,016 | 1,871 → 1,031 | 18,467 → 7,508 | 14,336 → 7,361 |
| OpenAI3072 | 25 | 76.24 → 79.65 | 87.27 → 88.75 | 2,281 → 965 | 1,886 → 1,034 | 15,664 → 7,915 | 12,885 → 7,489 |

## Observations

- At 1× overquery, ASH has higher recall in 16/16 matched graph cases, ties in 0, and lower recall in 0. Recall differences range from +2.12 to +24.47 percentage points.
- At 2× overquery, ASH has higher recall in 16/16 matched graph cases, ties in 0, and lower recall in 0. Recall differences range from +0.64 to +17.68 percentage points.
- ASH has higher QPS in 0/64 matched query cells and lower QPS in 64. The geometric mean ASH/PQ QPS ratio is 0.627.
- ASH graph/write is faster in 12/16 matched cases; its geometric mean graph/write time ratio versus PQ is 0.819.
- ASH total construction/validation process time is lower in all eight 4-bit-budget cases and higher in all eight 2-bit-budget cases. These totals include the single accuracy validation pass described above.
- Query sample CV exceeds 10% in 4/128 cells (maximum 13.75%). queries.csv includes all CVs; raw-results.json and logs retain the sample query counts and durations. Each construction case has one fresh parallel graph build; these data do not estimate graph-build variance or isolate JDK effects from graph/training nondeterminism.

ASH query preparation evaluates the dense projection Aq once per graph query: d=D/2 row dot products, each of length D. At D=3072 this is 1,536 × 3,072 = 4,718,592 multiply-add positions before LUT construction and graph scoring. The current projection and ASH symmetric construction scorers use the existing Java Vector API helpers inherited by the native provider; the L0 neighborhood scorer uses Highway. Both algorithms select NativeVectorizationProvider, but not every helper is a C++ Highway operation. This preparation and the current upper-layer scoring path are included in QPS. No runtime breakdown was profiled, so these results do not assign a measured fraction of query time to projection.

Native ASH neighborhood calls, deferred writer selection and AVX-512 Highway dispatch were confirmed through BenchYAML preflights on both JDKs before the measured runs. After the PCA correction, every non-AsymmetricHashing runtime file was verified identical, including the native provider, FFM wrapper and shared library. The corrected PCA was separately validated through four tall/wide BenchYAML cases on both JDKs. Full runs force the native provider; forcing it fails on provider initialization errors. Normal production ISA dispatch was used, with no benchmark CPU special casing.

Earlier ITQ-trained ASH full-scan comparisons warrant revalidation if they used the affected QR SVD path. The earlier scalar/Panama/native scorer parity results compare scoring of the same codes and are a separate question. Existing saved models are not rewritten by this correction; retraining and rebuilding caches is needed to adopt the corrected PCA.

Raw data: [construction.csv](construction.csv), [queries.csv](queries.csv), [comparisons.csv](comparisons.csv), [raw-results.json](raw-results.json), [summary.json](summary.json). Exact inputs and protocol: [inputs.json](inputs.json), [protocol.json](protocol.json), [source.json](source.json). The evidence archive retains configurations, commands, raw console samples, harness CSV and system metadata.

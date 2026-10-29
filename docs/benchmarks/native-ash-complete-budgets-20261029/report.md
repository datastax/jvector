# Native ASH versus PQ: both MPNet and OpenAI3072 budgets

Includes the previously omitted MPNet 4-bit and OpenAI3072 2-bit native-preparation reruns. ASH uses the native projection/symmetric implementation at 408df9b4 (code at 28e4c8c1), before the queued-batch POC. PQ remains the earlier full-study baseline; it was not recollected. All runs use BenchYAML.

ASH: trained ITQ, C=1, projected dimension D/2. PQ uses exactly matched encoded byte budgets. Deferred parallel writes, fused graph vectors, NVQ reranking, M32, hierarchy and refinement retained. Queries use the same saved graphs as the original study; fresh construction uses separate newly trained models/graphs.

Construction process time includes startup/loading and the standard validation query pass. It is not build-only. Each construction case has one fresh build; no build variance estimate is available.

## Construction: PQ → ASH, seconds

| Dataset | Bits | JDK | Training | Encoding | Graph/write | Process incl. validation |
|---|---:|---:|---:|---:|---:|---:|
| MPNet | 2 | 22 | 2.73 → 5.29 | 1.08 → 1.05 | 44.44 → 54.16 | 52.27 → 67.83 |
| MPNet | 2 | 25 | 2.77 → 5.33 | 1.05 → 0.95 | 33.77 → 34.23 | 41.55 → 46.53 |
| MPNet | 4 | 22 | 6.14 → 4.45 | 1.71 → 1.33 | 50.79 → 41.28 | 62.75 → 54.50 |
| MPNet | 4 | 25 | 4.65 → 4.78 | 1.75 → 1.05 | 48.55 → 36.52 | 58.95 → 48.43 |
| OpenAI3072 | 2 | 22 | 6.51 → 38.29 | 5.51 → 27.36 | 90.50 → 86.74 | 113.08 → 168.85 |
| OpenAI3072 | 2 | 25 | 5.90 → 41.54 | 5.58 → 26.93 | 113.01 → 85.79 | 135.15 → 171.22 |
| OpenAI3072 | 4 | 22 | 19.65 → 38.09 | 7.64 → 25.96 | 257.82 → 101.92 | 295.42 → 185.29 |
| OpenAI3072 | 4 | 25 | 16.62 → 36.40 | 7.78 → 25.75 | 247.53 → 106.24 | 281.56 → 186.01 |

## Query throughput: PQ → ASH, QPS

| Dataset | Bits | JDK | 1 core, 1× | 1 core, 2× | 8 cores, 1× | 8 cores, 2× |
|---|---:|---:|---:|---:|---:|---:|
| MPNet | 2 | 22 | 11,930 → 11,111 | 9,441 → 8,875 | 86,215 → 80,378 | 68,760 → 62,297 |
| MPNet | 2 | 25 | 11,812 → 12,118 | 9,294 → 9,286 | 76,802 → 83,550 | 71,112 → 69,338 |
| MPNet | 4 | 22 | 9,643 → 9,588 | 7,456 → 7,813 | 74,893 → 72,202 | 57,187 → 57,662 |
| MPNet | 4 | 25 | 10,001 → 10,001 | 7,557 → 7,791 | 74,887 → 74,362 | 56,905 → 59,344 |
| OpenAI3072 | 2 | 22 | 3,329 → 1,132 | 2,954 → 1,136 | 27,544 → 8,818 | 22,785 → 8,458 |
| OpenAI3072 | 2 | 25 | 2,170 → 806 | 2,068 → 812 | 27,772 → 7,747 | 23,214 → 7,388 |
| OpenAI3072 | 4 | 22 | 2,144 → 1,026 | 1,871 → 1,046 | 18,467 → 8,259 | 14,336 → 7,927 |
| OpenAI3072 | 4 | 25 | 2,281 → 1,012 | 1,886 → 1,054 | 15,664 → 8,559 | 12,885 → 7,983 |

## Saved-graph recall@10: PQ → ASH, percent

| Dataset | Bits | JDK | 1× | 2× |
|---|---:|---:|---:|---:|
| MPNet | 2 | 22 | 65.128 → 81.940 | 82.126 → 94.055 |
| MPNet | 2 | 25 | 65.484 → 82.017 | 82.210 → 93.971 |
| MPNet | 4 | 22 | 80.733 → 86.642 | 92.753 → 93.966 |
| MPNet | 4 | 25 | 80.751 → 86.293 | 92.601 → 94.033 |
| OpenAI3072 | 2 | 22 | 64.370 → 77.920 | 79.850 → 89.800 |
| OpenAI3072 | 2 | 25 | 65.100 → 76.760 | 80.320 → 89.170 |
| OpenAI3072 | 4 | 22 | 76.910 → 79.030 | 87.700 → 88.340 |
| OpenAI3072 | 4 | 25 | 76.240 → 79.650 | 87.270 → 88.750 |

ASH retains its existing adaptive stopping policy and PQ its existing policy. Equal overquery is not an exact recall match or a quantizer-only comparison. Cohere and Ada are not recollected in this native-preparation subset.

Data: [construction.csv](construction.csv), [queries.csv](queries.csv), [raw-ash-results.json](raw-ash-results.json).

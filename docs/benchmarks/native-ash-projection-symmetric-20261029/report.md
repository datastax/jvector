# Native ASH projection and symmetric construction

Dense graph-query projection and encoded symmetric construction arithmetic now use Highway. Java/Panama and missing-symbol fallbacks remain available. Source and the reference implementation review are described in [design.md](design.md).

Performance was collected through unchanged BenchYAML. Both arms use the native provider and the identical saved graph/model; **before** uses the inherited Panama projection/symmetric helpers and **after** uses the new Highway operations. Query preparation is included. These query runs measure the projection change; the construction pair kernel is exercised by separate fresh builds.

## Query throughput — before → after, QPS

| Dataset / budget | JDK | 1 thread, 1× | 1 thread, 2× | 8 threads, 1× | 8 threads, 2× |
|---|---:|---:|---:|---:|---:|
| MPNet, 2-bit | 22 | 9,534 → 11,111 (1.17×) | 7,996 → 8,875 (1.11×) | 73,300 → 80,378 (1.10×) | 59,441 → 62,297 (1.05×) |
| MPNet, 2-bit | 25 | 10,441 → 12,118 (1.16×) | 8,294 → 9,286 (1.12×) | 76,547 → 83,550 (1.09×) | 61,781 → 69,338 (1.12×) |
| OpenAI3072, 4-bit | 22 | 1,025 → 1,026 (1.00×) | 1,030 → 1,046 (1.02×) | 7,741 → 8,259 (1.07×) | 7,240 → 7,927 (1.09×) |
| OpenAI3072, 4-bit | 25 | 1,062 → 1,012 (0.95×) | 1,040 → 1,054 (1.01×) | 7,844 → 8,559 (1.09×) | 7,464 → 7,983 (1.07×) |

Native QPS was higher in 15/16 matched settings, with a geometric mean ratio
of 1.075. The exception was OpenAI3072/JDK25/one thread/1×: 1,062 → 1,012 QPS,
a 4.7% decrease. The small OpenAI single-thread changes should not be presented
as an established improvement from these three-sample runs.

A separate instrumented, accuracy-only BenchYAML diagnostic alternated both
projection methods for each of 2,000 queries, checked their outputs, and measured
calling-thread CPU time. Median native/Panama projection costs were 0.629/0.626 ms
on JDK22 and 0.628/0.626 ms on JDK25. Thus this diagnostic found similar isolated
projection cost at D=3072. It does not establish the cause of the full-query
4.7% difference or assign a hardware bottleneck. Diagnostic results are excluded
from the throughput tables and retained in [projection-diagnostic.json](projection-diagnostic.json).

## Recall@10 — same saved graph, before → after (%)

| Dataset / budget | JDK | 1× | 2× |
|---|---:|---:|---:|
| MPNet, 2-bit | 22 | 81.940 → 81.940 | 94.055 → 94.055 |
| MPNet, 2-bit | 25 | 82.017 → 82.017 | 93.971 → 93.971 |
| OpenAI3072, 4-bit | 22 | 79.030 → 79.030 | 88.340 → 88.340 |
| OpenAI3072, 4-bit | 25 | 79.650 → 79.650 | 88.750 → 88.750 |

## Fresh construction — current implementation, seconds

Existing BenchYAML phase timers are retained. Process wall time additionally includes startup/loading, one standard 1× recall pass and teardown. The fresh graphs below are separate from the identical saved graphs used for the throughput comparison.

| Dataset / budget | JDK | Training | Encoding | Graph/write | Process incl. validation | Recall@10, 1× (%) |
|---|---:|---:|---:|---:|---:|---:|
| MPNet, 2-bit | 22 | 5.29 | 1.05 | 54.16 | 67.83 | 81.888 |
| MPNet, 2-bit | 25 | 5.33 | 0.95 | 34.23 | 46.53 | 81.633 |
| OpenAI3072, 4-bit | 22 | 38.09 | 25.96 | 101.92 | 185.29 | 78.960 |
| OpenAI3072, 4-bit | 25 | 36.40 | 25.75 | 106.24 | 186.01 | 78.700 |

## Validation and limits

All 60 targeted JDK/ISA regression checks passed. Four traced BenchYAML preflights confirmed actual native projection, symmetric pair and neighborhood calls. Independent ARM NEON and Highway scalar executables each passed 64 arithmetic cases. The same-graph maximum recall change was 0.00000 percentage points. Maximum throughput sample CV was 8.75%.

Query runs use 1 or 8 physical cores; construction uses 48. Native kernels do not create workers or invoke multithreaded BLAS. Matrix preparation is once per immutable model; projection remains once per graph query. New models add rows × columns × 4 bytes of native matrix storage, included in model memory accounting; serialized formats are unchanged.

This focused study does not repeat the entire four-dataset PQ/ASH matrix or establish construction variance. It does not measure standalone pair/LUT kernel throughput in isolation. Native libraries and Java wrappers are frozen per JDK. Earlier development runs and the interrupted build remain archived; they are excluded from these final comparisons.

Data: [queries.csv](queries.csv), [construction.csv](construction.csv), [raw-results.json](raw-results.json). Reproduction: [protocol.json](protocol.json), [source.json](source.json), [validation.json](validation.json), [native-preflight.json](native-preflight.json), [implementation.patch](implementation.patch).

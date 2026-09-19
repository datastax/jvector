## Cohere construction investigation

The first 4-bit Cohere runs were anomalous. Their timings are retained in the primary tables and ranges; they are not silently replaced with faster repeats. The integrated ash-dev code is identical to the frozen ASH production code used for those runs.

| Cohere 4-bit run | ASH graph build/write | PQ graph build/write |
|---|---:|---:|
| Primary repetition 1, no JFR | 658.45 s | 1,224.67 s |
| Primary repetition 2, short JFR capture | 158.05 s | 320.37 s |
| Additional control, no startup JFR | 203.34 s | 350.11 s |

ASH's first/second insertion times were 511.91/89.53 seconds; cleanup 82.69/5.49 seconds; flush 63.85/63.03 seconds. This localizes that anomaly to graph computation rather than the bulk byte-write fix or SSD flush throughput. The extra control's insertion/cleanup/flush times were 105.59/5.58/92.17 seconds.

### What the evidence establishes

- The byte-write fix is present in the actual frozen JAR. No production-code change caused the faster second repetition.
- Source dispatch and live profiles use the intended SIMD kernels for d=512. There is no dimension-based Java dispatch to a scalar ASH kernel. This does not by itself guarantee that every Vector API operation is optimally compiled.
- During the monitored Cohere runs there was ample available memory, no swap, and zero sampled memory-pressure PSI. Each build used a fresh JVM. The pre-resume host had 340 GiB available and no benchmark JVM running. These observations cannot reconstruct the complete system state of the first repetition.
- Full-JVM GC pause totals in the monitored runs were 0.93–2.48 seconds. Large allocation volumes exist, particularly in the 2-bit PQ run, but long stop-the-world pauses do not explain the monitored build costs. JFR shows boxed Vector API objects and array allocation, including NVQ's nvqLoss path. The sampled allocation weights are estimates, not exact per-method allocation accounting.
- The active PQ construction implementation is ImmutablePQVectors' cached centroid-pair lookup. In its 4-bit profile, assembleAndSumPQ_512 was the top frame in 54.1% of execution samples, ThreadLocal.get in 25.2%, and assembleAndSum512 in 14.2%. ASH's 4-bit profile was concentrated in packed asymmetric projection scoring and symmetric packed comparisons. These are sampled CPU-stack proportions, not an exact wall-time decomposition.
- Hardware cycles/instructions/cache counters are not exposed by this VM's perf interface. Therefore these measurements do not distinguish execution throughput from cache/memory stalls precisely. Historical sar files were stale and did not cover the first runs.

### Diagnostic comparison counts

A separate diagnostic class counted actual pair evaluations while preserving the pruning decisions and score arithmetic. It was not added to production code. Cohere-100k is a scale control, not a substitute for measuring the original 1M execution history.

| Method / budget | Prune calls | Candidate checks | Pair comparisons | Diagnostic build/write |
|---|---:|---:|---:|---:|
| ASH 2-bit | 661,994 | 87,840,870 | 1,503,229,262 | 13.27 s |
| ASH 4-bit | 667,048 | 91,887,956 | 1,504,819,172 | 16.91 s |
| PQ matching 2-bit | 669,293 | 82,090,179 | 1,535,016,576 | 16.10 s |
| PQ matching 4-bit | 673,297 | 87,645,603 | 1,539,002,008 | 36.35 s |

Pair counts barely change with bit budget in this control. It supports the expected increase in cost per score, rather than a multiplicative increase in pair count at this scale. It does not identify the cause of the historical 1M outlier.

One initial diagnostic completed its build but failed in post-build reporting because an empty throughput key still enabled the benchmark. The configuration was corrected by omitting that key and the four diagnostics were rerun successfully. The failed attempt remains archived separately and is excluded from the table above.

### Remaining limitation

The first 1M outlier has not yet been causally explained. A missing byte-write fix and a fixed dimensionality fallback are not supported by the evidence. Runtime compilation and execution-state effects remain hypotheses; the original runs lacked the relevant GC/JIT/system traces. The additional controls completed without reproducing the large slowdown. Their full-JVM GC pauses totaled 0.676 seconds for ASH and 1.651 seconds for PQ; available memory stayed above 321/318 GiB, respectively, with zero sampled memory-pressure PSI. Compilation logs recorded class-loading/retry events, but those events alone establish neither a persistent kernel compilation failure nor the cause of the first outlier. No speculative production fix was applied. Do not use the outlying Cohere mean as a precise capacity-planning estimate.

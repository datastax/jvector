# ASH symmetric scorer: final study

Completed on the authoritative remote repository `/home/ted_willke/jvector-ash`, branch `ash-symmetric-scorer`. No deduplication changes.

# Findings and recommendation

Use symmetric ASH for comparisons between already encoded graph nodes. Keep the existing asymmetric scorer for raw-vector insertion and user queries. Both operands of symmetric scoring are already encoded; their earlier projection/encoding is excluded. Asymmetric query projection is timed once per query and reused over that query's candidates.

## Correctness

The old node-to-node implementation ignored the 2/4-bit payload, used folded offsets as though they were original centroid corrections, and returned a score on a different scale from insertion search. Binary tail bits were also handled incorrectly. The replacement follows Appendix B of the final paper at C=1 and preserves the library's normalized dot-product similarity scale.

Standalone scoring supports 1–9 bits, with SIMD single/range kernels for 1/2/4 bits and up to 256 landmarks. The graph provider retains its C=1 restriction. Multiple-landmark scoring uses a tested header-calibrated reconstruction generalization; Appendix B itself only presents C=1. Tests also uncovered and fixed signed-byte indexing for landmark IDs 128–255. The encoded layout is unchanged, but older implementations' C≤64 guard rejects newly written standalone C>64 compressor files.

## Kernel choices

The retained paths are binary XOR/popcount, scalar word and SIMD weighted popcounts for 2-bit pairs, SIMD unpack/multiply for 4-bit pairs, and exact integer LUT block scoring for 2/4 bits. The block implementation borrows the C++ fastscan approach of packed nibble lookup and narrow accumulation. Bounded widening prevents short overflow. It adds no extra score quantization. Experimental alternatives and switches were removed.

A fast isolated SIMD kernel initially allocated boxed Vector API objects when called from graph pruning. Profiling exposed this; splitting kernels by bit width and making species constants explicit removed the dominant boxed allocation pattern. A profiled Ada002-100k build fell from 24.7 to 12.8 seconds; the earlier corrected scalar implementation took 18.2 seconds. These diagnostic timings are separate from the final unprofiled tables.

## Graph evidence

Against the correct, projection-cached asymmetric construction control, symmetric construction was faster on all six measured cases, with slightly higher mean graph recall in each case. At 1M, build plus provider setup averaged:

| Dataset | Cached asymmetric | Symmetric | Recall@10: asymmetric / symmetric |
|---|---:|---:|---:|
| Ada002 | 231.69 s | 157.90 s | 82.462% / 82.669% |
| CAP | 213.31 s | 150.08 s | 78.865% / 78.889% |
| Cohere | 233.05 s | 121.57 s | 69.514% / 69.632% |

The asymmetric cache retains approximately 3.7–5.6 GiB of float arrays at 1M, plus object overhead; symmetric header corrections need approximately 3.8 MiB. No such asymmetric cache was added to production defaults.

These means need qualification. Cohere's cached-asymmetric build varied from about 155 to 308 seconds. Query QPS also varies between independently built graphs/JVMs, even with sustained timing. The table does not establish one universal speedup or statistically significant small recall differences.

Compared with the legacy ash-dev scorer, corrected recall improves on all nine 100k and all three 1M datasets. Construction is not uniformly faster than the legacy implementation, which was not performing the intended distance calculation. At 100k, mean query QPS decreases about 4.0% for E5-small and 3.2% for Gecko and improves for the other seven. These exceptions are retained in the report.

An earlier fully symmetric insertion control gave no consistent build benefit and lost 0.358 percentage points on E5-large versus mixed construction. Keep raw-vector insertion asymmetric. Full-scan symmetric query scoring also incurs additional quantization error; faster encoded-pair arithmetic is not a reason to replace raw-query asymmetric scoring universally.

## Measurement protocol

Graph settings: 32X, 2 bits, C=1, ITQ, M=64, efConstruction=200, overflow=1.2, hierarchy/refinement/fused storage, NVQ reranking, k=10 at 1×, pruning disabled, existing adaptive gamma=0.01. Fresh graph builds reuse identical compressors to isolate scoring/build changes. Build seconds include graph build/write and provider setup, excluding training and initial vector encoding.

Each graph mode has two independent builds. Legacy/candidate order is reversed on the second pass. QPS uses three samples of at least one second each, with actual completed-query counts recorded. The same benchmark-only overlay is applied to both library JARs; baseline library bytes are unchanged. Short-sample exploratory timings are archived separately.

Microbenchmarks run one mode per JVM, one scoring thread, five timing passes, and 1,000 held-out queries for untimed exhaustive recall. Symmetric query encoding is excluded; scorer setup is included. Asymmetric projection is included once per timed query. The 64-target diagnostics measure setup amortization and are not graph QPS.

Tests cover independent decoded-vector formulas, exact packed scalar/SIMD parity, all tail alignments, integer accumulator overflow, zero residuals, negative scores, serialization, unsigned landmark IDs, C=1 graph guards, and standalone C=2/64/256 across all bit widths. SIMD was checked at both 512-bit and forced 256-bit widths on JDK23.


# Final full-scan scorer comparison

One thread, one mode per JVM, 32X compression, C=1, ITQ. Symmetric starts from encoded vectors; previous query encoding is excluded. Asymmetric raw-query projection is included once per query. Scorer setup is included for both. Five timing passes use 128 queries each; held-out exhaustive recall uses 1,000 queries and is outside the timer. Values are million scored vector pairs per second.

| Dataset | Bits | Sym single scalar | Sym single SIMD | Sym block scalar | Sym block SIMD | Asym single SIMD | Asym block SIMD | Sym recall@10 | Asym recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ada002 | 2 | 8.87 | 22.05 | 2.42 | 30.85 | 5.51 | 21.50 | 84.37% | 88.08% |
| cohere_english_v3 | 1 | 71.16 | 103.80 | 83.87 | 84.58 | 18.87 | 12.40 | 69.31% | 78.91% |
| cohere_english_v3 | 2 | 13.81 | 29.76 | 3.95 | 37.98 | 8.05 | 33.08 | 80.45% | 85.30% |
| cohere_english_v3 | 4 | 8.25 | 18.35 | 3.96 | 43.80 | 14.80 | 33.16 | 79.38% | 80.36% |
| e5_large_v2 | 2 | 14.44 | 29.67 | 3.93 | 38.00 | 8.11 | 32.76 | 78.77% | 83.95% |

All four symmetric modes produced identical recall for each dataset/bit-depth case; both asymmetric modes also agreed. Kernel tests independently compare scores and arbitrary padding, rather than relying on recall agreement alone. Symmetric and asymmetric recall are expected to differ because symmetric scoring quantizes both operands. All widths here use the same 32X byte budget: changing bits changes projected dimensionality. These 4-bit results are not a 16X comparison.

Scalar block throughput can be lower than scalar single throughput; block layout is intended to benefit SIMD. No claim is made that every scalar fallback is faster than every asymmetric SIMD path.


# Final graph comparison

Means of two independent builds, with branch order reversed on the second pass. Baseline is ash-dev `67f532a2`, including its legacy incomplete symmetric construction scorer. Candidate uses corrected symmetric node-to-node scoring and existing asymmetric raw-vector insertion/query scoring. No deduplication changes.

32X ASH, 2 bits, C=1, ITQ, M=64, efConstruction=200, NVQ reranking, k=10, 1× overquery, search pruning disabled, existing adaptive gamma=0.01. Build seconds include provider setup plus the existing graph-build/write timer; compressor training and initial vector encoding are excluded. Query throughput uses identical benchmark-only code for both branches, at least one second per sample, three samples per build.

## 100k

| Dataset | Build seconds: old / new | QPS: old / new | Visited: old / new | Recall@10 %: old / new |
|---|---:|---:|---:|---:|
| ada002-100k | 11.77 / 14.58 | 132,120 / 133,631 | 558.5 / 544.3 | 83.897 / 86.375 |
| cohere-english-v3-100k | 11.82 / 12.69 | 222,249 / 226,772 | 649.8 / 644.1 | 81.538 / 82.798 |
| e5-base-v2-100k | 10.11 / 10.89 | 315,885 / 367,709 | 648.2 / 644.9 | 77.808 / 78.931 |
| e5-large-v2-100k | 12.17 / 12.37 | 216,955 / 235,406 | 658.1 / 666.1 | 78.497 / 81.195 |
| e5-small-v2-100k | 8.27 / 9.14 | 585,295 / 561,934 | 719.2 / 691.5 | 59.259 / 60.414 |
| gecko-100k | 9.30 / 11.35 | 362,181 / 350,590 | 660.3 / 673.5 | 78.487 / 80.398 |
| openai-v3-large-1536-100k | 14.67 / 13.87 | 128,633 / 142,892 | 516.4 / 500.9 | 86.616 / 87.943 |
| openai-v3-large-3072-100k | 17.54 / 21.31 | 45,233 / 46,530 | 508.4 / 498.7 | 89.584 / 91.139 |
| openai-v3-small-1536-100k | 12.32 / 14.03 | 133,045 / 149,298 | 541.1 / 533.6 | 85.122 / 86.738 |

## 1M

| Dataset | Build seconds: old / new | QPS: old / new | Visited: old / new | Recall@10 %: old / new |
|---|---:|---:|---:|---:|
| ada002-1M | 128.45 / 157.90 | 122,692 / 134,936 | 773.3 / 745.4 | 79.273 / 82.669 |
| cap-1M | 127.57 / 150.08 | 125,609 / 140,925 | 762.7 / 751.7 | 73.302 / 78.889 |
| cohere-english-v3-1M | 189.94 / 121.57 | 208,169 / 214,930 | 888.3 / 889.6 | 64.447 / 69.632 |

## Build-time ranges across repetitions

| Dataset | Old seconds, range | New seconds, range |
|---|---:|---:|
| ada002-100k | 11.56–11.98 | 14.28–14.89 |
| ada002-1M | 128.02–128.88 | 153.57–162.23 |
| cap-1M | 125.08–130.06 | 144.10–156.05 |
| cohere-english-v3-100k | 11.71–11.92 | 12.45–12.94 |
| cohere-english-v3-1M | 135.08–244.79 | 116.87–126.27 |
| e5-base-v2-100k | 10.06–10.16 | 10.35–11.44 |
| e5-large-v2-100k | 11.38–12.97 | 12.24–12.49 |
| e5-small-v2-100k | 8.09–8.45 | 8.62–9.66 |
| gecko-100k | 8.35–10.25 | 11.32–11.38 |
| openai-v3-large-1536-100k | 13.16–16.19 | 13.84–13.90 |
| openai-v3-large-3072-100k | 16.66–18.41 | 21.09–21.52 |
| openai-v3-small-1536-100k | 10.99–13.65 | 13.96–14.10 |

Cohere-1M baseline build time varies substantially (135–245 seconds); the mean improvement should not be interpreted as a stable fixed speedup. Across the nine 100k datasets, mean query QPS decreases by about 4.0% on E5-small and 3.2% on Gecko; it improves on the other seven. Recall increases on all datasets. Two builds do not establish statistical equivalence for small performance or recall differences.


# Correct symmetric versus asymmetric construction

Two builds per mode. The asymmetric control projects each raw source vector once, caches its existing immutable asymmetric scorer, and uses it for node-ID construction comparisons. The symmetric mode starts from encoded nodes and uses cached header corrections. Both use asymmetric raw-vector insertion and query scoring. Setup is included below. This control avoids attributing repeated projection work to pair-kernel speed. It is a benchmark adapter, not a proposed production cache.

| Dataset | Build seconds: asymmetric / symmetric | Symmetric speedup | Recall@10 %: asymmetric / symmetric | QPS: asymmetric / symmetric | Visited: asymmetric / symmetric |
|---|---:|---:|---:|---:|---:|
| ada002-100k | 19.70 / 14.58 | 1.35× | 86.243 / 86.375 | 143,930 / 133,631 | 546.8 / 544.3 |
| ada002-1M | 231.69 / 157.90 | 1.47× | 82.462 / 82.669 | 108,328 / 134,936 | 751.9 / 745.4 |
| cap-1M | 213.31 / 150.08 | 1.42× | 78.865 / 78.889 | 140,263 / 140,925 | 754.2 / 751.7 |
| cohere-english-v3-100k | 15.41 / 12.69 | 1.21× | 82.754 / 82.798 | 229,465 / 226,772 | 655.7 / 644.1 |
| cohere-english-v3-1M | 233.05 / 121.57 | 1.92× | 69.514 / 69.632 | 231,304 / 214,930 | 887.0 / 889.6 |
| e5-large-v2-100k | 15.70 / 12.37 | 1.27× | 81.089 / 81.195 | 221,213 / 235,406 | 669.0 / 666.1 |

Mean symmetric construction recall is slightly higher in every case (approximately +0.02 to +0.21 percentage points); two builds are insufficient to claim all of those small differences are significant. Query QPS varies with the built graph and JVM: all modes retain the same asymmetric query scorer. In particular, cached-asymmetric Ada002-1M measured roughly 85k and 132k QPS across its two builds, and cached-asymmetric Cohere-1M build time varied from roughly 155 to 308 seconds. Reported means do not remove that uncertainty.

At 1M, the asymmetric control retains approximately 3.7–5.6 GiB of float query-state arrays, plus object/array overhead. The symmetric scorer retains about 3.8 MiB of per-node header corrections at C=1, plus negligible centroid state. Encoded vectors and other shared index memory are excluded from both figures.


# Implementation and provenance

Final implementation commit: `1707d916`. The final graph/scorer measurements use kernel source `1338d46f` with the identical `f98fcf92` throughput benchmark overlay on both branches. The subsequent API commit delegates the previously unimplemented `ASHVectors.diversityFunctionFor` to the tested scorer; the measured graph provider already called that scorer directly. No measured kernel was changed afterward.

The final API regression run passed TestASHSymmetricScorer (7 tests), TestASHGraphSearch (2), TestASHScoringDispatch (4), and TestASHStandaloneVsFusedScoring (3). Earlier scalar/SIMD and forced 256-bit validation is described in [implementation notes](implementation-notes.md). Optional native gtest/Google Benchmark dependencies were unavailable; no native C++ test result is claimed.

Source references:

- Final paper: supplied `2606.07870v2 (3).pdf`, Appendix B (C=1 symmetric derivation).
- C++ reference: `https://github.com/tlwillke/ivf-ash-fastscan`, commit `56d78d0ff3aaaa61a02b94c6f49a97789706a717`; packed lookup and accumulation patterns in `ash_fastscan_avx512.cpp`, header/projection handling in `ash_projection_code.h` and `ash_query.cpp`, LUT setup in `ash_lut.cpp`.
- Java: `ASHSymmetricScorer` (encoded pair scoring and prepared factories), `PanamaVectorUtilSupport` (specialized SIMD kernels), `ASHVectors.diversityFunctionFor` (public API), `AsymmetricHashing` (unsigned landmark handling), `DistancesASH` (isolated timing and score oracle), `ThroughputBenchmark` (sustained samples).

## Commits

| Commit | Change |
|---|---|
| baa1027c | Complete symmetric ASH construction scoring for packed codes |
| f8e64213 | Benchmark prepared symmetric ASH queries in DistancesASH |
| 992f7c6c | Compare ASH construction scoring policies and setup costs |
| cc5d0d1b | Optimize symmetric ASH scoring from encoded vectors |
| 62bc6f6d | Measure ASH scoring with correct query setup and isolated modes |
| 1338d46f | Prevent boxed SIMD allocations during ASH graph pruning |
| f98fcf92 | Measure graph throughput over sustained query batches |
| 1707d916 | Expose symmetric ASH scoring through the diversity API |

## Data and reproduction

- [Graph observations, 48 builds](final-graph-results.csv)
- [Correct asymmetric construction controls, 12 builds](final-asymmetric-controls.csv)
- [Scorer measurements, 30 isolated cases](final-micro-results.csv)
- [Kernel experiments](kernel-experiments.md)
- [Implementation notes](implementation-notes.md)

Remote evidence, frozen JARs, raw logs, profiles, runner configuration and drivers remain under `/mnt/raid10/jvector-bench/ted_willke/ash-symmetric-study`. Local report data is also committed under `docs/ash-symmetric-study`. Existing user YAML files were preserved. No merge or push was performed.

Frozen artifact SHA-256:

```text
22212ed3842676dec1b32558888c178601a5d94ca5a989d828c731f8e7b8762e  /mnt/raid10/jvector-bench/ted_willke/ash-symmetric-study/candidate-final.jar
b19993e9a43a321428e613277afba3de15a9c5cc1bf1085d6579267d8e296f77  /mnt/raid10/jvector-bench/ted_willke/ash-symmetric-study/baseline.jar
40aede39e95da39ed82b6b373acd90554561a7edd5118a45912ae159dc9a44e2  /mnt/raid10/jvector-bench/ted_willke/ash-symmetric-study/throughput-overlay.jar
```

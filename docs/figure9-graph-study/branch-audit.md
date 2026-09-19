# ASH integration audit — 19 September 2026

The authoritative checkout is `/home/ted_willke/jvector-ash` on the benchmark host. `ash-dev` was fast-forwarded from `67f532a2` to `638edc37`. No production code differs from the frozen ASH JAR used in this study. No push was performed. Existing untracked `pr522-bench-*.yml` files were preserved.

| Branch/work | Audited tip | Decision |
|---|---|---|
| ash-symmetric-scorer | 638edc37 | Integrated all 10 commits: completed symmetric math and kernels, boxed-SIMD allocation fix, diversity API, microbenchmarks, sustained query timing, concurrency checks and report. |
| ash-adaptive-integration | 479a61e5 | Already an ancestor of ash-dev. Default gamma 0.01 is included through 28d9620a. |
| ash-training-optimization | 334185a5 | Already included: parallel encoding, divide-and-conquer SVD; subsequent 20D/early-stop default b2555f84 also included. |
| ash-query-performance | 21d2d211 | Already included; subsequent projection-tail fix 2715f71f, shared query setup b1fe8846 and bulk writes 67f532a2 included. |
| ash-neighbor-uniqueness | 638edc37 | Separate worktree, same baseline, no additional changes at audit time. Remains separate. |
| ash-unique-neighbors-integration / ash-unique-neighbors-score-study | 727ec140 | Excluded: unfinished uniqueness and compatibility/version-policy work, including version 8. |
| fix-neighbor-duplicates | 112ec68e | Excluded: PR522 node-ID ordering and related cleanup/format changes. |
| origin/ash-dev-rdevulap | 75fdd2a5 | Not merged: older native 1-bit masked-add/AVX512 tuning; not a validated replacement for current packed multibit Panama kernels. |
| origin/nb-ash-dev | 320f33db | Not merged: alternative multibit driver architecture; current scorer already implements the supported multibit math/kernels. Requires its own evaluation, not an integration cherry-pick. |
| pruning-fix | 6cfc6055 | Not merged: earlier L0 pruning policy; current ash-dev deliberately disables edge pruning. |
| origin/disable-pruning | 11d876cc | Equivalent intended behavior is already present in ash-dev's usePruning API; no old-branch replay. |
| origin/adaptive-term-feature | 2eda0678 | Original design superseded by ASH-specific integrated stopping rules; no merge. |
| origin/bq-adaptive | 63db574f | BQ experiment, not completed ASH work. |
| origin/main | 1a86718b | Frozen production PQ comparator. New file-versioning policy is not pulled into ASH during the separate compatibility discussion. |

## Byte-write fix verification

`67f532a2` replaces the per-byte loop in `FusedASH.writeBytes` with `out.write(bytes, 0, length)`. It is an ancestor of both the original benchmark commit and integrated ash-dev. `javap -p -c` of the actual frozen `study.jar` confirms one `IndexWriter.write:([BII)V` invocation and no per-byte loop. The upstream general vector bulk-write fix `c059e869` is also an ancestor of both ASH and latest-main PQ.

The earlier statement about an absent cleanup fix referred to `804a1d1c` on the deduplication branch, not the byte-write fix. A small independently reproduced NodeArray duplicate-at-capacity patch was set aside after the user clarified the intended fix. It is **not applied**; the source is clean apart from this documentation. The preserved patch is outside the repository at `ash-integration-cleanup-unapplied.patch`.

## Stashes

`preserve-asymmetric-score-study` contains an old diagnostic mode superseded by the completed construction-scoring experiment. `preserve-DistancesPQ-before-PR522` contains additional full-scan PQ-mode instrumentation whose validation was paused; it remains preserved and is not treated as finished production work. Neither stash was applied or deleted.

## Validation

JDK23 Maven package completed successfully on the final integrated source. Seven focused classes passed in scalar and SIMD executions, 33 tests per execution: TestNodeArray, TestNeighbors, GraphIndexBuilderTest, TestASHSymmetricScorer, TestASHGraphSearch, TestASHScoringDispatch and TestASHStandaloneVsFusedScoring. Authoritative log: `/mnt/raid10/jvector-bench/ted_willke/ash-dev-integrated-tests.log`.

Completed benchmark observations and indexes remain valid: the integration moved a branch pointer to the already-tested source. The interrupted CAP 2-bit ASH second attempt is archived separately and is rebuilt for a complete measurement set. Cohere is prioritized on resume, with GC logs, system sampling and short JFR profiles; those profiled timings are identified in the final analysis.

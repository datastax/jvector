# Design: Multi-Index (Sharded) Search

Status: Phases 1, 2, and 3 are all implemented (`MultiGraphSearcher`, `ShardedSearchResult`,
`MultiGraphSearcher.OverqueryStrategy`, `MultiGraphSearcher.Builder`, the `MultiShardBench`
recall/latency benchmark, and the `multisearch` tutorial example). The implementation deviates from
the API and algorithm originally sketched below in several ways, recorded here so this document stays
a reference for what actually shipped:

- **`resume()` accumulates, it doesn't replace.** `GraphSearcher.resume(int, int)` returns only the
  *newly*-discovered nodes since the previous `search()`/`resume()` call, not a replacement/superset
  of what was already returned -- so `MultiGraphSearcher` accumulates each shard's nodes across
  rounds (append), rather than replacing a shard's prior-round result with its latest one. A first
  pass at this feature assumed "replaces", which is wrong and was caught by a randomized test failure
  (the merged top-K silently dropped a shard's already-found best candidates once that shard was
  resumed).
- **The resume trigger is shortfall-based, not cutoff-based.** The algorithm sketched below (see the
  original "Decide which shards need another round" step) resumes a shard once it has exhausted its
  ask *and* its worst returned score is still `>=` the current global cutoff. That was tried and
  abandoned: once a shard's own results make up part of the current top-K, its worst-returned score is,
  by construction, close to -- or literally part of -- the cutoff it's being compared against, so the
  check fired on nearly every call and made a single `search()` cost close to an exhaustive scan
  instead of a bounded approximate search. The shipped trigger instead resumes only when the *merged
  candidate count* falls short of the requested `topK` and the shard in question wasn't exhausted -- a
  narrower, unambiguous condition. See `MultiGraphSearcher`'s class javadoc for the full rationale.
- **A second resume trigger was added after a real recall regression was found via `MultiShardBench`.**
  `proportionalShare()` rounds `topK_i` and `rerankK_i` independently for each shard. At small
  per-shard shares (shard count large relative to `topK`), both can round to the *same* integer for
  different `rerankK` values -- e.g. with 16 equal shards and `topK=10`, `round(10/16)` and
  `round(20/16)` both equal 1 -- silently erasing a shard's share of the caller's requested overquery
  ratio. Worse, once shard count exceeds `topK`, the shortfall trigger above is satisfied trivially on
  round 1 (`numShards` candidates alone already `>= topK`), so resume never fired to correct this,
  meaning raising `rerankK` had *zero* effect on the result. Measured on `ada002-100k` at
  `numShards=16, topK=10`: recall@10 was identical (0.5113) at overquery 1.0 and 2.0 before the fix.
  `MultiGraphSearcher` now also flags a shard when its rounded `rerankK_i` undershoots
  `round(topK_i * (rerankK/topK))` -- its fair share of the caller's requested ratio -- and forces a
  correcting resume round even when the shortfall trigger alone would have stopped. Same benchmark
  after the fix: recall@10 at overquery 2.0 jumped from 0.5113 to 0.8436 (and to 0.94/0.97 at
  overquery 5.0/10.0, up from 0.66/0.71), at the cost of roughly double the latency at those settings
  -- the correct tradeoff, since the old latency was cheap only because it was silently skipping work
  the caller had explicitly asked for.
- **`rerankFloor` is not threaded through rounds.** Since the cutoff-based trigger above was dropped,
  Astra's "use last round's `worstApproximateScoreInTopK` as this round's `rerankFloor`" technique
  doesn't apply either -- every round asks with `rerankFloor` effectively 0 (via the public
  `resume(additionalK, rerankK)`), at the cost of some redundant exact-reranking work. This is a
  closed decision, not an open item carried into future tuning: reintroducing `rerankFloor` would mean
  reintroducing the same cutoff comparison that was already found to misfire.
- **Construction is a `Builder`, not constructor overloads.** `MultiGraphSearcher(List<...> shards)`
  is the only public constructor; the executor, `OverqueryStrategy`, and `maxResumeRounds` are
  configured via `MultiGraphSearcher.builder(shards)....build()` rather than the three constructor
  overloads originally sketched. Unlike the original sketch, there is no default executor -- fan-out is
  sequential unless a caller explicitly supplies one via `Builder.withExecutor`.
- **A new caller-facing `resume(int additionalK)` was added**, not in the original API sketch. It lets
  a lazy consumer that does its own downstream filtering (e.g. Cassandra reconciling stale/duplicate
  rows across sstables) grow an already-returned result set without restarting the whole multi-shard
  query -- see the Proposed API section below.

## Motivation

jvector users routinely split one logical dataset across several physical `ImmutableGraphIndex`
instances — one per Lucene segment, one per Cassandra sstable, one per time-bucketed shard — and
need to answer a single top-K query against all of them combined. jvector has no in-library
support for this today; every caller re-derives the same fan-out/merge logic on its own.

This isn't a hypothetical gap. `GraphSearcher` already carries machinery that exists *specifically*
because of this use case. From the comment above `searchOneLayer` (`GraphSearcher.java:386-404`):

> Astra breaks logical indexes up across multiple physical `OnDiskGraphIndex` pieces, one per
> sstable. Each of these pieces is searched independently, and the results are combined... Astra
> will look at the `worstApproximateInTopK` value from the first ODGI, and use that as the
> `rerankFloor` for the next... `resume()` also drives the use of `CachingReranker`.

In other words: `rerankFloor`, `resume(int, int)`, and `SearchResult.getWorstApproximateScoreInTopK()`
were added *for* multi-index search, and are already exercised in production outside this repo.
What's missing is a packaged, reusable orchestrator — the equivalent of FAISS's `IndexShards` —
instead of every embedder hand-rolling the fan-out/merge loop against these low-level primitives.

## Goals

- Fan a single query out across N independent `ImmutableGraphIndex` shards and return a single
  merged top-K, ranked by score.
- Reuse the existing `resume()`/`rerankFloor` machinery rather than reinventing per-shard search.
- Support optional parallel fan-out across shards (they're independent — no shared mutable state).
- Ship as an additive, opt-in API. No existing class's behavior changes for callers who don't use it.

## Non-goals (explicitly out of scope for this proposal)

- **No cross-shard deduplication by external identity.** jvector has no concept of a stable
  "row key" — only per-shard ordinals, which collide across shards (ordinal 5 in shard A and
  ordinal 5 in shard B are unrelated vectors). If the embedding application needs to dedupe
  logical rows that appear in more than one shard (e.g. Cassandra rows updated across sstables),
  that mapping only the caller has, and must happen after `ShardedSearchResult` comes back.
- **Not an index-merge feature.** This is query-time fan-out only. Actually merging multiple
  on-disk indexes into one is already `OnDiskGraphIndexCompactor` (see `docs/compaction.md`); this
  proposal is for when you deliberately *don't* want to merge (e.g. per-segment indexes that are
  still being written).
- **Not cross-process/distributed.** In-process, same-JVM fan-out only. Cross-node distribution is
  the embedding application's job (that's how Astra/Cassandra already use jvector today).
- **No on-disk format changes.** Nothing here touches `GraphIndexFormat`/serialization.
- **Assumes comparable scores across shards.** All shards must use the same
  `VectorSimilarityFunction`, and for meaningful merging every shard's `SearchScoreProvider` should
  rerank with an *exact* score function. Merging purely-approximate (PQ-only, no reranker) scores
  across shards isn't recommended — independently trained PQ codebooks per shard can have different
  approximation bias, so "score 0.82 from shard A" and "score 0.82 from shard B" aren't guaranteed
  to mean the same thing unless both were exactly reranked.

## Implemented API

New classes only, both in `io.github.jbellis.jvector.graph` (same package as `GraphSearcher`, for a
reason explained below):

```java
public class MultiGraphSearcher implements AutoCloseable {
    public MultiGraphSearcher(List<? extends ImmutableGraphIndex> shards);

    public static Builder builder(List<? extends ImmutableGraphIndex> shards);

    public int shardCount();
    public ImmutableGraphIndex.View getView(int shardIndex);

    /**
     * @param scoreProviders     one per shard, in shard order; each closes over the same query
     *                           vector but the shard's own vectors/compressor
     * @param acceptOrdsPerShard one Bits per shard (shard-local ordinals); pass Bits.ALL per shard
     *                           if there's no per-query filter
     * @param topK               desired global result count
     * @param rerankK            global rerank budget, split proportionally across shards by the
     *                           OverqueryStrategy
     */
    public ShardedSearchResult search(List<SearchScoreProvider> scoreProviders,
                                       List<Bits> acceptOrdsPerShard,
                                       int topK,
                                       int rerankK);

    // convenience overload: Bits.ALL for every shard
    public ShardedSearchResult search(List<SearchScoreProvider> scoreProviders, int topK, int rerankK);

    /**
     * Grows the current query's result count by additionalK, reusing whatever shard results are
     * already accumulated and asking shards to search further only if needed. For lazy consumers
     * (e.g. Cassandra reconciling stale/duplicate rows) that come up short after downstream
     * filtering and want more results without restarting the whole multi-shard query. Must be
     * called after search().
     */
    public ShardedSearchResult resume(int additionalK);

    @Override
    public void close() throws IOException; // closes each shard's GraphSearcher/View

    @FunctionalInterface
    public interface OverqueryStrategy {
        int initialRerankKFor(int shardIndex, long shardSize, long totalSize, int topK, int globalRerankK);
        OverqueryStrategy DEFAULT = ...; // proportional-by-size split of globalRerankK
    }

    public static final class Builder {
        public Builder withExecutor(ExecutorService executor);          // default: sequential fan-out
        public Builder withOverqueryStrategy(OverqueryStrategy strategy); // default: OverqueryStrategy.DEFAULT
        public Builder withMaxResumeRounds(int maxResumeRounds);          // default: 2 (3 rounds total)
        public MultiGraphSearcher build();
    }
}
```

```java
public final class ShardedSearchResult {
    public static final class NodeScore implements Comparable<NodeScore> {
        public final int shardIndex;  // index into the shards list passed to the constructor
        public final int node;        // ordinal, local to shard `shardIndex`
        public final float score;
    }

    public NodeScore[] getNodes();           // best-first, size <= topK
    public int getVisitedCount();            // summed across shards, across all rounds
    public int getExpandedCount();
    public int getRerankedCount();
    public int getRoundsUsed();              // 1 = no resume was needed
}
```

A shard is just an `ImmutableGraphIndex` — no new wrapper type. `MultiGraphSearcher` owns one
internal `GraphSearcher` per shard (constructed once, reused across calls to `search`/`resume`,
exactly like a single `GraphSearcher` is today). Not safe for concurrent use by multiple threads —
scratch state is reused across calls, same as `GraphSearcher`.

## Algorithm

### Why placing this in `io.github.jbellis.jvector.graph` matters

`GraphSearcher.resume(int additionalK, int rerankK)` is public but hardcodes `threshold=0` and
`rerankFloor=0`. The version that actually accepts a `rerankFloor` —

```java
SearchResult resume(int topK, int rerankK, float threshold, float rerankFloor)  // package-private
```

— is package-private. Putting `MultiGraphSearcher` in the same package lets it call that overload
directly. In practice `MultiGraphSearcher` always calls it with `rerankFloor=0` (see the status
section above for why the Astra "feed the previous cutoff back as `rerankFloor`" technique isn't
used), but package placement is still what makes `resume(topK, rerankK, threshold, rerankFloor)`
reachable at all without changing `GraphSearcher`'s public surface.

### Steps, as implemented

1. **Initial fan-out.** For each shard `i` with size `size_i` (from `index.size(0)`) and
   `totalSize = Σ size_i`:
   - `topK_i = max(1, round(topK * size_i / totalSize))`
   - `rerankK_i = max(topK_i, OverqueryStrategy.initialRerankKFor(i, size_i, totalSize, topK, rerankK))`
     (default `OverqueryStrategy`: the same proportional-by-size split, applied to `rerankK` instead
     of `topK`)
   - Call `shard[i].searcher.search(scoreProviders.get(i), topK_i, rerankK_i, 0f, 0f, acceptOrds.get(i))`.
   - Dispatch is sequential on the calling thread by default. If a caller supplied an
     `ExecutorService` via `Builder.withExecutor`, each shard's call is submitted to it and the round
     blocks on `Future.get()` for all of them. No executor is chosen automatically -- there's no
     built-in default pool (see "A note on parallel fan-out" below for why not).

2. **Merge.** Collect every shard's `SearchResult.NodeScore[]` accumulated so far (see step 4),
   tag each with its `shardIndex`, sort descending by score, take the global top `topK`.

3. **Decide which shards need another round.** Fires for either of two reasons (see the status
   section above for how the second one was found):
   - **Shortfall:** the merge in step 2 has *fewer than `topK`* total candidates.
   - **Rounding undid the requested overquery ratio:** step 1's independent rounding of `topK_i` and
     `rerankK_i` left some shard's own ratio short of `rerankK/topK` (the ratio the caller actually
     asked for).

   When either fires, every shard that returned exactly its most-recently-asked count (i.e. wasn't
   exhausted -- returning fewer than asked means there's nothing left in that shard, regardless of
   score) is a resume candidate. A shard flagged for the second reason but *not* a resume candidate
   (i.e. genuinely exhausted) has its flag cleared instead of being retried -- there's nothing more to
   search, so it can never be corrected. There is no per-shard score/cutoff comparison; see the status
   section above for why that check was dropped.

4. **Resume qualifying shards.** For each, call the package-private `resume(budget, budget, 0f, 0f)`
   where `budget` is the shard's previous budget grown by a fixed 2x growth factor
   (`Math.max(budget + 1, round(budget * 2.0))`, guaranteeing forward progress even at `budget=0`).
   `resume()`'s result is the *newly*-discovered nodes only (not a replacement), so
   `MultiGraphSearcher` appends them to that shard's accumulated result rather than overwriting it.

5. **Repeat from step 2** until the merge has `topK` candidates, or `maxResumeRounds` additional
   rounds have been used (default: 2, so 3 rounds total; configurable via
   `Builder.withMaxResumeRounds`, where 0 disables resume entirely).

`resume(int additionalK)` (the caller-facing method) re-enters this same loop with a larger target
`topK` (the previous `topK + additionalK`), reusing every shard's already-accumulated results instead
of starting over.

### A note on parallel fan-out

The `Builder.withExecutor` path is implemented and unit-tested (`testParallelExecutorMatchesSequentialResult`
confirms it merges to the same result as sequential dispatch), but `MultiShardBench` -- the benchmark
that exercises `MultiGraphSearcher` under realistic concurrent load -- deliberately does *not* use it.
The benchmark gets its concurrency from running many queries in parallel (mirroring how `Grid` measures
QPS) with one `MultiGraphSearcher` per worker thread; nesting a second, bounded, shared pool underneath
that for intra-query shard fan-out would risk starvation, since `dispatchInParallel` blocks on a plain
`Future.get()` with no `ForkJoinPool` join-compensation. Intra-query fan-out parallelism is therefore
best suited to a caller issuing few, latency-sensitive queries at a time (one query, N shards
in parallel) rather than a high-QPS workload that's already parallel across queries.

### A note on completeness

This is still an approximate search — resuming reduces the chance that a small/oddly-distributed
shard was shortchanged by the initial proportional sizing, but it's a heuristic, not a proof. That
matches jvector's existing single-index guarantees (which are also approximate) and is a deliberate
scope choice, not an oversight.

## Implementation phases

1. **Phase 1 — fixed overquery, no resume.** Shipped, then superseded by Phase 2's proportional
   sizing + resume loop below (Phase 1's fixed-overquery behavior is still reachable via
   `Builder.withMaxResumeRounds(0)`, which disables resume and leaves a single proportional-sizing
   round with no adaptive refill).
2. **Phase 2 — proportional sizing + resume loop.** Implemented: `OverqueryStrategy`, the
   size-proportional default, and the resume loop described above -- shortfall-triggered plus the
   rounding-underserved correction added after `MultiShardBench` surfaced it (neither is the
   cutoff-based loop originally sketched for this phase -- see the status section).
3. **Phase 3 — parallel fan-out + tuning.** Implemented: `Builder.withExecutor` for optional
   intra-query parallel fan-out, and `MultiShardBench` (wired through `BenchYAML`, reusing `Grid`'s
   construction/search parameter grid plus a `shardCounts` axis) as the recall/latency comparison
   against a single unsharded index. Default growth factor (2x) and `maxResumeRounds` (2) are the
   originally-sketched defaults; empirical tuning against `MultiShardBench` results against real
   datasets, beyond what its own test coverage exercises, is still open -- see Open questions.

## Classes touched

| Class | Change | Why |
|---|---|---|
| `MultiGraphSearcher` (new) | new file, `graph` package | orchestrator |
| `MultiGraphSearcher.Builder` (new) | new, nested in above | configures executor/`OverqueryStrategy`/`maxResumeRounds`; not in the original sketch, which used constructor overloads instead |
| `MultiGraphSearcher.OverqueryStrategy` (new) | new, nested in above | pluggable sizing policy |
| `ShardedSearchResult` (new) | new file, `graph` package | merged result type carrying `shardIndex` |
| `ShardedSearchResult.NodeScore` (new) | new, nested in above | per-result shard tagging |
| `GraphSearcher` | **none required** | `MultiGraphSearcher` reuses the existing package-private `resume(topK, rerankK, threshold, rerankFloor)` by being in the same package (called with `rerankFloor=0` always -- see status section) |
| `SearchResult` | **none required** | `getWorstApproximateScoreInTopK()` exists but is unused by the shipped resume trigger |
| `ImmutableGraphIndex` | **none required** | `size(0)` and `getView()` already available |
| `Bits` | **none required** | `Bits.ALL`/`Bits.intersectionOf` already handle per-shard filtering; liveness is already intersected in automatically by each shard's own `GraphSearcher.initializeInternal` |
| `PhysicalCoreExecutor` | **none required, and not used automatically** | `Builder.withExecutor` accepts any caller-supplied `ExecutorService`; there's no built-in default pool, and `MultiShardBench` deliberately doesn't wire one in (see "A note on parallel fan-out") |
| `jvector-examples` (`MultiShardBench`, `MultiSearchExample`, `TutorialRunner`) | new files/edit | recall/latency benchmark against a single-index baseline, plus a runnable tutorial (`TutorialRunner` case `"multisearch"`) |

The headline result: **this shipped without modifying any existing public class.** Everything it
needs from `GraphSearcher` is either already public (`resume(int, int)`, though `MultiGraphSearcher`
bypasses it for the package-private 4-arg overload) or already accessible by virtue of package
placement.

## Impact on existing library users

- **Zero required changes for existing callers.** Nobody using `GraphSearcher`, `SearchResult`, or
  single-index search today needs to change anything — this is a pure addition.
- **New API surface to learn**, if opted into: `MultiGraphSearcher`, `MultiGraphSearcher.Builder`,
  `ShardedSearchResult`, and the new `ShardedSearchResult.NodeScore` (which is *not* interchangeable
  with `SearchResult.NodeScore` — it carries a `shardIndex` and callers need to route to the right
  shard's vectors/row-mapping using it).
- **New constraint to document, not enforce in code:** callers must supply one
  `SearchScoreProvider` and one `Bits` per shard, matching shard order — there's no runtime check
  that a caller didn't accidentally swap two shards' score providers, since jvector has no way to
  know they're mismatched. Worth a clear javadoc warning.
- **Marked `@Experimental`**, consistent with how `resume`/`rerankFloor` are already annotated in
  `GraphSearcher` — this is genuinely new, unproven-in-this-repo surface (even though the underlying
  primitives are proven externally), and marking it experimental leaves room to adjust the API shape
  (especially `OverqueryStrategy` and the default constants) after real usage.
- **No version/format bump needed** — this doesn't touch serialization, so it doesn't interact at
  all with the `GraphIndexFormat` versioning work.

## Testing plan

- Unit tests (`TestMultiGraphSearcher`, small synthetic 2-3 shard setups using
  `TestVectorGraph.CircularFloatVectorValues`) verifying:
  - Merged top-K matches brute-force merge of independently-run single-shard searches
    (`testMergesAcrossShards`).
  - Resume logic actually triggers when a shard is undersized relative to its data (one tiny shard,
    one large shard, query near the tiny shard's data) and recovers the exact expected top-K
    (`testResumeRecoversAcrossSkewedShardSizes`).
  - Exhausted shards (fewer live/accepted nodes than requested) don't spuriously trigger resume
    (`testExhaustedShardDoesNotTriggerSpuriousResume`: a single shard hard-capped via `acceptOrds` to
    3 live candidates against a `topK` of 50 must stop after round 1 -- asserted both on
    `getRoundsUsed()` and, via a task-counting `ExecutorService`, on the shard never being
    re-dispatched for the `maxResumeRounds` it would otherwise be eligible for).
  - `close()` releases all per-shard resources: `testCloseClosesAllShardSearchers` wraps two shards'
    `ImmutableGraphIndex`/`View` in close-tracking delegates, constructs `MultiGraphSearcher` over
    them, and asserts both shards' views are closed only after `MultiGraphSearcher.close()` is
    called, mirroring `GraphSearcher`'s own close tests.
  - Caller-driven `resume(int)` growth, per-shard accept-ords, the convenience `Bits.ALL` overload,
    parallel-executor fan-out matching sequential results, and constructor/builder equivalence
    (`testCallerResumeGrowsResultCount`, `testConvenienceOverloadMatchesExplicitAcceptAll`,
    `testRespectsPerShardAcceptOrds`, `testParallelExecutorMatchesSequentialResult`,
    `testBuilderWithoutExecutorBehavesLikeConstructor`).
  - Overquery keeps having an effect as shard count grows relative to `topK`
    (`testOverqueryKeepsMatteringAsShardCountGrows`): reproduces the 16-shard/`topK=10` case that
    `MultiShardBench` caught, asserting no correction round fires at overquery 1.0 (nothing was lost to
    rounding) but one does fire at overquery 2.0 (`round(20/16)` collapsing to the same integer as
    `round(10/16)` would otherwise go uncorrected).
- A recall/latency benchmark, `MultiShardBench` (driven via `BenchYAML`), comparing single merged
  index vs. N-shard `MultiGraphSearcher` over the same data and construction/search configuration, at
  a few shard counts.

## Open questions

- Empirical tuning of the default growth factor (2x) and `maxResumeRounds` (2) against
  `MultiShardBench` results on real (non-synthetic) datasets hasn't been done yet -- the current
  defaults are the ones originally sketched, not ones derived from benchmark data. This interacts with
  the rounding-underserved correction above: a shard whose ratio was badly undershot (e.g. a caller
  requesting 10x overquery against 16+ shards) may need several 2x growth steps to actually reach its
  fair share, bounded by `maxResumeRounds` -- confirmed on `ada002-100k` at `numShards=16, topK=10`,
  where recall kept improving from overquery 2.0 through 10.0 rather than saturating immediately,
  suggesting the correction is still growth-factor-limited rather than reaching the ideal fair share in
  one round for large ratios.
- Is a fixed `growthFactor` (doubling) the right default, or should it also scale with how far short
  of `topK` the merge came up, or how far a shard's rounded ratio undershot its fair share? Needs
  empirical tuning.
- Should `ShardedSearchResult` expose per-shard sub-results (not just the flattened merged list),
  for callers who want to inspect per-shard contribution? Leaning no for v1 — keep the surface
  minimal — but flagging it since Astra-style consumers might want it for diagnostics.
- Should `resume()`'s `rerankFloor`-accepting overload eventually become public on `GraphSearcher`
  itself, for consumers who want to build their own multi-index logic without being confined to
  jvector's package? Lower priority than originally scoped, now that `MultiGraphSearcher` itself
  doesn't use a non-zero `rerankFloor` -- but still worth revisiting if there's demand from a caller
  with a different resume strategy in mind.

# Design: Multi-Index (Sharded) Search

Status: proposal — not yet implemented. Estimates and class lists below are expected to shift once
implementation starts; treat this as a starting point for review, not a spec.

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

## Proposed API

New classes only, both in `io.github.jbellis.jvector.graph` (same package as `GraphSearcher`, for a
reason explained below):

```java
public class MultiGraphSearcher implements AutoCloseable {
    public MultiGraphSearcher(List<? extends ImmutableGraphIndex> shards);
    public MultiGraphSearcher(List<? extends ImmutableGraphIndex> shards, Executor executor);
    public MultiGraphSearcher(List<? extends ImmutableGraphIndex> shards, Executor executor, OverqueryStrategy strategy);

    /**
     * @param scoreProviders   one per shard, in shard order; each closes over the same query
     *                         vector but the shard's own vectors/compressor
     * @param acceptOrdsPerShard one Bits per shard (shard-local ordinals); pass Bits.ALL per shard
     *                         if there's no per-query filter
     * @param topK             desired global result count
     * @param rerankK          global rerank budget, divided across shards by the OverqueryStrategy
     */
    public ShardedSearchResult search(List<SearchScoreProvider> scoreProviders,
                                       List<Bits> acceptOrdsPerShard,
                                       int topK,
                                       int rerankK) throws IOException;

    // convenience overload: Bits.ALL for every shard
    public ShardedSearchResult search(List<SearchScoreProvider> scoreProviders, int topK, int rerankK) throws IOException;

    @Override
    public void close() throws IOException; // closes each shard's GraphSearcher/View

    @FunctionalInterface
    public interface OverqueryStrategy {
        int initialRerankKFor(int shardIndex, long shardSize, long totalSize, int topK, int globalRerankK);
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
    public int getVisitedCount();            // summed across shards
    public int getExpandedCount();
    public int getRerankedCount();
    public int getRoundsUsed();              // 1 = no resume was needed
}
```

A shard is just an `ImmutableGraphIndex` — no new wrapper type. `MultiGraphSearcher` owns one
internal `GraphSearcher` per shard (constructed once, reused across calls to `search`, exactly like
a single `GraphSearcher` is today).

## Algorithm

### Why placing this in `io.github.jbellis.jvector.graph` matters

`GraphSearcher.resume(int additionalK, int rerankK)` is public but hardcodes `threshold=0` and
`rerankFloor=0`. The version that actually accepts a `rerankFloor` —

```java
SearchResult resume(int topK, int rerankK, float threshold, float rerankFloor)  // package-private
```

— is package-private. Putting `MultiGraphSearcher` in the same package lets it call that overload
directly, which is what makes Astra's documented technique ("use the previous round's
`worstApproximateScoreInTopK` as the next round's `rerankFloor`") actually usable. This is also why
**no changes to `GraphSearcher`'s public surface are required** — see the impact section below.

### Steps

1. **Initial fan-out.** For each shard `i` with size `size_i` (from `index.size(0)`) and
   `totalSize = Σ size_i`:
   - `topK_i = max(1, round(topK * size_i / totalSize))`
   - `rerankK_i = OverqueryStrategy.initialRerankKFor(i, size_i, totalSize, topK, rerankK)`
     (default: `max(topK_i, round(topK_i * 2.0))`, i.e. the same fixed 2x overquery factor already
     used ad hoc throughout `jvector-examples`)
   - Call `shard[i].searcher.search(scoreProviders.get(i), topK_i, rerankK_i, 0f, 0f, acceptOrds.get(i))`.
   - This round is embarrassingly parallel — dispatch via the configured `Executor` (default:
     `PhysicalCoreExecutor.instance().pool()`, already exists, no new infra) and join.

2. **Merge.** Collect every shard's `SearchResult.NodeScore[]`, tag each with its `shardIndex`, sort
   descending by score, take the global top `topK`. Compute `cutoff` = score of the worst entry
   kept (or `-∞` if fewer than `topK` total candidates exist across all shards).

3. **Decide which shards need another round.** A shard is a resume candidate if **both**:
   - it returned exactly its requested `rerankK_i` entries (fewer means it's exhausted — there's
     nothing left to find, regardless of score), **and**
   - its own worst returned score `getWorstApproximateScoreInTopK()` is `>= cutoff` (otherwise every
     unreturned candidate in that shard — necessarily scored no better than what was already
     returned — would fall below the cutoff too, so there's nothing to gain).

4. **Resume qualifying shards.** For each, call the package-private
   `resume(rerankK_i * growthFactor, rerankK_i * growthFactor, 0f, cutoff)` — using the current
   global `cutoff` as `rerankFloor`, per Astra's technique, so shards don't waste time exactly
   reranking candidates that can no longer make the cut. `resume()`'s result **replaces** (not
   unions with) that shard's previous result — it already subsumes prior candidates plus the newly
   explored ones.

5. **Repeat from step 2** until no shard qualifies for resume, or `maxRounds` is hit (default: 2
   resume rounds after the initial one — i.e. 3 rounds total — configurable).

### A note on completeness

This is still an approximate search — resuming reduces the chance that a small/oddly-distributed
shard was shortchanged by the initial proportional sizing, but it's a heuristic, not a proof. That
matches jvector's existing single-index guarantees (which are also approximate) and is a deliberate
scope choice, not an oversight.

## Suggested implementation phases

Given this is new ground, I'd land it in stages rather than as one PR:

1. **Phase 1 — fixed overquery, no resume.** Every shard gets the same fixed `rerankK_i` (no
   proportional sizing, no resume loop). Simplest possible correct implementation; validates the
   merge logic and API shape before adding adaptive complexity.
2. **Phase 2 — proportional sizing + resume loop.** Add `OverqueryStrategy`, the size-proportional
   default, and the resume-based refill described above.
3. **Phase 3 — parallel fan-out + tuning.** Add the `Executor` constructor overload and tune
   default growth factor / `maxRounds` against real recall/latency benchmarks (the existing
   `jvector-examples`/`Grid` harness is the natural place to measure this).

## Classes touched

| Class | Change | Why |
|---|---|---|
| `MultiGraphSearcher` (new) | new file, `graph` package | orchestrator |
| `MultiGraphSearcher.OverqueryStrategy` (new) | new, nested in above | pluggable sizing policy |
| `ShardedSearchResult` (new) | new file, `graph` package | merged result type carrying `shardIndex` |
| `ShardedSearchResult.NodeScore` (new) | new, nested in above | per-result shard tagging |
| `GraphSearcher` | **none required** | `MultiGraphSearcher` reuses the existing package-private `resume(topK, rerankK, threshold, rerankFloor)` by being in the same package |
| `SearchResult` | **none required** | `getWorstApproximateScoreInTopK()` already public and sufficient |
| `ImmutableGraphIndex` | **none required** | `size(0)` and `getView()` already available |
| `Bits` | **none required** | `Bits.ALL`/`Bits.intersectionOf` already handle per-shard filtering; liveness is already intersected in automatically by each shard's own `GraphSearcher.initializeInternal` |
| `PhysicalCoreExecutor` | **none required** | `instance().pool()` already exposed as a reusable `ForkJoinPool` |
| `jvector-examples` (`Grid`/`Bench`) | optional, new example | no existing example currently demonstrates `resume`/`rerankFloor` at all — worth adding one alongside this feature |

The headline result: **this can ship without modifying any existing public class.** Everything it
needs from `GraphSearcher` is either already public (`resume(int, int)`, though we bypass it for
the package-private 4-arg overload) or already accessible by virtue of package placement.

## Impact on existing library users

- **Zero required changes for existing callers.** Nobody using `GraphSearcher`, `SearchResult`, or
  single-index search today needs to change anything — this is a pure addition.
- **New API surface to learn**, if opted into: `MultiGraphSearcher`, `ShardedSearchResult`, and the
  new `ShardedSearchResult.NodeScore` (which is *not* interchangeable with `SearchResult.NodeScore`
  — it carries a `shardIndex` and callers need to route to the right shard's vectors/row-mapping
  using it).
- **New constraint to document, not enforce in code:** callers must supply one
  `SearchScoreProvider` and one `Bits` per shard, matching shard order — there's no runtime check
  that a caller didn't accidentally swap two shards' score providers, since jvector has no way to
  know they're mismatched. Worth a clear javadoc warning.
- **Recommend `@Experimental`** on the new classes initially, consistent with how `resume`/
  `rerankFloor` are already annotated in `GraphSearcher` — this is genuinely new, unproven-in-this-
  repo surface (even though the underlying primitives are proven externally), and marking it
  experimental leaves room to adjust the API shape (especially `OverqueryStrategy` and the default
  constants) after real usage.
- **No version/format bump needed** — this doesn't touch serialization, so it doesn't interact at
  all with the `GraphIndexFormat` versioning work.

## Testing plan

- Unit tests with small synthetic multi-shard setups (2-5 shards, in-memory `OnHeapGraphIndex` or
  `TestUtil` graph helpers) verifying:
  - Merged top-K matches brute-force merge of independently-run single-shard searches with
    generous `rerankK` (ground truth).
  - Resume logic actually triggers when a shard is undersized relative to its data (e.g. one tiny
    shard, one huge shard, query near the tiny shard's data) and improves recall vs. Phase 1's fixed
    overquery.
  - Exhausted shards (fewer live/accepted nodes than requested) don't spuriously trigger resume.
  - `close()` releases all per-shard resources (mirror existing `GraphSearcher` close tests).
- A recall/latency benchmark added to `jvector-examples` (`Grid` or a new example) comparing:
  single merged index vs. N-shard `MultiGraphSearcher` over the same data, at a few shard counts.

## Open questions

- Should `resume()`'s `rerankFloor`-accepting overload eventually become public on `GraphSearcher`
  itself, for consumers who want to build their own multi-index logic without being confined to
  jvector's package? Not needed for this proposal, but worth revisiting if there's demand.
- Is a fixed `growthFactor` (doubling) the right default, or should it also scale with how far over
  the cutoff a shard's worst score was? Needs empirical tuning once Phase 2 lands.
- Should `ShardedSearchResult` expose per-shard sub-results (not just the flattened merged list),
  for callers who want to inspect per-shard contribution? Leaning no for v1 — keep the surface
  minimal — but flagging it since Astra-style consumers might want it for diagnostics.

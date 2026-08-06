# Retain-largest compaction: theory, implementation, and validity regime

*Prepared for upstream review. Branch `jvector-experiments` (formerly
`compaction-retain-largest`); the implementation is commits `55eec98b`
(stage 1) and `c67846a3` (stage 2), with the supporting measurements in
`DESIGN-multi-source-compaction-gather.md` and
`TRAVERSAL-PATTERNS-compaction.md` on the same branch.*

## Abstract

jvector's N:1 on-disk graph merge derives every output node's neighborhood
symmetrically: each surviving node in each source runs a beam search
against every *other* source. We measured, on a production-shaped
12-source merge, that this makes the search **machinery** — not similarity
arithmetic — the dominant CPU consumer, and that the search count scales
as `T × (k−1)` (T = surviving nodes, k = sources): ~220M searches to merge
20M nodes. We claim this symmetric derivation is redundant work whenever
one source dominates the survivor set, because that source's graph is
already a valid ANN graph over most of the output points, built by the
same construction the merge would repeat. The experiment retains the
largest source's adjacency verbatim (ordinal-remapped, zero searches),
inserts only the delta with **one** search each against the base, and
repairs reachability with a buffered backward-edge pass — reducing search
count to `|delta| × 1`. Measured recall: **parity with the symmetric merge
(0.9960 vs 0.9960) when the base holds 84% of survivors — the LSM tier
shape — and a collapse (0.7920 vs 0.9560) at 25%**, so the implementation
measures the base share per merge and falls back to the symmetric path
below a threshold. The strategy is a regime, not a universal win, and the
implementation encodes the regime.

## 1. Setting

The consumer is an LSM store: immutable sorted segments, each carrying its
own jvector graph, periodically compacted N:1. Size-tiered compaction
produces a characteristic merge shape — **one large accumulated source
plus a tail of small flush segments** (measured tier: one ~17M-node base
plus eleven ~1M-node segments). Sources are disjoint (no shared points),
each source's graph was built by the standard incremental construction,
and deletions are applied via per-source live-node bitsets at merge time.

## 2. The measured problem

Two independent measurements motivated the work.

**Superlinear merge cost in source count.** Wall-clock per million output
nodes, across production merges of similar total size:

| merge | sources | min/M-nodes |
|---|---|---|
| 1–10M ladder | 2–3 | 0.47–0.64 (flat) |
| 31M | 2 | 0.82 |
| 42M | 3 | 1.32 |
| 20M | **12** | 1.91 |
| 30M | **12** | 2.07 |
| 60M | 5 | 2.74 |

The per-node cost tracks source count, not total size — the `(k−1)`
fan-out made visible.

**Where the CPU actually goes.** A live 12-source, ~19.86M-row merge
(fused ADC path, ~40 of 64 cores busy, disk at 2%), 112 worker frames
across 3 stack samples:

| share | frame | character |
|---|---|---|
| 35% | `AbstractLongHeap.upHeap` | candidate-heap maintenance |
| 22% | `dotProduct` (Panama SIMD) | the actual similarity work |
| 15% | `FusedPQDecoder.<init>` | per-search score-provider setup |
| 12% | `Unsafe.copySwapMemory0` | vector-read copies |
| 12% | `processNeighbors` / `searchOneLayer` | traversal |
| 5% | `readFully` / `seek` | I/O |

Under a quarter of the machine computes similarities. The majority is the
*structure* of running `k−1` beam searches per node — heap sift-ups and
per-search provider construction. Two consequences follow: (a) tuning the
distance path (quantization, rerank policy, IO) cannot recover the 50%
spent on search machinery — only removing searches can; (b) the beams are
already quantization-driven (`FusedPQ.approximateScoreFunctionFor`, full
precision only on rerank), so this is not a misconfiguration artifact.

## 3. The premise

**Claim.** When one source holds the majority of surviving nodes, its
graph is already the answer for those nodes. It was built by the same
navigable-graph construction the merge re-runs; its neighborhoods are
exact with respect to its own point set, and that point set *is* most of
the output. Re-deriving them via `k−1` searches per node re-purchases, at
the merge's most expensive rate, structure the system already owns.

**Support from the literature.** No published disk-ANN system merges
symmetrically:

- **FreshDiskANN** (StreamingMerge) retains the long-term index and
  inserts only the delta: one GreedySearch per inserted point, backward
  edges buffered in a Δ map sized by the *change set* (`O(|N|·R)`), then a
  sequential patch-and-prune sweep. Reported: a 10% change to a
  billion-point index merges in ~10% of rebuild time, recall maintained.
- **Lucene HNSW** (9.6+) initializes the merged graph from the largest
  deletion-free segment (ordinal-remapped) and re-inserts only the other
  segments' vectors — ~43% faster merges, recall unchanged. Lucene 10.2
  extends this by seeding insertions through the small segments' own
  edges.
- **DiskANN's** sharded build merges by pure edge union with *zero*
  distance computations — valid only because its shards overlap
  (bridging points supply connectivity). LSM segments are disjoint, so
  this exact trick is unavailable to us, but it calibrates how cheap a
  merge can be when existing structure supplies connectivity.
- **FAISS IVF** merges by verified-compatible list concatenation — again
  zero distance work when structure allows it.

The pattern across all four: **reuse the structure you have; pay search
cost only for what is genuinely new.** Retain-largest is that pattern
applied to jvector's disjoint-segment merge, closest in shape to
FreshDiskANN's insert phase and Lucene's largest-segment initialization.

**The cost model.** Symmetric: `T × (k−1)` searches. Asymmetric:
`|delta| × 1` — delta nodes search the base once and do *not* search each
other; they reach each other through the base. For the measured tier
(~2/3 of survivors in the base, k=12): ~220M searches → ~10M, with the
35%+15% machinery overhead removed from the retained majority entirely.

**What breaks, and the repair.** Retaining the base's neighbor lists
verbatim leaves them pointing only at base nodes: nothing points at the
delta, and every delta vector becomes unreachable from the entry point —
a recall cliff that would *present as a large speedup* on any timing-only
benchmark. The repair is FreshDiskANN's: when a delta node's search
selects a retained-base neighbor, the reciprocal edge is buffered; base
nodes fold their incoming edges in when written. Crucially, backward
edges enter as ordinary scored candidates through the **same diversity
provider** that prunes every neighborhood — a retained neighbor and an
incoming delta neighbor compete on equal terms rather than the delta
being appended past the degree bound.

**The validity regime — the part that makes this a theory rather than a
trick.** Delta nodes route to each other *through the base*. If the base
is a small fraction of the output, it cannot connect a delta that
outnumbers it: neighborhoods that should span the delta collapse onto too
few routing points. The regime condition is therefore the retained
source's share of *surviving* nodes (survivors, not disk size — deletions
are exactly what makes a formerly-large segment not worth retaining).
This predicts recall parity at high share and degradation at low share,
which is what we measured (§5).

## 4. Implementation

Two stages on `OnDiskGraphIndexCompactor`, opt-in via
`setRetainLargest(boolean)`:

**Stage 1 — asymmetric gather** (`55eec98b`). `gatherCandidates` splits:
a node in the retained base contributes its own neighbor list only
(ordinal-remapped — the source-major remapping keeps the base block
contiguous, making edge reuse a pure ordinal rewrite); a delta node
contributes its own neighbors plus one search of the base. The retained
source is `argmax` over live-bitset cardinality (k popcounts, no I/O).
Stage 1 alone is deliberately unusable: enabling the flag without stage 2
throws, naming the reachability defect — a guard mutation-tested so it
cannot pass vacuously.

**Stage 2 — backward-edge pass** (`c67846a3`). Two batch passes, ordered,
*not* concatenated: delta first (buffering reciprocal edges), base second
(folding them in). Record writes are positional, so pass ordering — not
list concatenation — is the barrier; a windowed batch runner would
overlap a concatenated tail and silently drop edges (an earlier revision
made exactly this mistake; the commit corrects it in place). The Δ buffer
is **bounded and refuses**: exceeding its ceiling (worst case
`|delta| × degree`) throws rather than dropping edges, since dropped
edges cost recall on exactly the vectors the pass exists to make
reachable.

**The regime gate.** Per merge, the compactor computes the base share of
survivors and falls back to the symmetric merge below
`RETAIN_LARGEST_MIN_BASE_FRACTION` (0.5, provisional), logging the
decision either way. Enabling the feature *asks* for the strategy; the
merge-shape measurement decides. The threshold is bracketed, not
calibrated: the two measured points (0.25 collapse, 0.84 parity) straddle
it and the knee has not been located.

## 5. Measured results

Recall at equal degree, same query set, retain-largest vs symmetric:

| base share of survivors | symmetric | retain-largest |
|---|---|---|
| 84% (LSM tier shape) | 0.9960 | **0.9960** |
| 25% (four equal sources) | 0.9560 | 0.7920 |

Parity in the regime the strategy targets; a 0.164 collapse outside it —
caught by an equal-sources test that a timing-only benchmark would have
reported as a large speedup. That test is retained, now asserting the
fallback fires. Search-count reduction in the target regime: ~22× for the
measured tier (220M → 10M). Test state at the experiment tip: 50 green in
the compactor/cache/limiter suites, 39 across the broader graph suites.

## 6. Relation to `compaction-preencode-chunked-and-xlink`

The current upstream branch independently landed a **retained-only fast
path for offer-free nodes of the largest source** (`966a46df`) inside its
pair-asymmetric cross-linking scheme — evidence of convergent reasoning:
the largest source's structure is worth keeping. The difference is scope
and explicitness. The xlink fast path skips work for base nodes that
received no cross-source offers, within a scheme that still searches
across sources; retain-largest makes the stronger claim — skip *all*
neighborhood re-derivation for the base and *all* cross-delta search —
and pairs it with an explicit, measured validity regime and fallback.
The two are not mutually exclusive: retain-largest can be read as the
limiting case of the xlink asymmetry as base share → 1, and the regime
gate is the criterion for when the limiting case is safe.

That reconciliation is no longer hypothetical: this branch IS the
experiment ported onto the xlink base, and the composition rule that
survived both test families is **per-merge scheme selection**. When the
regime gate engages, retain-largest replaces the cross-link overlay for
that merge — the reverse-candidate buffer is never allocated (every
overlay site null-guards on it, including the retained-only fast path,
which would otherwise skip exactly the gather that folds backward edges
into base nodes), and reciprocal structure comes solely from the
delta-bounded backward-edge buffer under the two-pass barrier. When the
gate declines, the xlink scheme runs unmodified. The two reciprocal-edge
buffers thus turn out to be alternatives at different scopes — reverse
propagation sized by all nodes for the symmetric-ish regime,
backward-edge buffering sized by the delta for the dominated regime —
selected by the measured merge shape. Both suites pass on the composed
code: the xlink cross-link/fast-path/seeding tests and the experiment's
guard/fallback/recall-parity tests, 27 together.

## 7. Open questions for review

1. **The knee.** Recall vs base-share needs measuring between 0.25 and
   0.84 to place the fallback threshold on evidence instead of a
   bracket. (One suspects the knee interacts with delta *count* — many
   small deltas vs one large — not just aggregate share.)
2. **Hierarchy.** Upper layers are currently rebuilt conventionally;
   whether the retained base's upper layers can also be kept (with entry
   point revalidation) is unexplored.
3. **Interaction with xlink's reverse-candidate propagation** — whether
   the Δ backward-edge buffer and the reverse-candidate buffer are the
   same mechanism at different scopes (we believe they are), and if so,
   which generalizes.
4. **Δ ceiling policy.** Refuse-on-overflow is correct for an
   experiment; a production policy needs either spill or a pre-merge
   size check that routes to symmetric.
5. **Recall methodology.** Our parity numbers are ground-truth recall@K
   on the consumer's datasets at two share points; an upstream-standard
   benchmark run (ann-benchmarks shapes, multiple K) would make the
   claim portable.

## Reproduction

`git checkout jvector-experiments` — the flag-guard, source-selection,
equal-sources-fallback, and recall-parity tests are in
`TestOnDiskGraphIndexCompactor`; the profiling method (stack-sample
shares over live merges) and full cost model are in
`DESIGN-multi-source-compaction-gather.md` §"Measured confirmation of
gap #1"; access-pattern inventory in `TRAVERSAL-PATTERNS-compaction.md`.

# Experiment: retain-largest merge (branch `compaction-retain-largest`)

Branched from `compaction-livenodes-enum-integration2` @ `062587e2`, which
carries the upstream `fix/preencode-chunked-cache` merge plus our local
adaptations. **Goal: stop re-deriving every node's neighborhood.**

## The claim being tested

Gap #1 of `DESIGN-multi-source-compaction-gather.md`, now measured (see
"Measured confirmation of gap #1"): the merge runs `k−1` beam searches for
*all* `T` output nodes. At 12 sources that is ~220M searches for 20M nodes,
and the profile shows the search *machinery* — not similarity math —
consuming the majority of the CPU:

| 35% | `AbstractLongHeap.upHeap` | 22% | `dotProduct` | 15% | `FusedPQDecoder.<init>` |

No system in the literature merges this way. DiskANN unions overlapping
shards with **no distance computations at all** (`merge_shards`: union,
dedup, shuffle, truncate). That specific trick needs shard overlap, which
disjoint LSM segments do not have. FreshDiskANN and Lucene take the other
route available to us: **retain the largest structure, insert only the
delta.**

## The change, concretely

Today, symmetric (`compactLevels` → `gatherCandidates`, line ~1496/1547):

```
for every surviving node in every source:        # T nodes
    own source      -> reuse existing neighbors  # gatherFromSameSource
    each other src  -> beam search               # gatherFromOtherSource  x (k-1)
```

Proposed, asymmetric:

```
base = argmax(surviving nodes)                   # the large source
for every surviving node in base:                # ~2/3 of T here
    reuse its existing neighbor list, remapped   # NO search
for every surviving node in the other k-1 sources:   # the delta
    ONE search against the merged structure      # not k-1 searches
    buffer backward edges for base nodes         # FreshDiskANN's Δ
finally: patch buffered backward edges, prune overfull neighborhoods
```

Search count goes from `T × (k−1)` to `|delta| × 1`. For the measured tier
(one large source + eleven ~1M flush segments, 20M surviving): ~220M
searches → ~10M. The retained base keeps its edges for free.

## Where the code changes

- `OnDiskGraphIndexCompactor.compactLevels` (~1243) — split the node loop
  into retained-base and delta passes instead of one symmetric sweep.
- `gatherCandidates` (~1496/1547) — becomes delta-only; `gatherFromSameSource`
  generalizes to "reuse the retained neighbor list".
- Ordinal remapping already exists and is source-major
  (`CompactionGraphMerger` step 1) — the base source's block stays
  contiguous, which is what makes edge reuse a pure ordinal rewrite.
- Backward-edge buffer (Δ) is new: sized by the change set, not the index
  (FreshDiskANN sizes it O(|N|·R)).

## What must be proven before this is more than a speedup

1. **Recall parity** against the symmetric merge at equal degree — the
   retained base's edges were built against a smaller graph, so its
   neighborhoods are "stale" relative to the merged set. FreshDiskANN's
   answer is the final prune pass; ours must be measured, not assumed.
2. **Connectivity** — every delta node reachable from the entry point, and
   no base region orphaned by the ordinal rewrite.
3. **The Δ buffer's memory ceiling** must be bounded by the change set and
   charged through the existing `ProgressLimiter` budget, or a large delta
   reintroduces the heap pressure the streaming design removed.

## Non-goals

- Vector-value de-duplication. The merge path keys postings by ordinal
  handle (`ChronicleMap<Long, …>`); full value-keyed dedup exists only on
  Cassandra's REBUILD path and must not migrate here.
- Changing the beam's scoring. It is already PQ-driven with FP only on
  rerank; the win here is removing searches, not re-tuning them.

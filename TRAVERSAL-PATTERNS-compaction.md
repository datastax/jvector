# Traversal patterns in vector-index compaction — jvector vs FAISS vs DiskANN

What actually gets *walked, read, and computed* when a vector index is
compacted. Part 1 traces jvector's merge end to end; Parts 2–5 do the same for
FAISS, DiskANN, FreshDiskANN and Lucene; Part 6 puts them side by side and
draws out what the comparison says about our implementation.

Everything about jvector here is read off the current code
(`OnDiskGraphIndexCompactor`, branch `livenodes-integration`) and confirmed
against thread profiles of live 12-source merges. Everything about the other
systems is from primary sources (papers + implementation), cited in
`DESIGN-multi-source-compaction-gather.md`; raw copies in
`~/notes/ann-merge-literature/`.

---

## Legend

Access patterns are named consistently throughout:

| symbol | meaning |
|---|---|
| **SEQ** | sequential streaming — reads/writes advance monotonically through a file |
| **RAND** | random access — address determined by data, not by position |
| **RAM** | served entirely from memory; no disk access |
| **FP** | full-precision vector (e.g. 384 × float32 = 1536 B) |
| **ADC** | asymmetric distance computation — query scored against *compressed* codes via a lookup table, no FP read |

The unit that matters for cost is not "a read" but **a read that misses page
cache**. A RAND pattern over a working set that fits RAM costs CPU only; the
same pattern over a working set that doesn't costs a 4 KiB disk fault. This
distinction is the whole story at scale, so it is called out per phase.

---

# Part 1 — jvector: what a merge traverses

## The shape of the problem

An LSM compaction hands jvector **k source segments**, each a complete
on-disk Vamana/HNSW-style graph with its own ordinal space, plus a
per-source bitset of which ordinals survive. The output is **one** graph
containing all surviving nodes under a fresh, packed ordinal space.

```
   source A (10M nodes, own graph)  ─┐
   source B (1M nodes,  own graph)  ─┤
   source C (1M nodes,  own graph)  ─┼──►  merged graph (T nodes, one graph)
   ...                              ─┤
   source L (1M nodes,  own graph)  ─┘
        k sources                          T = Σ surviving nodes
```

The crucial structural fact: **the sources share no edges.** Each is an
independent graph built over disjoint data. Nothing in source A points into
source B. So the merged graph's neighbor lists cannot be assembled by copying
— for a node in A, its true nearest neighbors may live in B, and no edge in
any input reveals that. jvector therefore *rediscovers* neighborhoods, and
that choice determines every traversal pattern below.

## Phase 0 — row-pass ingest (Cassandra side)

Before jvector is invoked, the compaction's row iterator has already walked
every output row to build the postings/dedup map keyed by global ordinal.

```
  rows (sequential from sstables) ──► ChronicleMap[globalOrdinal → postings]
```

**Pattern: SEQ read + RAM hash writes.** No vector math. Measured at ~27% of a
low-source-count merge's wall time (8.3 s per million rows); parallelized
across the build pool when `ingest_parallel=true`.

## Phase 1 — codebook resolution

The merged graph needs a PQ codebook for its fused inline codes. Two paths:

**(a) `REFINE`** — sample vectors across sources and run warm-start Lloyd's:

```
  for each source: pick ~N/k sample ordinals (chunk-shuffled, 32-node runs)
       └─ prefetch each run, then read FP vectors        [SEQ-ish, MADV hints]
  train: k-means refinement over the sample set          [RAM, CPU-heavy]
```

**Pattern: near-SEQ read of a bounded sample (≤ MAX_PQ_TRAINING_SET_SIZE), then
pure RAM compute.** Deliberately chunked so the OS read-ahead covers it rather
than faulting per node.

**(b) `ADOPT_UNIFORM` / `ADOPT_CHECKED`** (current production setting) — if all
sources carry a content-identical codebook, adopt it verbatim:

```
  compare k ProductQuantization objects in RAM  ──►  adopt
```

**Pattern: RAM only. Zero traversal.** `ADOPT_CHECKED` adds one bounded sample
sweep (~16 K vectors) for the drift check. This is the O4 work: on a uniform
lineage the entire training traversal disappears.

## Phase 2 — pre-encode pass

Every surviving node's PQ code is materialized into a scratch cache (mapped
past the projected output end), because the fused format writes each node's
code inline in *its neighbors'* records — so codes get read many times during
the write.

```
  (b) with canonical codes (O4, pq_code_persistence=true):
       source PQ.db codes tail ──copy──► code cache      [SEQ read, SEQ write]

  (a) without:
       for each source, in ordinal order:
            prefetch block ──► read FP vector ──► encodeTo ──► cache
                                                             [SEQ read, CPU]
```

**Pattern: SEQ in both variants** — this pass was always block-ordered. The
difference is arithmetic: (a) re-derives every code from full precision
(measured ~95% of merge CPU when the cache was disabled entirely), (b) is a
memcpy (20 M codes in 0.5 s, measured).

**Residual encode sites, and why they are correct rather than gaps.**
`maybeEncodePQ` still calls `pq.encodeTo` for every **level-1** node, and
`onAfterLevels` encodes the single entry node, instead of copying those
ordinals' codes out of the cache. This looks like an oversight against "encode
once per vector lifetime" but is the better choice, for a reason specific to
where the cache lives: the pre-encode cache is a **`MappedByteBuffer` over the
output file**, so reading from it can page-fault. At a level-1 node the
compactor already holds the node's vector in `scratch.baseVec` (read during
the gather), so `encodeTo` is pure CPU with *zero* IO, whereas a cache copy
risks a fault. The cache is the right source only where the alternative is
reading a **full-precision vector** from disk — 100 B copy versus 1536 B read
— which is exactly the L0 neighbor-code write path that uses it.

The general rule this yields: **copy a code when the alternative is reading a
vector; compute a code when the vector is already in hand.** Remaining
`encodeTo` calls are either that rule applied correctly, or genuine fallbacks
(cache inactive, non-fused sidecar strategy).

## Phase 3 — the gather (the heart of the merge)

Now the expensive part. **For every one of the T output nodes**, independently:

```
 output node n  (lives in source A)
    │
    │ 1. read n's own FP vector                    source A, ordinal n
    │                                              [FP read, ~SEQ: nodes are
    │                                               visited in ordinal order]
    │
    │ 2. SAME-SOURCE candidates — walk n's existing neighbor list in A
    │      for each of ≤32 neighbors:
    │          read neighbor's FP vector           [FP read, RAND within A]
    │          exact compare vs n
    │
    │ 3. CROSS-SOURCE candidates — repeat for EACH of the other k−1 sources:
    │
    │      ┌── source B ──────────────────────────────────────────────┐
    │      │  build ADC table for query n against B's codebook        │
    │      │     (O1: shared across sources of the same codebook)     │
    │      │                                                          │
    │      │  hierarchy descent:  L2 ─► L1   (greedy, 1 candidate)    │
    │      │       each hop: read node record + neighbor codes  [RAND]│
    │      │                                                          │
    │      │  L0 beam search (beamWidth, topK=12):                    │
    │      │       pop best from frontier                             │
    │      │       read its packed neighbor codes           [RAND]    │
    │      │       score all neighbors by ADC              [no FP]    │
    │      │       push improvements onto frontier                    │
    │      │       ... repeat until beam converges ...                │
    │      │                                                          │
    │      │  for each of the 12 results:                             │
    │      │       read FP vector, exact rescore           [FP, RAND] │
    │      └──────────────────────────────────────────────────────────┘
    │      ... and again for source C, D, ... L  (k−1 independent beams,
    │          each with its own frontier, its own visited set)
    │
    │ 4. sort all candidates by score                    [RAM]
    │      candidates ≈ searchTopK·(k−1) + maxDegree = 164   (k=12, deg=32)
    │
    │ 5. diversity selection (Vamana α-escalation):
    │      for α in {1.0, 1.2}:
    │        for each candidate in score order, until 32 selected:
    │            get candidate's FP vector    ◄── was a RE-READ from disk;
    │                                             now reused from the gather
    │                                             (commit 9eeaae61)
    │            compare against already-selected  [RAM, cached vectors]
    │
    └─ 6. write n's record: neighbors + their inline codes  [SEQ append]
```

### What that costs, per output node

| step | reads | pattern | notes |
|---|---|---|---|
| 1. own vector | 1 FP | ~SEQ | nodes visited in ordinal order |
| 2. same-source | ≤32 FP | RAND in home source | exact-only, no ADC path |
| 3. beams | (k−1) × beam-visits × record | **RAND across all k−1 sources** | ADC scored, no FP |
| 3. rescore | (k−1) × 12 FP | **RAND** | = 132 FP reads at k=12 |
| 5. diversity | **0** (was ~164 × α-rounds) | — | fixed 2026-07-27 |

Two properties dominate everything:

**(i) The multiplier is k.** Steps 3 repeats per source. A 12-source merge does
11× the beam work of a 2-source merge for the *same* output size. Measured on
the same binary: ~0.36 min per merged million at k=3 versus ~0.99 at k=12.

**(ii) The pattern is RAND over all sources at once.** Each beam hops
data-dependently through a source's records. Every source is being hopped
through concurrently by different worker threads on different output nodes. So
the merge's working set is *all k sources simultaneously*, and there is no
locality to exploit: a record read for node n is unlikely to be wanted again
for node n+1.

This is why the merge has two entirely different performance regimes:

```
   working set fits page cache          working set exceeds page cache
   ────────────────────────────         ──────────────────────────────
   RAND reads = memory reads            RAND reads = 4 KiB disk faults
   CPU-bound: beam heap + ADC math      IOPS-bound: workers block on IO
   ~32-38 of 64 cores busy              ~9 of 40 workers runnable,
   disks ~1% utilized                    NVMe ~93% utilized, ~57K reads/s
```

Measured crossing of that cliff on this rig, at constant k=12:
0.99 min/M at 20 M nodes → 1.11 at 30 M → **1.42 at 40 M**. Cost per million
climbing at fixed source count is the IOPS signature; nothing about the
algorithm changed, only whether the sources still fit RAM.

### Aside: "didn't we already eliminate the compaction scans?"

Yes — but a *different class* of scan, and the distinction matters because the
two are easy to conflate.

The `compaction-livenodes-enum` line (commits `3c814fac`, `16f5f3cb`, both on
`livenodes-integration` already) eliminated **enumeration/metadata scans**:
counting live nodes from the in-memory bitset instead of walking them, and
enumerating L0 nodes from `liveNodes` rather than `source.getNodes(0)` — which
was the first cold access of a merge and dominated its startup. Those were
one-per-merge whole-file walks of node *headers*.

They did not touch, and were never about, the per-candidate **full-precision
vector reads** in Phase 3, which are data-dependent, occur T × k times, and are
the subject of this section. Neighbouring branches don't address them either:
`compaction-rerank-prefetch` duplicates commits we already carry,
`compaction-bench-fusedpq-rerank` is search-side benchmark configuration, and
`compaction-pq-improved` tunes refinement parameters.

So: startup enumeration — fixed, upstream, already ours. Per-candidate FP
reads during gather — still present, and unaddressed anywhere upstream.

### Where full precision is actually read

Worth stating explicitly, because it is easy to assume the fused/ADC path
removed FP reads entirely. It did not — it removed them from *beam steering*
only. FP reads remain at: the node's own vector (1), same-source neighbor
scoring (2), and cross-source rescore (3). Before commit `9eeaae61`, diversity
(5) re-read them all again — measured at 41% of all merge FP reads.

**Measured after the diversity fix (2026-07-27, live k=12 merge, 14
merge-active thread dumps, 183 `readFloatVector` samples):**

| site | samples | share |
|---|---:|---:|
| `gatherFromSameSource` — exact-scoring the home source's existing neighbors | 157 | **86%** |
| `rescore` after each cross-source beam | 26 | 14% |
| `retainDiverse` | **0** | — (was 41% before commit `9eeaae61`) |

The diversity re-reads are gone as intended. But the remaining split inverts
the assumption behind O3, and the reason is instructive. By *count*, the
cross-source rescore should dominate: at k=12 it reads
`(k−1) × searchTopK = 132` vectors per node versus same-source's ≤32. By
*cost* it is six times cheaper, because the two sites have opposite cache
behaviour:

- **Same-source neighbours are scattered.** A node's neighbour list points at
  arbitrary ordinals across the home source, none of them touched recently —
  every read is a cold random access.
- **Cross-source rescore reads what the beam just walked.** The ADC beam has
  already faulted those records in while traversing to them, so the rescore is
  a warm re-read of the region the search converged on.

**The implication is that the dominant full-precision cost in the merge is
scoring neighbours we did not need to read at all.** The fused layout stores
each node's *neighbours' PQ codes inline in its own record* — that is what
makes cross-source ADC possible. The home source's neighbours are therefore
already scoreable by ADC straight out of the base node's record, with **zero**
additional reads, exactly as cross-source candidates are. `gatherFromSameSource`
predates the fused path and still exact-scores them from disk.

Making same-source scoring ADC-based (with the same optional rescore policy as
everything else) would remove ~86% of the merge's remaining full-precision
reads and is independent of O3 and O5. Call it **O7**; on this evidence it
outranks O3.

Note also what does **not** happen: the cross-source search is constructed
with the single-argument `DefaultSearchScoreProvider`, so its internal
reranker is `null`. The search never reranks; the explicit rescore afterward is
the only exact pass. FP reads are not doubled.

## Phase 4 — write

```
  output nodes, in ordinal order ──► one growing file  [SEQ write, single pass]
```

Each record carries the node's neighbors *and a copy of each neighbor's PQ
code* (that's the "fused" layout — it makes search read one block instead of
two). Code copies come from the Phase-2 cache, so no re-encode.

## Phase 5 — epilogue

```
  footer CRC:   re-read the ENTIRE output file       [SEQ read, whole file]
  codes tail:   append canonical codes               [SEQ write]
  postings/PQ:  write SAI components                 [SEQ write]
```

**Pattern: SEQ, but large.** At the 40 M tier the CRC re-reads ~150 GB. This
phase does no vector math at all; it was invisible to progress reporting until
byte-level progress was added.

## jvector merge — whole-run summary

```
 Phase        Pattern                  Vector math          Scales with
 ─────────────────────────────────────────────────────────────────────────
 0 row-pass   SEQ + RAM hash           none                 rows
 1 codebook   RAM (adopt) / SEQ sample k-means (refine)     sample size
 2 pre-encode SEQ copy (or SEQ+encode) none (or encodeTo)   T
 3 GATHER     ***RAND, all sources***  ADC + exact rescore  T × k   ◄── cost
 4 write      SEQ                      none                 T
 5 epilogue   SEQ (whole file)         none                 output bytes
```

---

# Part 2 — FAISS: merge without traversal

FAISS's IVF indexes are inverted lists: a coarse quantizer assigns each vector
to one of `nlist` centroids, and the vector is stored as a PQ/SQ code inside
that centroid's list.

```
  shard1.ivfdata:  list0 [codes|ids]  list1 [...]  ...  listN [...]
  shard2.ivfdata:  list0 [codes|ids]  list1 [...]  ...  listN [...]
  shard3.ivfdata:  list0 [codes|ids]  list1 [...]  ...  listN [...]
                          │
                          ▼   per list: concatenate bytes, shift ids
  out.ivfdata:     list0 [s1|s2|s3]   list1 [s1|s2|s3]  ...
```

**Traversal: SEQ concatenation. Zero vector reads. Zero distance
computations. Zero graph walks.**

This is legal only because of a precondition FAISS checks aggressively:
`check_compatible_for_merge` requires identical `d`, `nlist`, `code_size` and
concrete type, and in expensive-check mode it *reconstructs every centroid
from both coarse quantizers and asserts they are bitwise equal*. Given a
shared quantizer and shared codebooks, a code written by shard 2 means exactly
what it would have meant in shard 1 — so the merged index is just the
concatenation. The merge body is two lines.

The on-disk variant (`OnDiskInvertedLists::merge_from_multiple`) opens shards
with `IO_FLAG_MMAP` so shard payloads are never materialized in RAM, and
streams lists into the output file.

**FAISS HNSW has no merge at all.** Graph indexes there are rebuild-only.

**The lesson jvector already applies:** this is precisely the O4 doctrine —
shared codebook + conservative content-equality verification makes the *code*
component copyable rather than recomputable. jvector's canonical-codes tail
does for its code component exactly what FAISS does for its lists. What
jvector cannot borrow is the rest: FAISS has no topology to reconcile.

---

# Part 3 — DiskANN: merge by edge union

DiskANN's billion-scale build partitions data into **overlapping** shards
(k-means, each point assigned to its ℓ=2 nearest centroids), builds a Vamana
graph per shard in RAM, then merges.

```
  shard1.graph ──stream──┐
  shard2.graph ──stream──┤   for each global node id, in order:
  shard3.graph ──stream──┼──►   collect its neighbor lists from every shard
       ...               │      that contains it  (ids renamed via idmap)
  shard40.graph ─stream──┘      union + dedup
                                shuffle, truncate to max_degree
                                └──► merged.graph   [SEQ write]
```

**Traversal: k interleaved SEQ streams in, one SEQ stream out. Zero vector
reads. Zero distance computations.** Verified in `merge_shards`
(`src/disk_utils.cpp`): the truncation is `std::shuffle` + `min(size,
max_degree)` — there is *no* RobustPrune and no distance math in the merge.

Why union suffices, and why jvector can't copy it: the shards **overlap by
construction**. A point assigned to two clusters appears in two shard graphs
and carries edges from both, so it bridges them; greedy search can walk from
one shard's region into another's. The paper's own words: the overlap
"provides sufficient connectivity for GreedySearch to succeed even if the
query's nearest neighbors are actually split between multiple shards."

**jvector's segments are disjoint** — they are LSM flush artifacts partitioned
by *arrival time*, not by geometry, with no overlap points. Union of disjoint
graphs yields a disconnected graph. This is the structural reason jvector
searches instead of unioning, and it is not a fixable oversight — it follows
from where the segments come from.

Cost of DiskANN's approach, for calibration: the sharded path took ~5 days for
1 B points but held peak RAM **under 64 GB**, versus ~2 days and ~1100 GB for
the one-shot build. Merge itself is cheap; the per-shard *builds* are the cost.

---

# Part 4 — FreshDiskANN: two sequential passes plus bounded random

The closest published analogue to what jvector does, and the most instructive
contrast. It merges an in-memory delta (N inserts, D deletes) into an
SSD-resident long-term index.

```
 PHASE 1 — DELETE (repair neighborhoods pointing at deleted nodes)
   ├────────────── SEQ sweep of the whole LTI, block by block ──────────────┤
   for each block:  load ─► for each node with deleted out-neighbors:
                              splice in the deleted node's neighbors,
                              RobustPrune               [PQ distances, RAM]
                    write block back                    [SEQ write]

 PHASE 2 — INSERT (place the new points)
   for each new point p:
        GreedySearch against the SSD-resident index     [~100 RAND 4 KiB reads]
        RobustPrune to choose p's out-edges             [PQ distances, RAM]
        backward edges NOT written to disk:
              Δ[p'] += p                                [RAM map, O(|N|·R)]

 PHASE 3 — PATCH (apply the buffered backward edges)
   ├────────────── SEQ sweep of the whole LTI, block by block ──────────────┤
   for each block:  load ─► add Δ entries, prune if degree > R
                    write block back                    [SEQ write]
```

Three design decisions worth naming, because each maps to something jvector
either has, lacks, or does differently:

1. **All distance computations use RAM-resident PQ vectors — the merge reads
   no full precision at all.** jvector reads FP at three sites (Part 1).
   jvector's O4 canonical-codes tail is the substrate that *could* supply the
   same RAM-resident code table.

2. **Whole-index work is exactly two SEQ passes.** The only RAND access is the
   per-inserted-point search, bounded at ~100 4 KiB reads by the graph's α-RNG
   property, and proportional to the *delta*, not the index. jvector's RAND
   access is proportional to T × k — the whole output, times the source count.

3. **Backward edges are buffered in RAM sized by the change set** (2·R·4 bytes
   per inserted point; ~7 GiB at 30 M inserts) and applied in the second SEQ
   sweep — converting what would be scattered random writes into sequential
   ones. jvector needs no such buffer only because its symmetric gather has
   every node search for itself.

Measured: merging a 10% change into a billion-point index costs ~10% of a
rebuild. Also measured, and directly relevant to any move jvector makes toward
this shape: query latency during the merge rose from ~5 ms to ~15 ms, with
~40 ms spikes **during the sequential phases**, attributed to head-of-line
blocking behind large sequential IO.

---

# Part 5 — Lucene: retain the largest graph, re-insert the rest

Lucene merges HNSW segments in memory at segment-write time.

```
  segment L (largest, no deletions)  ──► adopt its graph wholesale,
                                          remapping ordinals only
                                          [no searches, no distance math]

  segments S1..Sn (the rest)         ──► for each vector: ordinary HNSW
                                          insert into the adopted graph
                                          [beam search per vector, in RAM]
```

Lucene 10.2 goes further: rather than discarding the small graphs' structure,
it picks a "join set" (~1/5 of their vertices, greedy by degree), inserts those
normally, then uses the small graphs' *own edges* to seed cheap low-`ef`
insertions for the remainder — i.e. it mines the inputs' connectivity instead
of rediscovering it.

**Traversal: zero for the retained graph; one beam search per re-inserted
vector.** Cost is proportional to the *smaller* segments' size, not the total.
Reported: ~43% faster merges at 500 K-doc segments with unchanged recall
(9.6), then a further 1.3–1.7× (10.2).

Limitation worth noting: the seed graph must be deletion-free, and large
segments usually have deletions, so in practice the optimization often does
not fire.

---

# Part 6 — Side by side

| | graph search during merge? | FP vector reads | distance computations | dominant IO pattern | cost scales with |
|---|---|---|---|---|---|
| **jvector** | **yes — (k−1) per output node** | yes (own, same-source, rescore) | ADC beams + exact rescores + diversity | **RAND over all k sources** | **T × k** |
| FAISS IVF | none | none | none | SEQ concat | total bytes |
| DiskANN shard merge | none | none | none | SEQ streams | total bytes |
| FreshDiskANN | only for inserted points | **none** (PQ in RAM) | PQ only | 2 SEQ sweeps + bounded RAND | delta size |
| Lucene HNSW | only for re-inserted vectors | in-memory | beam per re-inserted vector | n/a (in-memory) | smaller segments' size |

## What the comparison actually says

**1. jvector is the only one that re-derives every neighborhood.** Everyone
else either exploits a shared quantizer to make merge a copy (FAISS), exploits
purpose-built overlap to make it a union (DiskANN), or retains the dominant
structure and only does work proportional to what changed (FreshDiskANN,
Lucene). jvector's symmetric gather costs T × k regardless of how similar the
inputs are to the output.

This is not a bug — it follows from merging *disjoint* graphs with no
overlap, which is what LSM flush segments are. But it does mean jvector is
paying the most expensive option available, and the tier shape we actually run
(one large source plus ~11 small flush segments) is precisely the case where
retain-largest would pay off most: ~10 M insertions instead of 30 M full
gathers at the 30 M tier.

**2. Nobody else reads full precision during a merge.** FreshDiskANN runs
entirely on RAM-resident PQ; DiskANN's merge does no distance math whatsoever.
jvector reads FP at three sites. The infrastructure to stop doing so already
exists — the O4 canonical-codes tail is a per-segment array of exactly the PQ
codes a RAM-resident distance table needs.

**3. The RAND/SEQ divide is the scaling story, and only jvector is on the
wrong side of it.** Every other system confines whole-index work to sequential
passes and bounds random access to the change set. jvector's gather is random
access proportional to the entire output times the source count, which is why
it runs beautifully while sources fit page cache and falls off a cliff when
they don't.

**4. But sequential is not free either.** FreshDiskANN's *worst* query-latency
spikes come from its large sequential phases starving concurrent reads. Any
jvector redesign that converts random IO into big sequential sweeps must pace
them and watch query latency, not just merge wall time.

## Where this leaves the roadmap

The optimizations already landed attack the constant factors of the pattern in
Part 1 without changing its shape:

- **O1** (decoder-table sharing) — removed redundant ADC table construction
  across sources sharing a codebook; measured 40% → 2% of gather CPU.
- **O4** (codebook adoption + canonical codes) — deleted Phase 1's training
  traversal and turned Phase 2 into a copy.
- **Diversity vector reuse** (`9eeaae61`) — removed step 5's re-reads, ~41% of
  merge FP reads, with bit-identical output.

The two designed-but-unbuilt steps change the shape itself:

- **O5 (block-wavefront gather)** keeps the symmetric algorithm but inverts the
  loop nesting so sources are swept block-sequentially with many output nodes'
  frontiers in flight — converting the RAND pattern to SEQ, i.e. moving
  jvector to the right side of point 3.
- **O6 (asymmetric retain-largest merge)** adopts the FreshDiskANN/Lucene
  shape — retain the largest source, insert only the smaller sources' nodes,
  buffer backward edges, patch in a sequential sweep — attacking point 1, the
  T × k multiplier itself.

They compose: O5 is *how disk is touched*, O6 is *what work is done*. Details
and phasing in `DESIGN-multi-source-compaction-gather.md`.

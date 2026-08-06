# Compaction: multi-source candidate gathering — cost model and redesign

## Measured problem

Merge wall time is dominated by per-node cross-source candidate gathering, and
its cost axis is **source count**, not merged size. From one day of production
merges (same binary, cache enabled, `min/M` = minutes per million surviving
ordinals):

| merge | sources | wall | min/M |
|------:|--------:|-----:|------:|
| 1–10M ladder | 2–3 | — | 0.47–0.64 (flat) |
| 31M | 2 | 25.4 min | 0.82 |
| 42M | 3 | 55.6 min | 1.32 |
| 20M | 12 | 38.3 min | 1.91 |
| 30M | 12 | 62.0 min | 2.07 |
| 60M | 5 | 164 min | 2.74 |

Same ~30M rows: 2 sources → 25 min, 12 sources → 62 min. A second, milder
axis is size itself (per-search cost grows roughly logarithmically:
0.47 min/M at 1M → 0.82 min/M at 31M, both 2-source).

Thread-sampling a live 12-source-era merge (240 samples across ~40 workers)
attributes the time inside `computeBaseBatch → processBaseNode`:

- ~40% `FusedPQDecoder.newDecoder` under `buildCrossSourceScoreProvider`
  — **per-(node, source) ADC table construction**, not scoring;
- the rest split across `GraphSearcher.search`/`searchOneLayer` (beam
  expansion), `PanamaVectorUtilSupport.dotProduct`, and full-precision
  `rescore` reads.

## Current algorithm (per node, level 0)

`gatherCandidates(node)` iterates every source `s`:

- **home source**: `gatherFromSameSource` — walk the node's existing
  neighbors, exact-score each (full-precision read + compare).
- **each other source** (`k−1` times): `gatherFromOtherSource` —
  1. `buildCrossSourceScoreProvider` → `FusedPQ.approximateScoreFunctionFor`
     → `FusedPQDecoder.newDecoder`: build the query's ADC similarity table
     against *that source's* codebook;
  2. full hierarchical ANN search (`searchTopK` / rerankK = `searchTopK`);
  3. **rescore every returned candidate full-precision** (`rescore`: read
     the vector from the source's mapped file, exact compare).

Then `CompactVamanaDiversityProvider.retainDiverse` selects `maxDegree`
neighbors (more full-precision compares among candidates).

Per-node cost ≈ `(k−1) × [table build + beam + topK exact rescores]`, so a
12-source merge does ~11× the gather work of a 2-source merge — exactly the
measured curve.

## Redesign — composable optimizations

Ordered by (leverage ÷ risk). Each is separately benchmarkable, and the
workload's per-tier `recall_postcompact` measurement is the acceptance gate
for all of them (recall must hold within noise of the current baseline).

Roadmap state: **O1 and O4 are landed and measured**; O2 and O3 are retained
below as the original per-optimization analysis but are superseded in
execution by **O5**, which restructures both for sequential IO; **O6** (in the
literature section) is the further step of not re-deriving every neighborhood
at all. O5 and O6 compose — O5 is how disk is touched, O6 is what work is
done.

### O1. Share the ADC table across sources with identical codebooks
*(pure constant-factor; zero search-semantics change)*

The `newDecoder` table is a function of `(query vector, codebook)`. Under
codebook adoption (the production configuration), most or all sources of a
merge carry the **same** `ProductQuantization` — only retrain events break
the lineage. Today we still build the table `k−1` times per node.

- At compactor construction, partition sources into codebook-equivalence
  classes (`ProductQuantization` equality: same codebooks/subspaces —
  compare centroid arrays, not object identity).
- In `Scratch`, cache one decoder table per equivalence class per node
  (built on first use for that node, reused by the remaining sources of the
  class). The existing `reusableNeighborCodes`/`reusableResults`
  thread-locals already make decoders per-thread; this extends the reuse
  from per-(node, source) to per-(node, class).
- API shape: a `FusedPQ.approximateScoreFunctionFor(...)` overload accepting
  a prebuilt table/decoder, or a `FusedPQDecoder.reuseFor(view, esf)`
  rebinding an existing table to another source's packed-neighbor reader
  (the table is codebook-scoped; the neighbor reader is source-scoped —
  they are separable today, the constructor just conflates them).

Expected effect at 12 sources with one shared codebook: the ~40% table-build
share collapses to ~40%/11 ≈ 4% → **~1.6× merge speedup**, larger as source
count grows. No effect at 2–3 sources (table build amortizes anyway).

### O2. One composite beam over k graphs instead of k−1 independent beams
*(attacks the multiplier itself)*

Replace the per-source searches with a single multi-graph best-first search
per node:

```
seeds    = entry point of every non-home source (k−1 seeds)
frontier = one global priority queue ordered by approximate score
visited  = per-source bitsets (ordinals are per-source)
expand:
  pop globally best candidate (source s, node n)
  push n's neighbors WITHIN s (graphs have no cross edges),
       scored via s's decoder (per-class shared, per O1)
stop     = standard beam termination at a single global budget
           (searchTopK + α·k slack, α small, to keep some per-source
           exploration; α tunable, start α=2)
result   = global top-searchTopK across all sources
```

Rationale: the merged node needs the **global** best-K diverse neighbors;
per-source top-K is an over-collection that diversity selection then throws
away (the same insight that justified rerankK = searchTopK). One global
budget means total expansions ≈ one beam instead of `k−1` full beams; the
frontier naturally starves sources whose best candidates are far.

Implementation map:
- New `MultiSourceGatherer` in `graph/disk/` owning: per-source
  `GraphSearcher` (reuse `scratch.gs`), per-source score function (from O1
  classes), the shared frontier. The hierarchy descent (upper layers) stays
  per-source as today — upper layers are cheap (`processUpperNode`
  unchanged initially).
- `gatherCandidates` L0 path calls it once instead of looping sources.
- The per-source `indexAlive` bits and `liveNodes` filtering apply at
  expansion exactly as `search(..., indexAlive)` does today.

Expected effect: the beam-search share (~40–50% of samples) divides by up to
`k−1`, bounded by the seeds' overhead — call it **2–4× on 12-source
merges**, neutral at 2 sources. Quality note: candidates are drawn by global
competition; per-source coverage is *not* guaranteed, which is the correct
objective for neighbor selection (validated by recall_postcompact).

### O3. Defer full-precision rescoring to survivors
*(cuts the exact-read multiplier and the copySwapMemory tax)*

Today every candidate from every source is rescored full-precision
(`(k−1) × searchTopK` mapped-file reads per node) *before* diversity
selection; selection keeps only `maxDegree`.

Two-phase instead:
1. Pre-select `2 × maxDegree` candidates on approximate (fused) scores;
2. rescore only those full-precision; final `retainDiverse` on exact scores.

Exact-read volume per node drops from `(k−1)·searchTopK` to `2·maxDegree`
(**corrected 2026-07-27**: `searchTopK` is not 100 — it is
`max(MIN_SEARCH_TOP_K, ceil(maxDegree/k) * SEARCH_TOP_K_MULTIPLIER)` = **12**
at 12 sources / degree 32, so the volume is `11 × 12 = 132` exact reads per
node, not 1100. Pre-selecting `2 × maxDegree` = 64 therefore buys only ~2×,
not the ~17× first claimed. O3's standalone value is correspondingly modest;
its real weight is as the Phase-B policy inside O5, where the reads it keeps
are also block-batched). Risk:
approximate-score ordering errors near the selection boundary — the 2×
margin is the standard mitigation; validate with recall_postcompact and, if
needed, raise the margin (cost is linear in it).

### O4. Encode once per vector lifetime: codebook adoption + canonical-code persistence
*(kills the per-merge retrain and the per-merge full-precision re-encode pass)*

Vectors are PQ-encoded at ingest (amortized), but every merge re-encodes every
surviving node from full precision anyway — for two reasons:

1. **The fused layout discards the canonical per-node code.** `FusedPQ` stores
   only *neighbors'* codes, packed/transposed for ADC; the segment's PQ
   component stores only the codebook. So the codes computed at ingest do not
   survive the flush in reusable form.
2. **Every merge refines the codebook** (`PQRetrainer`: balanced sample +
   warm-start Lloyd's). The output codebook differs bit-wise from every
   source's, so even persisted codes would be stale one merge later.

O4 fixes both, gated by explicit configuration:

- **Codebook policy** (`OnDiskGraphIndexCompactor.setCodebookPolicy`):
  - `REFINE` — historical behavior (per-merge retrain).
  - `ADOPT_UNIFORM` — when all sources carry a content-identical
    `ProductQuantization` (one O1 codebook class), adopt it verbatim: no
    sampling pass, no Lloyd's, and all existing codes remain valid. Falls back
    to REFINE on multi-class merges.
  - `ADOPT_CHECKED` — adopt only after a sparse drift check
    (`PQRetrainer.sparseAlignmentCheck`): sample ~16k vectors balanced across
    sources, split train/eval, refine a copy on the train half, and compare
    reconstruction error of retained vs sample-refined codebooks on the
    held-out eval half. Adopt unless refinement improves error beyond the
    configured margin (`setCodebookDriftMargin`, e.g. 0.02). This is the
    retain-and-verify posture: the codebook is kept until the data actually
    drifts, so re-encode cost is paid only at genuine retrain boundaries.
- **Canonical-code persistence**: fused segments append their per-node codes
  as a tail in the Cassandra PQ component (flush: the ingest-computed codes;
  merge: streamed from the pre-encode cache via `setCanonicalCodeSink` before
  truncation). A codebook-adopting merge receives each source's codes via
  `setSourceCanonicalCodes` and the pre-encode pass becomes a **copy**
  (`codesReused` counter) instead of an `encodeTo` — sources without codes
  (cold-start segments, pre-O4 segments) re-encode as before, per source.

Net effect under an unbroken lineage: quantization is computed **exactly once
per vector** — at ingest (or the cold first flush) — and every merge up the
ladder copies. The retrain sampling pass and Lloyd's iterations disappear from
adopted merges entirely. Measured shares: the 1× encode pass is ~0.1–0.15
min/M (backed out of the cache-on/off delta over degree), i.e. ~15–25% of a
low-source-count merge, plus the full retrain phase.

**Follow-on (not yet implemented): ingest-amortized drift statistics.** The
sparse sampling in ADOPT_CHECKED is a stopgap. Since ingest already encodes
every vector, computing each vector's reconstruction error at the same time is
nearly free (one decode + one distance, ~1% of encode cost); persisting
per-segment error stats (count, Σerror) would let the merge decide
adopt-vs-refine from an **exact, already-amortized aggregate over all
vectors** — no merge-time sampling or mini-refine at all. Full incremental
codebook *training* at ingest was considered and rejected: a moving codebook
re-introduces continuous code staleness (the very cost O4 removes), and a
shadow "next codebook" only pays off at actual retrain boundaries.

### O5. Block-wavefront gather — O2+O3 restructured for sequential IO

*Status: design. Motivated by the 2026-07-26 30M-tier measurement: once the
tier working set outgrows the page cache, the gather flips from compute-bound
to IOPS-bound (four NVMes ~93% util at ~57K × 4 KiB random reads, build-pool
workers IO-blocked, ~9 of 40 CPU-busy vs 32–38 when cache-warm).*

#### Access-pattern inventory of the current merge (what is random, what is not)

Per output node, in output-ordinal (source-major ascending) order:

| access | pattern | volume per node (k=12, topK=100, deg=32) |
|---|---|---|
| own record read (`baseVec`) | ~sequential (home source ascending) | 1 × record |
| `gatherFromSameSource` neighbor scoring | random within home source, **full precision** | ≤ 32 × vector |
| cross-source beam traversal (ADC) | **random across every other source** | (k−1) × ~beam visits × record |
| result rescore (`rescore()` after each search) | **random, full precision** | (k−1) × searchTopK = 132 × vector (k=12, searchTopK=12) |
| diversity selection (`retainDiverse:2433`) | **random, full precision — re-read** | candSize × vector, per alpha round |
| output write | sequential, single pass | 1 × record |

Two clarifications this table settles:

- **The write side is already near-optimal**: one sequential pass plus small
  tails (codes, cache). Perceived "write pressure" during big merges is dirty
  page writeback competing with the read faults, not algorithmic write
  amplification.
- **Full-precision reads during the merge are the implementation's design,
  not a misconfiguration.** Three hardcoded sites, with the measured split
  from a mid-gather thread profile (266 `readFloatVector` samples, 12-source
  merge): (a) inside `gatherCandidates` — same-source neighbor scoring is
  exact-only, plus every cross-source beam result is exact-rescored
  (`(k−1)×topK` reads/node) — **155 samples / 58%**; (b) inside
  `retainDiverse` — each surviving candidate's vector is **re-read** at
  `:2433` to compute candidate-to-candidate diversity, *independently of the
  scores gather already computed* — **109 samples / 41%**; (c) the base
  node's own vector, once per output node. The beams themselves steer by
  fused ADC as intended, and the search does **no** internal reranking (the
  cross-source SSP is built with the single-arg `DefaultSearchScoreProvider`
  constructor, so `reranker == null`) — so the rescore is not doubled.
  `isDiverse` compares against cached vectors and adds no reads. There is no
  configuration that disables any of this today.

  Note the redundancy this exposes: `retainDiverse` re-reads vectors that
  `rescore()` read moments earlier for the same candidates. Caching the
  gather's already-read vectors into the diversity pass would remove ~41% of
  merge full-precision reads *with no semantic change whatsoever* — strictly
  smaller and safer than O3's deferral, and independent of it. Worth landing
  first as its own measurable step. The 2026-07-25 "rescore ≈ 0.5%
  duty" measurement was CPU duty on a cache-warm 3-source merge; on a
  cache-cold 12-source merge the same reads are ~15% of runnable samples and
  a major share of the random IOPS. O3/O5 make this explicit policy instead.

Read amplification: every source record is re-read each time any beam crosses
it. Cache-resident that costs CPU only; past the cache cliff each revisit is
a 4 KiB fault, and LRU thrashes under uniform random access.

#### The design: split the scatter-gather in the middle

**Phase A — block-swept wavefront search (sequential reads).** Invert the
loop nesting: instead of "for each output node, search k−1 sources," keep a
large set of output nodes' search frontiers pending simultaneously and sweep
each source **block-by-block in ordinal order**. When a block is resident,
service every pending frontier whose next expansion lands in it: score its
candidates there (ADC against packed records), push follow-on hops into the
buckets of *their* blocks. Beam search is iterative, so a wavefront costs
roughly hop-depth sweeps (shallow for Vamana), but each sweep is sequential.
Bounding the active query set ("cohorts") bounds memory; block scheduling is
what amortizes: at 64 MiB logical blocks a 20M-node source is ~1500 blocks,
and 100K in-flight queries × ~150 visits ≈ 10K record touches per block per
round — one sequential 64 MiB read servicing 10K would-be faults.

**The middle — bounded per-output-node candidate accumulators.** Each output
node owns a fixed-K top heap of `(source, ordinal, approxScore)` entries
(~12 B each): K=128 → ~1.5 KiB/node, ~45 GiB at 30M nodes — in-RAM on this
class of box; a block-bucketed spill is the fallback for 100M+ tiers.

**Phase B — ordinal-order finalize (sequential write, shrunken reads).** Walk
output ordinals in order; per node take its accumulated candidates and apply
the O3 policy: full-precision rescore only ~2×maxDegree survivors (or none —
pure-ADC selection is on the table now that codebooks are adopted and stable;
the recall gate decides), with those reads themselves batched block-wise;
`retainDiverse`; write the record. Same-source neighbors join the candidate
pool scored by ADC like everything else instead of their own exact reads.

**ADC-table scaling (the one new constraint).** O1 shares one decoder table
per (query, codebook-class) — fine when one node is in flight, but a
wavefront holds thousands: at ~100 KiB per query table, 100K in-flight
queries ≈ 10 GiB (acceptable), 1M ≈ 100 GiB (not). Three options, in
preference order: (a) cohort sizing keeps tables bounded (start 64–128K
queries); (b) table-free asymmetric scoring (compute subspace dots directly —
~2–4× the scoring FLOPs, zero per-query memory); (c) symmetric (SDC) scoring
via one shared codebook×codebook table (~26 MiB total, query-independent,
unbounded cohorts) with exactness recovered by the Phase-B rescore — recall
gate required.

Expected effect: read traffic collapses from `visits × 4 KiB random` to
`≈ hop-depth × cohort-count × touched-source-bytes sequential`; the beam-heap
CPU (the 50% `upHeap` share) drops with the composite budget (below); merges
stay near the write-bandwidth line regardless of k and of cache fit.

**O2 relationship.** The original O2 (one composite beam per node across k
graphs) attacks expansion *count*; the wavefront is the disk-aware way to run
it: a single global budget per output node across sources composes naturally
with block scheduling (frontiers are per-(node, source-block); the budget is
per-node). O5 = O2's composite budget + O3's deferred rescore + block-swept
execution. They are one design, not three passes.

#### What we already have — and what each piece is NOT

- **`BlockAmortizedMergeSearcher` / `BlockAmortizedReader` /
  `BlockReadScheduler`** (branch `native-byte-order`, commit `542723c0`,
  ~650 lines + tests, runtime flags, default off): density-ordered coalesced
  block reads driving a greedy search that is parity-exact with
  `GraphSearcher`. **Scope it correctly:** it coalesces the reads of ONE
  query's rounds (the compactor integration calls it per (node, source)), it
  reads **full-precision inline vectors**, and it is gated to the
  **exact-scoring** merge (`!compressedPrecision`) — it was built for the
  pre-fused era's FP-read storm and is inert under today's fused path. It is
  the right *read-engine seed* (scheduler + reader + parity discipline) but
  it is not the wavefront: no cross-node batching, no packed-record reads,
  no accumulator middle, no rescore deferral. The `searchBatch` lockstep
  mode is the germ of multi-query coalescing.
- **O1 decoder-table sharing** (landed): per-(node, class) ADC tables. The
  wavefront keeps the idea but changes the lifecycle — tables now live for a
  cohort, hence the scaling options above.
- **O3** (designed, unimplemented): unchanged as policy; becomes Phase B.
- **O4** (landed): orthogonal and prerequisite-adjacent — adopted codebooks
  make ADC scores comparable across sources (one class ⇒ one query table per
  node; SDC option only works because codebooks are shared), and the
  canonical-codes pre-encode copy already runs block-sequential.
- **Pre-encode pass / `PQRetrainer` sampling** (landed): existing
  block-sequential sweeps with explicit prefetch — precedent for the IO
  discipline, but they stream *fixed* addresses; the wavefront schedules
  *data-dependent* addresses, which is why it needs the bucket scheduler.
- **The row-pass** (merge ingest) and the epilogue: already sequential;
  unchanged by O5.

#### Phasing and gates

1. Land after the current round's O1+O4 numbers are read out (one variable
   per round holds).
2. Rebase the block-amortized trio onto `livenodes-integration`; generalize
   the reader from inline-vectors to L0 packed records; parity tests first.
3. Wavefront Phase A with cohorts + accumulators; Phase B with explicit
   rescore policy (`survivors` | `none` | `all` — `all` reproduces today's
   semantics for A/B). Explicit config:
   `cassandra.sai.vector.compaction_gather_mode = per_source_beams |
   block_wavefront` (+ cohort size, + rescore policy), REQUIRED-EXPLICIT like
   the rest.
4. Gate: `recall_postcompact` tier-by-tier vs the O1+O4 baseline; measure at
   30M×12src where today's IOPS-saturated baseline lives.

## Literature backdrop (2026-07-26 survey) and gap analysis

How the major systems merge disk ANN indexes, and where our implementation
stands against each. (Primary-source survey; citations at section end.)

**FreshDiskANN (StreamingMerge)** — the canonical blocked disk-graph merge.
Merges an in-memory delta of N inserts + D deletes into an SSD-resident
long-term index in three phases: a *sequential* block sweep repairing deleted
neighborhoods; an insert phase where each new point runs one GreedySearch
against the disk index (~100 random 4 KiB reads per insert, bounded by the
graph's α-RNG property) with backward edges buffered in an in-RAM map Δ sized
by the *change set* (O(|N|·R), ~7 GiB at 30M inserts); and a second
*sequential* sweep that patches Δ into each block and RobustPrunes overfull
neighborhoods. Two properties matter enormously for us: (1) **every distance
computation in the whole merge uses in-RAM PQ vectors — zero full-precision
reads**; (2) whole-index work is confined to exactly two sequential passes.
Measured: a 10% change to a billion-point index merges in ~10% of rebuild
time.

**DiskANN (one-time sharded build)** — partitions into *overlapping* shards
(each point in its ℓ=2 closest clusters), builds Vamana per shard in RAM,
then merges by **pure edge union**: sequential interleaved streams, dedup,
shuffle-truncate to max degree — *no distance computations at all*. The
overlap points bridge shard subgraphs, which is what makes union sufficient.
Our LSM segments are disjoint, so this exact trick is unavailable — but it
calibrates how cheap merge can be when structure supplies connectivity.

**FAISS (IVF family)** — merge is inverted-list **concatenation**, valid
precisely because the coarse quantizer and codebooks are shared and verified
compatible (`check_compatible_for_merge` reconstructs centroids and asserts
bitwise equality). Zero re-encoding, zero distance computation. HNSW has no
merge at all (rebuild only). This is O4's design validated independently:
codebook adoption + content-equality gating + conservative checks is exactly
the FAISS doctrine, applied to the code component of our segments.

**Lucene HNSW segment merges** — since 9.6, initialize the merged graph from
the **largest deletion-free segment** (ordinal-remapped) and re-insert only
the other segments' vectors (~43% faster merges, recall unchanged). Lucene
10.2 goes further: use *all* input graphs' connectivity — insert a "join set"
(~1/5 of the small graphs' vertices) normally, then seed the rest through the
small graphs' own edges with cheap low-ef insertions (1.3–1.7× further
speedup, quality parity). All in-memory at segment-write time, but the
algorithmic shape is the point.

**The in-place school** (SPFresh/LIRE posting splits, IP-DiskANN, CleANN) —
dissolves compaction into per-update repair and is architecturally disjoint
from LSM-immutable segments; noted as counterpoint, not adopted. **Starling**
is the read-side complement: block layouts that co-locate neighbors — relevant
to O5's block efficacy and to output layout, not a merge scheme.

#### Gap analysis: where our merge deviates from everything above

1. **We re-derive every node's neighborhood, symmetrically.** Our gather runs
   k−1 beam searches for *all* T output nodes — cost ∝ T×(k−1). No system in
   the literature does this. They retain the largest structure and insert
   only the delta (FreshDiskANN, Lucene), union overlap (DiskANN), or
   concatenate (FAISS). Our tier shape (one big source + eleven ~1M flush
   segments) is the *best possible* case for retain-largest: at the 30M tier
   that's ~10M insertions versus 30M full gathers before any per-node
   savings.
2. **We read full precision during merge; FreshDiskANN reads none.** Their
   entire merge runs on RAM-resident PQ. Our O4 codes tail is the missing
   piece already built: canonical codes per segment (~100 B/node → ~3 GiB at
   30M) can serve as the in-RAM distance substrate for a merge that touches
   disk only for adjacency.
3. **We have no backward-edge buffer.** Symmetric gather sidesteps it (every
   node searches for itself); an asymmetric merge needs FreshDiskANN's
   Δ-then-sequential-patch — small, bounded by delta size, and the pattern is
   proven.

#### O6 (sketch): asymmetric retain-largest merge

Adopt the FreshDiskANN shape on our surfaces: (a) delete-consolidation sweep
over the largest source (sequential; covers its dead ordinals); (b) insert
each smaller source's live node via one search against the base (distances
from in-RAM canonical codes — O4's tail loaded once; adjacency reads
block-amortized by the O5 scheduler; Lucene-10.2-style seeding from the small
segments' own edges can cut search cost further since those graphs already
know their local neighborhoods); (c) buffer backlinks in Δ (O(delta×degree));
(d) sequential patch+prune sweep writing the output, which also carries O4's
codes tail forward. Cost model: ~delta-proportional instead of
total-proportional — composes *with* O5 (the scheduler is the insert phase's
IO engine) rather than competing with it, and subsumes O3 (PQ-only distances
end-to-end, optional exact rescore for final neighborhoods as policy).
Distinct risks: output ordinal identity (our postings contract remaps
source-major — retained-base ordinals keep order, inserted nodes append;
needs a remapper variant), hierarchy layers over the union, and recall parity
(literature reports parity for both FreshDiskANN and Lucene; our
recall_postcompact gates it).

Ordering: O5's read engine is needed by O6's insert phase either way; O6
changes *what* work is done, O5 changes *how it touches disk*. Evaluate O6's
feasibility spike right after the O5 rebase rather than after full O5.

#### Measured reference points from the literature (for calibrating ours)

Published numbers worth holding next to our own measurements. Note the
hardware gap: these are 2019–2023 SSD/DRAM-constrained machines, ours is a
64-core box with 495 GB RAM and striped NVMe — treat the *ratios and shapes*
as the signal, not the absolute times.

| system | workload | result |
|---|---|---|
| FreshDiskANN | 30M inserts into ~800M index, 40 merge threads | ~8,400 s/cycle → ~3,500 inserts/s sustained |
| FreshDiskANN | steady state, 30M inserts + 30M deletes | ~16,277 s/cycle → ~1,800 ins/s + 1,800 del/s (burst 40K/s) |
| FreshDiskANN | Δ buffer sizing | 2·R·4 bytes per inserted point (~7 GiB at 30M, R=64) |
| FreshDiskANN | insert-phase random IO | ~100 × 4 KiB reads per inserted point (L=75) |
| FreshDiskANN | search latency during merge | ~5 ms idle → ~15 ms during merge; ~40 ms spikes in the sequential phases (head-of-line blocking behind large sequential IO) |
| DiskANN | 1B one-shot build | ~2 days, ~1100 GB peak RAM, avg degree 113.9 |
| DiskANN | 1B sharded build+merge (k=40, ℓ=2, R=64) | ~5 days, **&lt;64 GB RAM throughout**, 348 GB index, avg degree 92.1, ≤20% extra query latency at target recall |
| Lucene 9.6 | 500K-doc segment merge, graph-init from largest | ~655 ms → ~373 ms (~43%), recall unchanged (~0.98) |
| Lucene 10.2 | join-set seeding from all graphs | 1.28–1.33× indexing, 1.34–1.72× force-merge, quality parity |

Three calibration lessons for us:

1. **Merge is expected to cost search latency, and the sequential phases are
   the spiky ones.** FreshDiskANN's worst latency spikes come from *large
   sequential* IO starving concurrent queries — relevant to O5, which
   deliberately converts random IO into large sequential sweeps. If we adopt
   it, watch tier `concurrent_query` latency, not just merge wall time, and
   consider bounded read sizes / pacing for exactly this reason.
2. **Bounded memory is treated as a first-class merge property.** DiskANN's
   sharded path exists purely to hold peak RAM under 64 GB; FreshDiskANN
   sizes Δ by the change set, not the index. Our current merge is
   RAM-comfortable, but O5's accumulators (~45 GiB at 30M) and O6's Δ need
   the same explicit sizing discipline, with the tier ladder as the test.
3. **Nobody pays full precision during merge.** Both DiskANN merge (zero
   distance computations) and FreshDiskANN (RAM-resident PQ only) treat
   exact vector reads during merge as something to design out, not tune.

*Citations: DiskANN (Subramanya et al., NeurIPS 2019); FreshDiskANN (Singh,
Subramanya, Krishnaswamy, Simhadri, arXiv:2105.09613); FAISS wiki "Special
operations" + IndexIVF.cpp + "The Faiss Library" (arXiv:2401.08281); Lucene
PR #12050 / 9.6 CHANGES, PR #14331 / 10.2 CHANGES, IncrementalHnswGraphMerger;
SPANN (arXiv:2111.08566); SPFresh (SOSP 2023); Starling (arXiv:2401.02116);
IP-DiskANN (arXiv:2502.13826); CleANN (arXiv:2507.19802); Ponomarenko HNSW
merge algorithms (arXiv:2505.16064). Verification caveats: FreshDiskANN is
arXiv-only (no formal venue found); FAISS's when-to-retrain-on-drift guidance
is paper/folklore level, not a documented threshold. The DiskANN "edge union"
claim is verified against the paper AND `merge_shards` in microsoft/DiskANN
`cpp_main` — the implementation shuffles and truncates to max degree with no
distance computation, i.e. there is no RobustPrune in the merge. Retained
primary sources (paper texts, `merge_shards` and `IndexIVF.cpp` extracts):
`~/notes/ann-merge-literature/` on this node — kept out of this repo
deliberately.*

## Measured confirmation of gap #1 (2026-07-31, live 12-source merge)

A production-shaped merge was profiled in flight, which settles the gap
analysis above from "derived from the literature" to "measured on our own
code". Context: `path=MERGE`, 12 sources, ~19.86M rows, `enable_fused=true`,
`enable_nvq=false`, `v5_postings=true`, jvector at
`compaction-livenodes-enum-integration2` + upstream
`fix/preencode-chunked-cache` (`PreEncodedCodeCache` verified present in the
loaded jar). So none of what follows is a misconfiguration or a fallback
path — it is the intended algorithm running as designed.

**Where the CPU goes.** 112 worker frames sampled across 3 jstacks, ~40 of 64
cores busy, disk at 2%:

| share | frame | character |
|---|---|---|
| 35% | `AbstractLongHeap.upHeap` | candidate-heap maintenance |
| 22% | `PanamaVectorUtilSupport.dotProduct{,64}` | **the actual similarity work** |
| 15% | `FusedPQDecoder.<init>` | per-search score-provider setup |
| 12% | `Unsafe.copySwapMemory0` | vector-read copies |
| 12% | `processNeighbors` / `searchOneLayer` | traversal |
| 5% | `readFully` / `seek` | I/O |

Under a quarter of the machine is computing similarities. The majority is the
*machinery* of running 11 beam searches per node — heap sift-ups and
per-search provider construction — which is precisely the cost gap #1
predicts and which union-overlap (DiskANN) and retain-largest
(FreshDiskANN/Lucene) do not pay at all.

**Scale of the fan-out.** `gatherCandidates` is explicit about the asymmetry:
`gatherFromSameSource` walks the node's existing neighbor list (no search),
while `gatherFromOtherSource` calls `gs[sourceIdx].search(...)`. At 12
sources that is 1 reuse + 11 searches per node, ≈220M searches for 20M
surviving nodes.

**The searches are already quantization-driven.** `buildCrossSourceScoreProvider`
uses `FusedPQ.approximateScoreFunctionFor` for the beam with full precision
only on rerank, so the waste is *not* full-precision scanning during the beam
— it is the search structure itself. This matters for prioritization: tuning
the FP-read path cannot recover the 35%+15% that only removing the searches
recovers.

**I/O is phase-dependent — do not generalize either way.** This sample was
taken late, during the index-build tail, and shows a CPU-bound regime (2%
disk). Earlier phases of the same run were observed to be random-I/O heavy.
Both regimes are real; a profile from one phase must not be quoted as the
merge's overall character. (No `sysmon` history was captured for this run to
quantify the earlier phase — enable it next cycle.)

**De-duplication, for the record.** The merge path keys its postings map by
global ordinal handle (`ChronicleMap<Long, CompactionVectorPostings>`,
`base[src] + nodeId`) — integer keying, no vector-value comparison. Full
value-keyed de-duplication (`ChronicleMap<VectorFloat<?>, …>`, hashing every
vector) exists only on the REBUILD path in Cassandra's `CompactionGraph`, and
is one more reason single-input compactions are disproportionately expensive.
The merge path does not de-duplicate by value and should not start.

### Consequence for the roadmap

This measurement does not add a new option; it re-prices the ones already
listed. Retain-largest is the literature-backed fit for disjoint LSM segments
(DiskANN's union-overlap being unavailable to us, per the section above), and
the tier shape here — one large source plus eleven ~1M flush segments — is
its best case: ~10M insertions against a retained 20M-node base, versus 30M
symmetric gathers each fanning out 11 ways. The profile says the searches are
not a constant factor to be tuned down but the dominant term to be removed.

## Explicit configuration (fail-fast, no defaults)

All recent behavior switches are REQUIRED-EXPLICIT end to end. jvector: a
fused `compact()` throws unless `setPqCodeCacheConfig`, `setCodebookPolicy`,
and `setDecoderTableSharing` were called (plus `setCodebookDriftMargin` for
ADOPT_CHECKED). Cassandra (`VectorFeatureFlags`, validated at index
construction so a misconfigured node refuses startup):

| property (`cassandra.sai.vector.`) | values |
|---|---|
| `pq_code_cache_enabled` | true / false |
| `amortize_pq_encoding` | true / false |
| `serialize_flush_pq` | true / false |
| `compaction_decoder_sharing` | true / false (O1) |
| `compaction_codebook_policy` | refine / adopt_uniform / adopt_checked (O4) |
| `compaction_codebook_drift_margin` | float in [0,1), required for adopt_checked |
| `pq_code_persistence` | true / false (O4) |

Additionally the whole `cassandra.sai.vector*` property namespace is scanned:
any key not declared in `CassandraRelevantProperties` fails startup — the
class of bug where a misspelled property silently no-ops (the
`sai.vector.latest.version` incident) is now structurally impossible.

## Measured (2026-07-26, O1+O4 active, fresh lineage, 384-dim, degree 32)

Merge sub-phases from one tier ladder (row-pass = ordinal-identity ChronicleMap
ingest; gather+write = compactLevels; k=2–3 for 1–10M, k=12 for 20M):

| tier | k | row-pass | gather+write | total | min/M |
|---:|---:|---:|---:|---:|---:|
| 1–10M | 2–3 | 8.3 s/M (27%) | ~22 s/M | — | **0.42–0.51 flat** |
| 20M | 12 | 193 s | 1217 s | 1411 s | **1.18** |

vs the pre-O1/O4 baseline: 0.47–0.82 min/M (k≤3), 1.9–2.1 min/M (k=12) —
≈1.7× at k=12, and the per-M cost is now **size-independent** at fixed k.
O4 evidence: every merge "1 codebook class", pre-encode = pure copy
(20M codes in 0.5 s), no retrain phase anywhere. Epilogue at 20M: footer CRC
17 s, codes tail 1 s, postings 24 s. Disks ~1% util (page cache), writer
thread idle — the merge is beam-compute-bound.

Thread-sampled gather profile (jstack, ~1900 RUNNABLE build-pool samples,
k=12): `NodeQueue.push`/`upHeap` via `searchOneLayer` **50%**; full-precision
`readFloatVector` **14–15%** (O3's share); `processNeighbors` 7%; packed-code
reads 6%; ADC `similarityTo` + dot products ~10%; `newDecoder` **2%** (was
~40% pre-O1 — O1 confirmed). Remaining k-multiplier ≈2.3× (1.18 vs 0.51
min/M) lives in the k−1 independent beams — O2's territory exactly.

Ingest (MutationStage, ~7000 samples): graph-insert similarity math ~85%
(`dotProductPreferred` 75%, neighbor processing/diversity ~10%), amortized PQ
encode (`closestCentroidIndex`) **~2%**, trie put ~3% — insert cost IS the
graph build, the amortized encode is noise (as designed). Flush (per ~1M-row
segment): median 50 s = pre-graph 40 s (builder.cleanup refinement — same
dot-product math, single flush thread) + graph write 10 s (serdes); codes
tail adds <1 s.

## Composition and projected totals (12-source, 30M-class merge)

| stage | today | O1 | O1+O3 | O1+O2+O3 |
|---|---:|---:|---:|---:|
| table builds | 40% | ~4% | ~4% | ~4% |
| beam expansion | ~35% | ~35% | ~35% | ~5–10% |
| exact rescores | ~15% | ~15% | ~1% | ~1% |
| other (diversity, IO, write) | ~10% | ~10% | ~10% | ~10% |
| **projected wall vs today** | 1.0 | ~0.65 | ~0.50 | **~0.2–0.25** |

i.e. the 62-minute 12-source 30M merge lands near the 2-source curve
(~25 min), which is the design goal: **make merge cost independent of how
the input happened to be segmented.**

## Risks and validation

- **Recall**: the per-tier `recall_postcompact` measurement is the gate;
  run the same ladder and compare tier-by-tier against the current
  baseline's ledger. O1 is exactly-equal-by-construction; O2/O3 need the
  gate.
- **Codebook divergence** (O1): merges spanning a retrain boundary have >1
  class; the design degrades gracefully to per-class builds (worst case =
  today).
- **Ordinal spaces** (O2): the frontier stores (source, ordinal) pairs; all
  existing per-source machinery (liveNodes, remappers, views) is reused
  unchanged.
- **Phasing**: land O1 alone first (measurable, riskless), then O4
  (adopt_uniform → adopt_checked, code persistence), then O5 — the
  block-wavefront restructure that subsumes O2 and O3 in one disk-aware
  design (see its section). Each phase is a separate benchmark round on the
  same tier ladder; the explicit per-run flags select exactly one new
  variable per round.
- **Adoption quality** (O4): adopting instead of refining freezes the
  codebook across merges. On stationary data warm-start Lloyd's is refining an
  already-converged codebook (near-noise centroid movement bought with a full
  re-encode); on drifting data ADOPT_CHECKED bounds the error regression by
  the margin. recall_postcompact remains the gate for both.

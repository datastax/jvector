# Compaction seam conversion (illustration)

`integration-robustness` introduced the embedding seams as standalone types, with no
consumer:

| Seam | Types | Role |
| --- | --- | --- |
| Execution | `graph.ParallelExecutor`, `graph.EmbeddedExecutionContext` | host supplies the pool (or caller-runs); no work escapes to `ForkJoinPool.commonPool()` |
| Progress + throttle | `util.work.ProgressLimiter` (`ProgressTracker` + `WorkLimiter`), `WorkStage` | progress reported up, write bandwidth admitted down, cancellation checkpoints |
| Output | `graph.disk.CompactionDestination`, `disk.SeekableSink` | compaction output written through a caller-owned channel |
| Runtime mode | `util.RuntimeMode` | diagnostic-only work gated off by default |
| Reader safety | `OnDiskGraphIndex` bounded reads (already landed) | a stale/out-of-range offset fails diagnosably instead of faulting |

This branch converts the **general** build/quantize calling conventions onto the execution
seam as code:

- `graph: convert GraphIndexBuilder to the ParallelExecutor seam`
- `quantization: accept ParallelExecutor for PQ/NVQ/BQ encode`
- `graph: EmbeddedExecutionContext as the single execution carrier`

The **compactor** conversion is described here rather than carried as code: in the current
lineage the compactor's seam-wiring is inseparable from the ten compaction-algorithm
commits (a single ~1,100-line diff), so converting it cleanly is the job of the compaction
work that will be rebuilt on top of `integration-robustness`. This note is the target that
work should hit — each section is `before` (today's calling convention on `main`) and
`after` (the seam-based convention), using the real signatures from the integration lineage.

---

## 1. Execution — a host pool instead of the common pool

**Before.** The compactor reaches for a process-wide pool internally, so compaction work
competes on `ForkJoinPool.commonPool()` / `PhysicalCoreExecutor.pool()` with everything else
in the host.

**After.** The primary constructor accepts any `Executor` and only falls back to the shared
pool when none is supplied. `taskWindowSize` bounds in-flight batches (the compactor drives
its merge through an `ExecutorCompletionService` over this executor).

```java
public OnDiskGraphIndexCompactor(
        List<OnDiskGraphIndex> sources,
        List<FixedBitSet> liveNodes,
        List<OrdinalMapper> remappers,
        VectorSimilarityFunction similarityFunction,
        Executor executor,          // host pool; null => PhysicalCoreExecutor.pool()
        int taskWindowSize)         // bound on concurrent in-flight batches
```

`EmbeddedExecutionContext` is where a host passes its pool once; `newCompactor(...)` wires
the compactor to the carried `mergeExecutor()`:

```java
public OnDiskGraphIndexCompactor newCompactor(List<OnDiskGraphIndex> sources,
                                              List<FixedBitSet> liveNodes,
                                              List<OrdinalMapper> remappers,
                                              VectorSimilarityFunction similarityFunction,
                                              int taskWindowSize) {
    return new OnDiskGraphIndexCompactor(sources, liveNodes, remappers,
                                         similarityFunction, mergeExecutor(), taskWindowSize);
}
```

`callerRuns()` collapses all three executor roles onto the calling thread, so a memtable
flush can run the whole merge on its own flush-writer thread with no worker pool at all.

---

## 2. Progress + throttle — `ProgressLimiter`

**Before.** `compact()` runs opaque: no progress until it returns, and its (often dominant)
internal write bandwidth is outside any host throughput budget.

**After.** A single opt-in limiter, defaulting to a no-op, is installed per operation:

```java
private volatile ProgressLimiter limiter = ProgressLimiter.UNLIMITED;

/** null restores ProgressLimiter.UNLIMITED. Returns this for chaining. */
public OnDiskGraphIndexCompactor setProgressLimiter(ProgressLimiter limiter) { ... }
```

Inside `compact()`, at each phase boundary the compactor calls the limiter both ways, and
each call doubles as the cancellation checkpoint:

- **up:** `limiter.onProgress(stage, completed, total)` — advances `nodetool compactionstats`
  / `system_views.sstable_tasks` *while* the merge runs, instead of jumping 0% → 100%.
- **down:** `limiter.acquire(bytes)` before a write batch — admits the merge's write
  bandwidth against the host's shared compaction rate limiter.
- **cancel:** both entry points throw if the host requested stop; jvector drains its in-flight
  workers before `compact()` unwinds, so no source read survives the cancellation.

Phases are scoped with `startPhase(WorkStage)` so each stage (extract, retrain, link, write,
footer) is separately observable.

The limiter is deliberately **not** carried on `EmbeddedExecutionContext`: throttle/progress
is per host operation, so it is set on the returned compactor per call.

---

## 3. Output virtualization — `CompactionDestination`

**Before.** The compactor allocates and opens its own output file.

**After.** `compact(CompactionDestination)` writes the graph body through a caller-owned
target, so the host can hand it a slot inside a larger container (e.g. an SAI component after
a reserved header) with no temp file and one write of the body:

```java
public long compact(CompactionDestination destination) throws IOException {
    try (CompactionDestination.Target target = destination.open()) {
        // ...write the compacted graph into target.file() at target.startOffset()...
        target.commit(bodyLength);   // success: body durable; embedder finalizes its own footer
        return bodyLength;
    }                                // close() always runs; no commit() => aborted, partial output discarded
}
```

The embedder computes its own footer/checksum over the body via
`SeekableSink.over(channel, target.startOffset())`. The path-based `compact(Path)` /
`compact(Path, long)` entry points remain for callers that still want jvector to own the file.

---

## 4. PQ retrain through the executor

**Before.** Retrain during compaction submits to the common pool — the historical "retrain
leak" where work escaped the host's budget.

**After.** Retrain takes the compute executor, closing the leak:

```java
// PQRetrainer
public ProductQuantization retrain(VectorSimilarityFunction sim,
                                   ParallelExecutor simd, ParallelExecutor parallel);
public ProductQuantization retrain(VectorSimilarityFunction sim, ProductQuantization basePQ,
                                   ParallelExecutor simd, ParallelExecutor parallel);
```

`EmbeddedExecutionContext.retrainPQ(retrainer, sim[, basePQ])` forwards the carried compute
executor. (The general PQ train/refine/encode calls already route through the executor on
this branch — see `EmbeddedExecutionContext.trainPQ/refinePQ/encodePQ`.)

---

## 5. Parallel (IO-bound) graph writer

**After.** The parallel writer runs on the host's IO executor rather than a default pool:

```java
new OnDiskParallelGraphIndexWriter.Builder(graph, outputPath).withExecutor(io);
```

`EmbeddedExecutionContext.newParallelWriter(graph, path)` supplies `ioExecutor()`.

---

## 6. Compactor-side memory safety (lands with this conversion)

The reader-side guard — bounded record reads in `OnDiskGraphIndex`, plus the mmap-lifecycle
`close()` contract on `ReaderSupplier` / `SimpleMappedReader` — is already on
`integration-robustness`. The remaining, compactor-local guards belong with this conversion
because they live inside `compact()`:

- **drain on unwind** — on any exception or host cancel, in-flight merge workers are drained
  before `compact()` returns/throws, so no worker read outlives the call (no use-after-unmap
  when the host reclaims source mappings).
- **truncate reused outputs** — an output being reused is truncated before the new body is
  written, so stale tail bytes past the new body can never be mistaken for graph data.

---

## What `EmbeddedExecutionContext` regains after the conversion

The trimmed context on this branch carries `mergeExecutor()` / `ioExecutor()` but omits the
compactor/writer/retrain factory methods, because they depend on the signatures above. Once
the compactor is converted, they return as thin wiring:

```java
newCompactor(...)      -> new OnDiskGraphIndexCompactor(..., mergeExecutor(), window)
newParallelWriter(...) -> new ...Writer.Builder(graph, path).withExecutor(ioExecutor())
retrainPQ(...)         -> retrainer.retrain(sim, parallelExecutor(), parallelExecutor())
```

## What stays for the clean compaction work

Everything above is the **calling-convention** surface. The body of `compact()` — layer
enumeration, cross-source linking, retain-largest, pre-encode caching, and the rest of the
ten algorithm commits — is the compaction work proper, to be rebuilt cleanly on top of
`integration-robustness` against exactly these seams.

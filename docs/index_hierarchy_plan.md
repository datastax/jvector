# Index / IndexBuilder Hierarchy — Implementation Plan

Status: **implemented** on branch `index_hierarchy`, in three passes: the
hierarchy itself (§1–§10), a follow-up adding incremental construction and
API cleanups (§11), and a simplification pass that replaced the validating
builder and `MutableHnswIndex` with a smaller builder centered on ease of use
(§12). Where §12 supersedes an earlier section, that section says so; the
earlier text is kept as the record of what was tried and why.
Source of the approach: [`four_perspectives.md`](four_perspectives.md) (the aci
"one configuration interface, four perspectives" demonstrator), kept alongside
this plan for reference. The aci files it names are in the aci project, not in
this repository.

Two decisions shape what follows. No IVF-specific parameter (such as
`nProbes`) is defined until the IVF developers specify them. And `IvfIndex`
lives in `jvector-base` alongside `GraphIndex`, not in the new `jvector-api`
module as an earlier draft proposed, for the structural reason given in §4.

## 1. What maps from the source document, and what doesn't

The source document describes **aci**: a system with several backing index
types, one generic contract (`Index`/`IndexSpec`), type-first fluent
builders, a declarative parameter catalog, a tiered verdict system
(verboten/suboptimal/acceptable/prescriptive), named recipes, and a JSON/SQL
config surface for embedding systems and their end users (perspectives 3
and 4).

jvector-base is a library, not a standalone system with its own config
service or SQL surface — perspectives 3 and 4 belong to whatever embeds
jvector (Cassandra, Astra, etc.), not to jvector itself. What *does*
transfer cleanly, and is the actual scope of this plan:

- **Perspective 1 (vector system builder)** → jvector maintainers, adding
  HNSW today, IVF next, and other backing types later.
- **Perspective 2 (API user)** → jvector's Java callers: pick the type first
  via a type-specific builder (wrong parameter = compile error), get a
  generic `Index` handle back when the caller doesn't need to know the
  backing, and recover type-specific behavior safely when they do.

Confirmed out of scope for jvector-base: a declarative parameter catalog /
rule engine, a `Catalog` of named live indexes, JSON descriptor
round-tripping, and the verboten/suboptimal/prescriptive verdict tiers.
Revisit only if a concrete embedder need shows up later.

## 2. Prior state (before this work)

- `Index` — marker interface (`Accountable`, `AutoCloseable`, `searcher()`),
  plus static factories `Index.hnswBuilder()` / `Index.ivfBuilder()`. A
  commented-out sketch of `Index.withRecipe(...)` was present but unused.
- `IndexSearcher` — empty marker interface. `GraphSearcher implements
  IndexSearcher`.
- `GraphIndex extends Index` — the one real backing type.
- `HnswIndexBuilder` — fully implemented, including hand-rolled "collect
  every missing value, throw one exception naming all of them" validation.
- `IvfIndex` / `IvfIndexBuilder` — empty stubs (no fields, no interface
  implemented, `build()` returned `null`).
- `IndexRecipe` — a flat 4-value enum, not wired to anything.
- `IndexBuilder` (interface) had been added and then deleted; its one
  static factory method had moved onto `Index` directly.

## 3. Decisions from review

1. **IVF construction parameters**: unknown beyond a vague sense that
   `nProbes` (an `int`) is IVF-specific; the real parameters have to come
   from the IVF developers. Until then `nProbes` is **left out entirely** —
   not added to `IvfIndexBuilder`, `IvfSearcher`, or anywhere else — and
   nothing else IVF-specific is guessed at either.
2. **IVF persistence**: needed eventually, format TBD, and the two-pass /
   inline-write strategy `PersistableGraphIndex` uses for graphs is
   explicitly *not* the plan for IVF. The seam is reserved conceptually
   (§5.5) but no interface was added for it yet — there's nothing to name
   safely until the format exists.
3. **Recipes**: scaffolding implemented — enum types + `applyRecipe(...)`
   wired end-to-end on both builders — with every constant currently
   throwing `UnsupportedOperationException` until real formulas exist.
4. **Perspective 3 scope**: stays the embedder's responsibility, as in §1.

## 4. `jvector-api`: the public-contract module

The question here was whether a separate module holding just the
interfaces — so a different implementation could be wired in later without
touching callers — makes sense for JVector. It does, and a small, real first
cut of it is what was built.

### Why it's feasible

Two precedents already exist for exactly this shape:
- The source doc's own system splits `aci-subject` (the contract) from
  `aci-subject-mock` (a backing implementation of that contract) as
  separate modules.
- This repo already does it for a different axis: `jvector-base` (portable,
  Java 11) is the canonical implementation, `jvector-twenty` depends on it
  and swaps in Java 20 vectorized versions of low-level pieces, and
  `jvector-multirelease` assembles both into one multi-release jar.

### Why it can't be *total* — and where `IvfIndex` lives

`Index` itself is tiny and clean — `Accountable`, `AutoCloseable`,
`searcher(): IndexSearcher` — a genuine fit for a pure api module.
`GraphIndex` is not: its nested `View`/`ScoringView`/`NeighborProcessor`/
`IntMarker` types are internal graph-traversal machinery used only by
`GraphSearcher` and the on-heap/on-disk implementations, and `ScoringView`'s
methods take `VectorSimilarityFunction`, whose enum bodies call straight
into `VectorUtil` — the exact thing `jvector-base`/`jvector-twenty` already
swap per JDK release. So `GraphIndex` conflates the "public handle" a
caller holds with an "internal traversal contract" only jvector's own
search algorithm should touch; untangling that is real, separate work, out
of scope here. For now, `GraphIndex` stays in `jvector-base`.

**An earlier draft put `IvfIndex` in `jvector-api`, which doesn't work.**
The reasoning for `GraphIndex` staying in `jvector-base` isn't only about
the traversal-internals leak — it's also, more fundamentally,
that its covariant `searcher()` override returns `GraphSearcher`, a
concrete algorithm class. For an interface to declare a method returning
`GraphSearcher`, `GraphSearcher` must be visible at that interface's own
compile time — which means the interface can't live in a module that
`GraphSearcher`'s module depends on. The exact same constraint applies to
`IvfIndex`: it covariantly returns `IvfSearcher`, so `IvfIndex` must live
alongside `IvfSearcher` — in `jvector-base` — not in `jvector-api`. This was
caught before implementing (would have been a circular-dependency compile
error otherwise) and fixed: both `GraphIndex`/`GraphSearcher` and
`IvfIndex`/`IvfSearcher` live in `jvector-base` as pairs, mirroring each
other.

The "don't leak internal traversal machinery onto the public interface"
principle is a *separate* concern from module placement, and
still holds for `IvfIndex` even though it's in `jvector-base`: keep it to
the handle surface a caller actually needs (`searcher()`, `Accountable`,
`close()`, future read-only descriptors), not whatever an internal search
algorithm needs to walk centroids/posting lists.

### What actually landed in `jvector-api`

- `io.github.jbellis.jvector.index.Index`
- `io.github.jbellis.jvector.index.IndexSearcher`
- `io.github.jbellis.jvector.index.HnswRecipe` / `IvfRecipe` (replacing the
  old flat `IndexRecipe`, §5.4)
- `io.github.jbellis.jvector.util.Accountable`
- `io.github.jbellis.jvector.annotations.Experimental` (moved from
  `jvector-base` in the follow-up, §11.8)

`GraphIndex` and `IvfIndex` (and their respective `Searcher` types) both
stay in `jvector-base`, extending `jvector-api`'s `Index`/`IndexSearcher` —
the correct dependency direction (impl depends on api).

### Static factories moved

`Index.hnswBuilder()` / `Index.ivfBuilder()` used to live on `Index` itself
and directly referenced `HnswIndexBuilder`/`IvfIndexBuilder`. Since `Index`
now lives in `jvector-api` and the builders stay in `jvector-base` (they
construct concrete implementation objects), `Index` can no longer reference
them. Moved to a new facade in `jvector-base`:

```java
// io.github.jbellis.jvector.index.Indexes  (jvector-base, impl module)
public final class Indexes {
    public static HnswIndexBuilder hnswBuilder(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf) { ... }
    public static HnswIndexBuilder hnswBuilder(BuildScoreProvider bsp, int dimension) { ... }
    public static IvfIndexBuilder ivfBuilder() { return new IvfIndexBuilder(); }
}
```

Callers now write `Indexes.hnswBuilder(...)...` instead of
`Index.hnswBuilder()...`. (The HNSW factories took no arguments until §12,
which moved the scoring inputs into them.) Confirmed before the rename that nothing
outside the stub code itself called `Index.hnswBuilder()`/`ivfBuilder()`
yet, so this was a safe, consequence-free rename.

### Module mechanics (as implemented)

- New `jvector-api/pom.xml`, same shape as `jvector-base`'s (parent +
  artifactId only — `--release 11` and everything else is inherited from
  the parent's `<build><plugins>`, not redeclared). Added
  `<module>jvector-api</module>` to the parent pom, listed first.
- `jvector-base/pom.xml` gets a `<dependency>` on `jvector-api`.
- `jvector-twenty` picks up `jvector-api` transitively through its existing
  dependency on `jvector-base`; untouched — it's swapping SIMD backends, an
  orthogonal axis.
- `jvector-multirelease`'s assembly (`src/assembly/mrjar.xml`) — **checked,
  and it was broken**: its first `moduleSet` explicitly listed only
  `io.github.jbellis:jvector-base` (with `includeDependencies=false`), so
  `jvector-api` was silently left out of the shaded `jvector` jar entirely.
  Confirmed by building the jar and inspecting it: `GraphIndex.class` was
  present but `Index.class`, `IndexSearcher.class`, `Accountable.class`,
  `HnswRecipe.class`, and `IvfRecipe.class` were all missing —
  `NoClassDefFoundError` waiting to happen for anyone loading `GraphIndex`
  from the published artifact. Fixed by adding
  `io.github.jbellis:jvector-api` as a second `<include>` in that same
  `moduleSet` (same tier as `jvector-base` — both are plain Java 11 code,
  no `META-INF/versions/` treatment needed). Rebuilt and verified with
  `javap` against the standalone jar: `GraphIndex extends Index` now
  resolves correctly. No longer an open item.

## 5. Design goals

1. Picking the index type is a compile-time decision: a disjoint parameter
   on the wrong builder (`.nlist()` on `HnswIndexBuilder`) doesn't exist to
   call.
2. A caller holding the concrete type from its own builder never needs to
   cast to reach type-specific behavior (covariant return types do this for
   free in Java).
3. A caller holding only the generic `Index` (heterogeneous collection,
   code that doesn't know/care which backing) can still recover
   type-specific behavior safely — via `instanceof` narrowing to the
   backing's own interface (`GraphIndex`, `IvfIndex`), not casts to a
   concrete class.
4. Every builder validates the same way: collect every problem, report them
   together. Adding a new builder shouldn't mean re-inventing that
   bookkeeping. (Relaxed in §12.1: with defaults for every HNSW setting there
   is little left to aggregate, so HNSW now relies on `GraphIndexBuilder`'s
   own checks.)
5. Adding a third backing type later touches a small, enumerable set of
   places (§7), and nothing upstream of that (existing HNSW/IVF code, the
   `Index` contract) needs to change to accommodate it.
6. The public contract (`Index`, `IndexSearcher`, recipes) stays free of
   implementation-specific dependencies, so the `jvector-api` boundary from
   §4 stays real instead of eroding the first time something needs "just
   one more thing" from the impl side.

## 5.1 `Index` — no new abstract methods

Considered adding an `IndexType type()` accessor (enum tag) for dispatch,
rejected: it would be a second source of truth alongside the actual type
hierarchy (`GraphIndex`, `IvfIndex`), and Java's own type system already
gives exhaustive-enough dispatch via `instanceof` on those interfaces.
Revisit if IVF's on-disk format (once designed) ends up wanting a type
discriminator that isn't naturally a Java type.

Static factories live on `Indexes` in `jvector-base` (§4).

## 5.2 `IndexBuilderValidation` (new, `jvector-base`)

> **Superseded by §12.1:** `IndexBuilderValidation` was removed.
> `HnswIndexBuilder` no longer has required values to collect, and
> `IvfIndexBuilder` checks its two inputs inline with the same message format.

`HnswIndexBuilder.build()` hand-rolled the "collect missing required values
into a list, throw one `IllegalStateException` naming all of them" pattern.
`IvfIndexBuilder` needed the exact same pattern — two real call sites
justified extracting it:

```java
// io.github.jbellis.jvector.index.IndexBuilderValidation
public final class IndexBuilderValidation {
    public IndexBuilderValidation require(String name, Object value) { ... }
    public IndexBuilderValidation requireCondition(String name, boolean present) { ... }
    public void throwIfAny(String builderDescription) { ... }
}
```

`require` covers a plain "must be non-null" field; `requireCondition`
covers a field that's only required unless some other field covers for it
(e.g. HNSW's `maxDegree`/`maxDegrees`, not required when
`withExistingGraph()` is set) — the exact shape `HnswIndexBuilder`'s
original hand-rolled checks already needed. `HnswIndexBuilder` was
refactored to use it (message text preserved); `IvfIndexBuilder` uses it
from the start.

Still no shared `IndexBuilder<T extends Index>` marker interface — no
caller needs to hold "a builder, type unknown yet."

## 5.3 `IndexSearcher` — tightened return types

```java
// GraphIndex.java — was:
default IndexSearcher searcher() { return new GraphSearcher(this); }
// now (covariant return):
default GraphSearcher searcher() { return new GraphSearcher(this); }
```

Same treatment for `IvfIndex.searcher()`, declared to return `IvfSearcher`.
(Follow-up: `IndexSearcher` is no longer an empty marker; it now extends
`java.io.Closeable`, see §11.4.)
A caller holding only `Index` still sees `IndexSearcher` from
`Index.searcher()`'s declared type, and narrows via `instanceof GraphIndex`
/ `instanceof IvfIndex` first — the Java-11-safe version of the source
doc's sealed-`Reader`-hierarchy switch (§8).

## 5.4 Recipes — scaffolding implemented, values still pending

`IndexRecipe` (the old flat enum mixing `HNSW_*`/`IVF_*` values — exactly
the cross-type mixup the source doc's rules exist to prevent) was replaced
by two enums in `jvector-api`:

```java
public enum HnswRecipe { HIGH_RECALL, HIGH_PERFORMANCE }
public enum IvfRecipe  { HIGH_RECALL, HIGH_PERFORMANCE }
```

Each builder got a typed `applyRecipe(...)`:

```java
HnswIndexBuilder.applyRecipe(HnswRecipe.HIGH_RECALL)
IvfIndexBuilder.applyRecipe(IvfRecipe.HIGH_RECALL)
```

Both currently throw `UnsupportedOperationException` unconditionally — the
mechanism is wired end-to-end, but no recipe has real fixed-value formulas
yet, not even HNSW's. (Since then, `HnswRecipe.DEFAULT` has been added and is
defined: it restates the builder's defaults. See §12.4.) Filling those in later
is a small, isolated change to each `applyRecipe` body, not a redesign.

One deliberate deviation from the source doc, unrelated to phasing: aci's
recipe builders expose *only* the free parameters (a distinct builder shape
per recipe); doing that in Java per recipe isn't worth the extra classes
right now, so once real, `applyRecipe(...)` will just pre-set fields on the
same fluent builder, and the other setters remain technically callable.

## 5.5 IVF: the stubs, as filled in

- **`IvfIndex`** (`jvector-base`, `io.github.jbellis.jvector.ivf`) is now
  an interface: `public interface IvfIndex extends Index`, with covariant
  `IvfSearcher searcher()`. No accessors beyond that yet (e.g. no `nlist()`)
  — nothing IVF-specific is confirmed. No concrete implementation exists;
  there's nothing to build yet.
- **`IvfSearcher`** (`jvector-base`, same package) is, for now, also just
  an interface extending `IndexSearcher` — no methods. Unlike
  `GraphSearcher` (a concrete algorithm class), there's no IVF search
  algorithm yet to make it concrete, and nothing about `nProbes` or any
  other query-time option was added (§3.1). A concrete
  implementing class arrives alongside the real algorithm.
- **`IvfIndexBuilder`** (`jvector-base`) got the construction inputs every
  backing needs, mirroring `HnswIndexBuilder` exactly at the time (§12 later
  moved HNSW's scoring inputs into `Indexes.hnswBuilder(...)`; IVF's are still
  `withXxx` setters): `withVectorValues`,
  `withScoreProvider`/`withSimilarityFunction` (mutually exclusive, same
  message), `withSimdExecutor`/`withParallelExecutor`. Nothing
  IVF-specific (`nlist` included — never actually confirmed as a real
  parameter here, only assumed from the source doc's own example) was
  added. `build()` validates those common inputs via
  `IndexBuilderValidation` (§5.2), then throws
  `UnsupportedOperationException` explaining that IVF's own construction
  parameters and backing implementation don't exist yet. This is honest
  scaffolding rather than the previous silent `return null`.
- **Persistence**: not touched — no `PersistableIvfIndex` or
  similar was added. Per §3.2, there's nothing safe to name yet since even
  the interface shape depends on decisions (two-pass vs. something else)
  that haven't been made. Revisit once the on-disk format exists.
- `Indexes.ivfBuilder()` returns the real `IvfIndexBuilder`.

Blocked on the IVF developers' input: the concrete parameter list for
`IvfIndexBuilder` (§3.1) — everything above this point (the interface,
searcher, and common-input builder shape) didn't need to wait for that, and
is ready for those parameters to be added to once they're known.

## 6. Validation approach, right-sized

> **Superseded by §12.1:** HNSW settings are now validated by
> `GraphIndexBuilder` when the graph is built (first error wins), plus the
> existing-graph conflict check in `HnswIndexBuilder`. Only `IvfIndexBuilder`
> still aggregates.

- Per-builder aggregate validation (§5.2) — every builder collects and
  reports all problems at once.
- Coupled constraints specific to one type stay in that type's own
  `build()`, same as `HnswIndexBuilder` always did. No generic rule engine.
- No verboten/suboptimal/prescriptive verdict tiers inside jvector-base —
  that belongs to whatever embeds jvector, not the library (§3.4).

## 7. Checklist: adding a fourth backing type later

1. New package, new `XxxIndex extends Index` interface + concrete impl(s),
   both in `jvector-base` (not `jvector-api` — see §4's correction; only
   move a backing's index interface to `jvector-api` if it *doesn't* need a
   covariant `searcher()` override, which none realistically won't).
2. New `XxxSearcher implements IndexSearcher` (`jvector-base`), returned
   covariantly from `XxxIndex.searcher()`.
3. New `XxxIndexBuilder` (`jvector-base`), taking the inputs it can't do
   without as arguments to its `Indexes` factory and everything else as
   `withXxx` settings with defaults (§12).
4. New static factory `Indexes.xxxBuilder()`.
5. (Optional) `XxxRecipe` enum (in `jvector-api` — pure value type, no
   algorithm dependency) + `applyRecipe(...)` on the new builder.
6. Any place that pattern-matches over backing types (there should be very
   few) gets a new `instanceof` branch; see §8 for how to catch a missed
   one.

Nothing in `jvector-api`'s `Index`/`IndexSearcher`, or the other backings'
code, needs to change.

## 8. Java 11 constraint: no sealed hierarchy, so what replaces exhaustiveness checking?

`jvector-api`/`jvector-base` target Java 11, so a compiler-checked
exhaustive `switch` over a sealed `Reader` hierarchy (the source doc's
approach) isn't available. Substitute, still not needed yet (only HNSW is
real): once a second real place in the codebase must branch over all known
backing types, add a small architecture test that lists the known concrete
`Index` types and fails if one isn't accounted for at that dispatch point.

## 9. Sequencing — what was done, in order

1. **Module split (§4)**: created `jvector-api`; moved `Index`,
   `IndexSearcher`, `Accountable` into it, and added `HnswRecipe`/
   `IvfRecipe` there directly (replacing `IndexRecipe`); added the
   `jvector-base` → `jvector-api` dependency; added `Indexes` to
   `jvector-base` with the two static factories.
2. Added `IndexBuilderValidation`; refactored `HnswIndexBuilder` to use it
   (both later removed, §12.1).
3. Fixed `GraphIndex.searcher()`'s return type to `GraphSearcher`.
4. Defined `IvfIndex`/`IvfSearcher` in `jvector-base` (corrected from the
   round-2 plan of putting `IvfIndex` in `jvector-api` — see §4) — seam
   only, no clustering logic, no `nProbes`.
5. Filled in `IvfIndexBuilder`'s common (backing-agnostic) construction
   inputs using `IndexBuilderValidation`; `build()` throws
   `UnsupportedOperationException` pending real IVF parameters. **Still
   blocked on the IVF developers** for the actual parameter list.
6. Wired recipe scaffolding: `HnswRecipe`/`IvfRecipe` + `applyRecipe(...)`
   on both builders, placeholder bodies.
7. Architecture/exhaustiveness test: still deferred, no second dispatch
   point exists yet.

Also changed the return type of `HnswIndexBuilder.build()` from `Index` to
`GraphIndex` (since narrowed further to `PersistableGraphIndex`, §11.3) (and `IvfIndexBuilder.build()` is declared to return
`IvfIndex`) — not explicitly called out in earlier rounds, but a direct
consequence of goal 2 (§5): a caller using the concrete builder gets the
concrete backing-index type back, no cast needed, consistent with what
§5.5 already said `IvfIndexBuilder.build()` should do.

## 10. Remaining open items

- **Recipe values and IVF** — still `@Experimental`. Only
  `HnswRecipe.DEFAULT` is defined (§12.4).
- **Consumer migration** — Cassandra and OpenSearch still call
  `GraphIndexBuilder` directly; moving them to `Indexes.hnswBuilder(...)` is
  work in those repositories. §12.6 maps each call site.
- **Legacy entry points still take the package-private `MutableGraphIndex`**
  — `GraphIndexBuilder`'s existing-graph constructor and
  `GraphIndexBuilder.builder(bsp, dimension, MutableGraphIndex)`. Callers
  can pass an `OnHeapGraphIndex`, but can't name the parameter type.
  `HnswIndexBuilder.withExistingGraph` doesn't have this problem (§11.2).
- **PQ codes from a compressed build aren't exposed.** A builder using
  `withCompressionType(PQ)` trains a codebook internally, but on-disk layouts
  that store PQ codes (fused PQ, separate PQ vectors) need the caller's own
  `ProductQuantization`. An accessor on the builder would let those layouts
  reuse it.
- **Saving a graph to continue later needs a cast**:
  `((OnHeapGraphIndex) graph).save(out)`, since `save` is only on
  `OnHeapGraphIndex`.
- **IVF construction parameters** — needed from the IVF developers before
  `IvfIndexBuilder` can do anything beyond validate common inputs and
  refuse (§3.1, §5.5).
- **IVF persistence** — no interface added yet; revisit once the on-disk
  format is decided (§3.2, §5.5).

`jvector-multirelease` assembly wiring (previously listed here as
unverified) was checked, found broken, and fixed — see §4.

## 11. Follow-up pass: incremental construction and API cleanups

Writing a fuller `IndexApiExample` against real Cassandra and OpenSearch
usage exposed a gap that blocked merging: `HnswIndexBuilder.build()` only
did a one-shot batch build, while both consumers build graphs
incrementally. This follow-up closes that gap and fixes the smaller API issues
found at the same time.

> **Partly superseded by §12.** `MutableHnswIndex` and `buildMutable()`
> (§11.1) were removed in favor of building incrementally on the builder
> itself; the consumer table (§11.6), the deprecated alias (§11.7), the
> example description (§11.9) and the aggregated validation (§11.10) are
> out of date. §11.2–§11.5 and §11.8 still hold, apart from their
> references to `MutableHnswIndex`.

### 11.1 `MutableHnswIndex` and `HnswIndexBuilder.buildMutable()`

What the consumers do with `GraphIndexBuilder` today, none of which
`build()` covered:

| Operation | Cassandra memtable (`CassandraOnHeapGraph`) | Cassandra compaction (`CompactionGraph`) | OpenSearch merge (`JVectorWriter`) |
|---|---|---|---|
| `addGraphNode` concurrently with search, returning bytes used | ✓ | ✓ | ✓ |
| `markNodeDeleted` | ✓ | | ✓ |
| `cleanup()` before writing | ✓ | ✓ | ✓ |
| `GraphIndexBuilder.rescore` with a refined PQ codebook | | ✓ | |
| Build without all vectors in one `RandomAccessVectorValues` | | ✓ | |

`buildMutable()` validates the same configuration as `build()` and returns a
`MutableHnswIndex` (`jvector-base`, `io.github.jbellis.jvector.graph`): a
handle around a `GraphIndexBuilder` whose graph starts empty (or as the
`withExistingGraph` graph) and grows as the caller adds nodes.

- **Operations:** `addNode(ordinal, vector)` (returns bytes used),
  `markDeleted`, `removeDeletedNodes`, `cleanup`, `rescore`, `graph()`
  (a `PersistableGraphIndex`), `searcher()`, `insertsInProgress`,
  `ramBytesUsed`, `close`. It implements `Index`.
- **Validation by mode:** `buildMutable()` needs `withVectorValues` only
  together with `withSimilarityFunction`. With `withScoreProvider`, the
  dimension comes from `withVectorValues` if set, otherwise from
  `withDimension` (now required in that case). `build()` still always
  requires `withVectorValues`.
- **`build()` is layered on it:** `buildMutable()`, insert every vector in
  parallel on the SIMD executor, `cleanup()`, return the graph. The
  existing-graph path uses the same code, inserting from the existing
  graph's `getIdUpperBound()`.

**Locking.** The handle owns a read/write lock so callers no longer need
one. `addNode` and `markDeleted` take the read lock and run concurrently.
`cleanup`, `removeDeletedNodes` and `rescore` take the write lock, because
`GraphIndexBuilder` documents the first two as unsafe during concurrent
modification and `rescore` replaces the builder. Searches take no lock.

**`rescore(Supplier<BuildScoreProvider>)`.** The supplier runs with inserts
locked out. That matters because the old score provider reads state the
caller is about to replace: Cassandra refines the codebook and re-encodes
every vector added so far before rescoring. Running all of that inside the
supplier covers what `CompactionGraph` currently guards with its own
`trainingLock`. `rescore` replaces the graph instance, so callers must not
cache `graph()` or its searchers across it. `rescore(BuildScoreProvider)`
is a convenience for when nothing else needs swapping.

**ForkJoinPool-cooperative waiting.** The exclusive operations run parallel
work on the builder's executors while holding the write lock (and a rescore
supplier may too, e.g. `ProductQuantization.refine`). If inserts ran on
those same pools and their workers simply parked on the read lock, that
work would have no free worker, and the build would deadlock. This was
reproduced with a caller-held lock in the example. Inserts therefore wait
through `ForkJoinPool.managedBlock`, which lets the pool start a spare
worker. Outside a pool it behaves like a plain `lock()`. A regression test
fails with a plain lock and passes with this.

Every attempt to take the read lock without blocking uses the timed
`tryLock(0, NANOSECONDS)`, never the untimed `tryLock()`. The untimed form
takes a free read lock even while a writer is queued, so under a steady
stream of inserts `cleanup()` or `rescore()` waited until the stream ran dry
(measured: a `cleanup()` requested after about 5,000 of 100,000 inserts
waited while the other 95,000 ran). The timed form respects the queued
writer. `cleanupIsNotStarvedByAContinuousStreamOfInserts` covers it.

**Documented precondition on `addNode` (not new).** `GraphIndexBuilder.addGraphNode`
has always required every node in the graph to be scoreable by ordinal
through the build score provider: later inserts score the nodes they visit
by ordinal, and so do pruning and `cleanup()`; the vector passed in is only
used for the new node's own neighbor search. Verified on unchanged `main`:
with a provider that only knows ordinals 0..49, inserting 50 succeeds but
inserting 51 then fails with `IndexOutOfBoundsException` while scoring node
50. `MutableHnswIndex` only wraps `addGraphNode`, so it neither adds nor
removes the requirement; it now documents it, including that the failure
surfaces on a later call. Cassandra already satisfies it (it adds or encodes
each vector before inserting).

### 11.2 `withExistingGraph` takes `OnHeapGraphIndex`

It took the package-private `MutableGraphIndex`, which callers outside the
package couldn't name. `OnHeapGraphIndex` is that interface's only
implementation and is what `OnHeapGraphIndex.load` returns, so the parameter
is now `OnHeapGraphIndex`. There were no external callers yet.

Constraint documented alongside it: the existing graph keeps the
`DiversityProvider` it was created with (the one passed to
`OnHeapGraphIndex.load`, or one derived from the score provider of the
original build). That provider must be able to score the ordinals being
appended, not only the builder's new score provider.

### 11.3 `build()` returns `PersistableGraphIndex`

The writer accessors (`getWriterBuilder`, `getParallelWriterBuilder`) live
on `PersistableGraphIndex`, so writing the result of `build()` needed a
cast. `build()` now returns `PersistableGraphIndex`: still an interface
(design goal 3), and a subtype of `GraphIndex`, so callers assigning the
result to `GraphIndex` still compile. `MutableHnswIndex.graph()` returns the
same type. Returning the concrete `OnHeapGraphIndex` was considered and
rejected. It would have exposed `save()` and fed `withExistingGraph`
without a cast, but hard-wires the implementation class into the public
API; `save()` callers can still narrow.

### 11.4 `IndexSearcher extends Closeable`

`GraphSearcher` holds a `View` that must be closed; for an on-disk graph
that view holds a file reader. With `IndexSearcher` an empty marker, code
holding only `Index` couldn't release a searcher without narrowing to
`GraphSearcher`. `IndexSearcher` now extends `java.io.Closeable` (in
`jvector-api`), and `Index.searcher()` documents that the caller must close
the result. `Closeable` rather than `AutoCloseable`, so generic
try-with-resources only has to handle `IOException`. `GraphSearcher` already
implemented `Closeable`, and `IvfSearcher` has no implementations, so
nothing else changed.

### 11.5 Fixes found along the way

- **Inserting after `cleanup()` could create self-edges.** This predates
  this branch. `cleanup()` sets `allMutationsCompleted`, after which
  `OnHeapGraphIndex.getView()` returns a `FrozenView` that doesn't hide
  incomplete nodes, and nothing cleared the flag. A node inserted after
  `cleanup()` (including into a graph produced by `build()` and then passed
  to `withExistingGraph`) could find its own half-added entry in an upper
  layer and link to itself, which trips an assertion in `ConcurrentNeighborMap`.
  `OnHeapGraphIndex.addNode` now clears the flag before the node becomes
  visible. Graphs from `OnHeapGraphIndex.load` start unfrozen, which is why
  OpenSearch's leading-segment merge never hit this. Views obtained while
  the graph was frozen shouldn't be reused across new insertions.
- **`rescore` dropped pending deletes.** This predates this branch.
  `GraphIndexBuilder.rescore` copies every node into a new graph but did not
  copy the set of nodes marked deleted, so a node deleted before a rescore
  became live again and survived `cleanup()`. Cassandra's compaction never
  deletes before rescoring, so it wasn't affected, but `MutableHnswIndex`
  offers `markDeleted` and `rescore` together. `rescore` now carries the marks
  over; covered by `GraphIndexBuilderTest.testRescoreKeepsPendingDeletes` and
  `MutableHnswIndexTest.rescoreKeepsNodesMarkedDeleted`.
- **Writer Javadoc corrected.** `PersistableGraphIndex.getWriterBuilder(Path)`
  was described as sequential but returns the random-access
  `OnDiskGraphIndexWriter`. The parallel options were described as "ignored
  with a WARN log" but throw `UnsupportedOperationException` on the
  random-access and sequential builders. A worker count of 0 was described
  as disabling parallelism but means "all available processors".
  `GraphIndexWriterTypes.RANDOM_ACCESS` was described as sequential with
  async I/O.

### 11.6 Migrating the consumers

| Consumer call site | Today | With this API |
|---|---|---|
| Cassandra `CassandraOnHeapGraph` | `new GraphIndexBuilder(ravv, vsf, …)`, `addGraphNode`, `markNodeDeleted`, `cleanup` | `withVectorValues(ravv).withSimilarityFunction(vsf)…buildMutable()`, then `addNode`/`markDeleted`/`cleanup` |
| Cassandra `CompactionGraph` | 10-arg constructor with a PQ score provider, `addGraphNode` under `trainingLock`, `GraphIndexBuilder.rescore` | `withScoreProvider(bsp).withDimension(d)…buildMutable()`; codebook refinement moves into `rescore(() -> …)` and `trainingLock` goes away |
| OpenSearch `JVectorWriter.getGraph` | 7-arg constructor, parallel `addGraphNode`, `cleanup` | `withVectorValues(ravv).withScoreProvider(bsp)…build()` |
| OpenSearch leading-segment merge | existing-graph constructor, `addGraphNode`, `markNodeDeleted`, `cleanup` | `withExistingGraph(loaded)…buildMutable()`, then `addNode`/`markDeleted`/`cleanup` |

### 11.7 `ImmutableGraphIndex` renamed to `GraphIndex`, with a deprecated alias

The first pass renamed `ImmutableGraphIndex` to `GraphIndex` (so that it can
extend `Index`), which was not recorded here. Leaving no alias would break
every caller at compile time (Cassandra: 3 files, 11 references) and at run
time for anything compiled against 4.0.x.

`ImmutableGraphIndex` is back as `@Deprecated(forRemoval = true) interface
ImmutableGraphIndex extends GraphIndex`, to be removed in the release after
this one:

- Nested types and constants are inherited, so `ImmutableGraphIndex.View`,
  `.ScoringView`, `.NodeAtLevel`, `.IntMarker`, `.NeighborProcessor` and
  `.ENTRY_NODE_ABSENT` still resolve. Static methods are not inherited, so it
  redeclares `prettyPrint(ImmutableGraphIndex)`.
- `MutableGraphIndex` (and so `OnHeapGraphIndex`) and `OnDiskGraphIndex`
  implement it again, as they did before.
- The four `GraphIndexBuilder` methods that returned it before still do:
  `build(RandomAccessVectorValues)`, `getGraph()`, and both
  `buildAndMergeNewNodes` overloads. That keeps `ImmutableGraphIndex g =
  builder.build(ravv)` compiling, and those four method signatures are
  binary-compatible with 4.0.x again. When the alias is removed, they go back to
  returning `GraphIndex`.
- This is source compatibility: classes compiled against 4.0.x that use the
  nested types still need recompiling, since those types are now members of
  `GraphIndex`.

`ImmutableGraphIndexCompatibilityTest` compiles the old spellings the way
Cassandra and the tutorials use them, from outside JVector's packages. The
tutorials now use `GraphIndex`.

### 11.8 Experimental and closeable stubs

- Public API that only throws until it is implemented is marked
  `@Experimental`: `HnswRecipe`, `IvfRecipe`, `HnswIndexBuilder.applyRecipe`,
  `Indexes.ivfBuilder()`, `IvfIndex`, `IvfSearcher`, and `IvfIndexBuilder`. The
  annotation moved from `jvector-base` to `jvector-api` (same package and
  name) so the recipe enums can use it, and is now `@Documented` so it shows
  in the generated Javadoc.
- `Index.close()` is declared to throw only `IOException`, matching
  `IndexSearcher`, so generic try-with-resources over an `Index` doesn't have
  to catch `Exception`.

### 11.9 Example coverage

`jvector-examples/.../IndexApiExample.java` section 9 shows `MutableHnswIndex`
on its own (inserting while searching, deletes, and writing with each ordinal
mapping), and section 11 runs each old construction path next to its new
counterpart, including the memtable delete pattern (11g) and the compaction
rescore pattern (11h).

### 11.10 Validation reports every problem, not only missing values

`IndexBuilderValidation` gained `check(valid, problem)` for values that are
present but invalid, and for settings that conflict; `throwIfAny` reports them
together with the missing values in one `IllegalStateException` (the message
for missing values alone is unchanged). Before, only missing values were
aggregated (design goal 4): a bad `beamWidth`, `neighborOverflow`, `alpha` or
per-layer degree list reached `GraphIndexBuilder`'s constructor and failed
there, one at a time, as `IllegalArgumentException`.

`HnswIndexBuilder` now checks the same ranges `GraphIndexBuilder` enforces
(plus a positive `withDimension`), and reports as conflicts both the scoring
pair (`withScoreProvider` with `withSimilarityFunction`) and `withMaxDegree(s)`
or `withAddHierarchy` set alongside `withExistingGraph`. Those last two used
to be silently ignored. `IvfIndexBuilder` reports its scoring conflict the
same way.

## 12. Simplification pass: an easier builder

Using the §11 API in a rewritten example showed it was hard to use. Every
build had to set seven or more values that already have well-known
defaults, the two scoring modes were mutually exclusive setters checked only
at runtime, compression meant computing PQ or BQ vectors by hand, and
`MutableHnswIndex` was a second type to learn for the incremental case. This
pass reshaped the builder around the common case.

### 12.1 Scoring chosen up front; defaults for everything else

`Indexes.hnswBuilder()` became two factories:

- `Indexes.hnswBuilder(RandomAccessVectorValues, VectorSimilarityFunction)`
  returns a `RavvHnswBuilder`, which scores with the vectors.
- `Indexes.hnswBuilder(BuildScoreProvider, int dimension)` returns a
  `ScoreProviderHnswBuilder`, which uses the caller's provider.

`HnswIndexBuilder` is now an abstract class with a package-private
constructor, so these are its only subclasses. Each subclass holds its own
scoring inputs as `private final` fields; the base class holds only the
graph shape and tuning settings, as primitives with `GraphIndexBuilder`'s
defaults (max degree 32, beam width 100, overflow 1.2, alpha 1.2, hierarchy
and refinement on). With nothing left that is required, the
collect-every-missing-value validation, `withScoreProvider`,
`withVectorValues`, `withSimilarityFunction`, `withDimension` and
`IndexBuilderValidation` were all removed. Out-of-range values are rejected
by `GraphIndexBuilder` when the graph is built.

The one cross-setting rule kept is the existing-graph conflict:
`withMaxDegree(s)` or `withAddHierarchy` together with `withExistingGraph`
throws `IllegalStateException`, naming every conflict. The builder tracks
whether those were set explicitly, so values applied by a recipe don't count.

### 12.2 Compression is a setting

`withCompressionType(CompressionType)` (NONE, PQ or BQ) on the vector builder
makes the builder train the quantizer, encode the vectors on its own
executors, and build with the compressed scores. The PQ/BQ code is shared
with `GraphIndexBuilder`'s JMX path through a package-private
`GraphIndexBuilder.buildScoreProvider(...)`. The score-provider builder logs a
warning and ignores the setting, since its provider is fixed.

### 12.3 One builder for batch and incremental construction

`MutableHnswIndex` and `buildMutable()` were removed. The builder itself now
covers both cases:

- `buildAndPopulate()` builds and inserts the builder's own vectors in one
  call (the vector builder only; the score-provider builder throws).
- `build()` returns the graph, initially empty, for incremental use with
  `addGraphNode`, `markNodeDeleted`, `removeDeletedNodes`, `cleanup()` and
  `insertsInProgress()`, which delegate to the underlying
  `GraphIndexBuilder`. `build()` is idempotent and thread-safe: the
  underlying builder is created once, and later calls return the same
  graph.
- `populateGraph(ravv)` inserts a whole `RandomAccessVectorValues` and calls
  `cleanup()`. It adds only ordinals from the graph's current id upper
  bound onward, so on an existing graph it appends rather than re-inserting
  the existing nodes.
- `HnswIndexBuilder.rescore(HnswIndexBuilder, BuildScoreProvider)` returns a
  new builder holding a re-scored copy of the graph, carrying over the
  source's settings (max degrees and hierarchy taken from its graph).
- `HnswIndexBuilder` is `Closeable`, closing the underlying builder's
  per-thread scratch space.

This drops §11.1's locking and `ForkJoinPool`-cooperative waiting. Callers
coordinate `cleanup()`, `removeDeletedNodes()` and `rescore` with inserts
themselves, exactly as with `GraphIndexBuilder`; reworking the library's
concurrency was out of scope for an API change.

`build()`, `getGraph()` and `populateGraph()` return `PersistableGraphIndex`,
and so do `GraphIndexBuilder.build(ravv)` and `getGraph()`.
`MutableGraphIndex` went back to being package-private, as on `main`.

### 12.4 Recipes

`HnswRecipe` gained `DEFAULT`, which restates the builder's defaults, and a
small key/value representation: each recipe carries values keyed by the
constants in `HnswRecipe.Param`, which `applyRecipe` reads. `HIGH_RECALL` and
`HIGH_PERFORMANCE` have no values yet and are still refused. Recipe types
and both `applyRecipe` methods are `@Experimental`; the representation is a
prototype (values are loosely typed, and an unknown key is ignored).

### 12.5 `ImmutableGraphIndex` removed

The deprecated alias from §11.7 was removed, along with
`ImmutableGraphIndexCompatibilityTest`. This is a breaking change for 4.0.x
callers that name `ImmutableGraphIndex`; `UPGRADING.md` lists the renames.

### 12.6 Migrating the consumers

| Consumer call site | Today | With this API |
|---|---|---|
| Cassandra `CassandraOnHeapGraph` | `new GraphIndexBuilder(ravv, vsf, …)`, `addGraphNode`, `markNodeDeleted`, `cleanup` | `Indexes.hnswBuilder(ravv, vsf)…build()`, then `addGraphNode`/`markNodeDeleted`/`cleanup` |
| Cassandra `CompactionGraph` | 10-arg constructor with a PQ score provider, `addGraphNode` under `trainingLock`, `GraphIndexBuilder.rescore` | `Indexes.hnswBuilder(bsp, dimension)…build()`, `addGraphNode`, `HnswIndexBuilder.rescore(builder, refined)`; `trainingLock` stays |
| OpenSearch `JVectorWriter.getGraph` | 7-arg constructor, parallel `addGraphNode`, `cleanup` | `Indexes.hnswBuilder(bsp, dimension)…populateGraph(ravv)`, or `Indexes.hnswBuilder(ravv, vsf).withCompressionType(PQ).buildAndPopulate()` if the builder can own the quantization |
| OpenSearch leading-segment merge | existing-graph constructor, `addGraphNode`, `markNodeDeleted`, `cleanup` | `…withExistingGraph(loaded)`, then `populateGraph(superset)` or `addGraphNode`, `markNodeDeleted`, `cleanup` |

### 12.7 Example

`IndexApiExample` was rewritten around ease of use: a one-call quickstart,
tuning and compression by setting, writing the populated graph with the
sequential and parallel writers (inline and NVQ vectors), incremental
construction with deletes, a score-provider build with a rescore, continuing
a saved graph, search options, and the remaining validation. The
legacy-comparison section was dropped, as were the fused-PQ and
separate-PQ layouts, which need a caller-computed codebook (§10).

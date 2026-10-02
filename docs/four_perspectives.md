# One Configuration Interface, Four Perspectives

> **About this document.** This guide comes from the aci project and is kept here as the
> source of the approach in [`index_hierarchy_plan.md`](index_hierarchy_plan.md), which
> records what JVector adopted from it and what it deliberately left out (§1). The SRDs and
> test classes it names (SRD 02 and 03, `BuilderPerspectiveTest`, and so on) are part of the
> aci project, not this repository.

*How the aci demonstrator is used by the people who build the index system,
the people who program against it, the people who embed it, and the people
who only ever see it through something else.*

aci — the Algebraic Configuration Interface — is a demonstrator of one idea:
that a specialized index system with several backing implementations can be
configured through **one interface** that serves very different people
equally well. This guide walks the system as built to
SRD 03, the minimum scenario, through four
perspectives. The code shown is the real API; where a perspective's outer
surface is not part of this project (the database syntax in §4), that is
said plainly.

---

## The system in one picture

```
              ┌──────────────────────────────────────────────────────────┐
  4. outside  │  an embedding system — e.g. another database             │
     user ──▶ │  CREATE INDEX … USING hnsw WITH OPTIONS {…}   (its syntax) │
              └───────────────┬──────────────────────────────────────────┘
                              │ JSON / recipe name
              ┌───────────────▼──────────────────────────────────────────┐
  3. system   │  aci-index  — the configuration interface                │
  integrator  │  IndexConfig.fromJson · toJson · Readers.forIndex        │
              │  IndexConfig.hnsw()/ivf() · recipe() · assessment        │
  2. API user │  Rules: verboten V1–V2 · suboptimal S1–S3 · Recipes      │
              └───────────────┬──────────────────────────────────────────┘
                              │ IndexSpec  (the seam)
              ┌───────────────▼──────────────────────────────────────────┐
  1. vector   │  aci-subject — the index system's contract               │
  system      │  Catalog · Index · Query · ParameterCatalog              │
  builder     │  SpecValidation · QuerySchema · journal                  │
              │  (aci-subject-mock: five backings, deterministic)        │
              └──────────────────────────────────────────────────────────┘
```

Two layers, one seam. The **subject** (bottom) is the vector index system
itself: what the vector system builder produces. Above it, **aci-index** is
the configuration interface everyone else uses. The seam between them is a
single value, `IndexSpec` — the complete description of an index — and the
one thing every perspective shares is the **verdict** on a configuration:
*prescriptive*, *acceptable*, *suboptimal*, or *verboten*.

The scenario uses two of the subject's index types, `hnsw` and `ivf`. They
share `dimensions`, `metric`, and `capacity`; `hnsw` alone has `m` and
`efConstruction`; `ivf` alone has `nlist`. On the query side, `hnsw` has
`efSearch`, `ivf` has `nprobe`.

---

## 1. The vector system builder

*Owns the index system. Writes the knowledge once; everyone downstream
inherits it.* — living example:
`BuilderPerspectiveTest`

### Declaring the parameter space, once

The builder does not write validation code per parameter. They **declare**
each parameter in the subject's catalog — its type, default, constraints,
and *where it applies* — and validation, schema derivation, and the
descriptor all read that one declaration:

```java
// aci-subject: ParameterCatalog — the actual declarations behind the scenario
spec("capacity",       CONSTRUCT, MUTABLE,   INT,  10000, [Range 1..MAX, AtLeastCurrentSize], universal())
required("dimensions", CONSTRUCT, IMMUTABLE, INT,         [Range 1..4096],                   family(VECTOR))
spec("metric",         CONSTRUCT, IMMUTABLE, ENUM, cosine,[OneOf cosine|euclidean|dot],       family(VECTOR))
spec("m",              CONSTRUCT, IMMUTABLE, INT,  16,    [Range 2..100],                    backing(HNSW))
spec("efConstruction", CONSTRUCT, IMMUTABLE, INT,  200,   [Range 1..1000, AtLeastParameter m],backing(HNSW))
spec("nlist",          CONSTRUCT, IMMUTABLE, INT,  100,   [Range 1..65536],                  backing(IVF))

spec("efSearch",       QUERY, INT, 64, [Range 1..1000, AtLeastParameter limit], backing(HNSW))
spec("nprobe",         QUERY, INT, 8,  [Range 1..MAX,  AtMostParameter nlist],  backing(IVF))
```

Three things fall out of that declaration with no further work:

- **Total, all-at-once validation.** `SpecValidation.validateCreate(spec)`
  reports every problem in one exception, and each violation says where the
  parameter *would* have applied: *"`nlist` does not apply to backing hnsw
  (applies to backing ivf)"*.
- **A derived query schema per index.** `index.querySchema()` is exactly the
  universal options ∪ the family's ∪ the backing's — so `efSearch` appears
  for an `hnsw` index and `nprobe` for an `ivf` one, without a line of
  per-type code.
- **A generic handle that hides the backing.** Every operation — `upsert`,
  `query()`, `tune`, `descriptor()` — is on `Index`; no caller ever needs a
  backing-specific type (SRD 02 axiom S4).

The coupled constraints are the interesting ones: `efConstruction ≥ m` ties
two construct parameters; `nprobe ≤ nlist` ties a *query option* to the
*index's* configuration. Both are declared, not coded, and both surface with
both values named.

### Encoding judgement: rules and recipes

Validity is the subject's business. **Advice** — what is verboten beyond the
structural rules, what is suboptimal, what is the recommended shape — is
knowledge the builder has and users don't. In aci-index the builder writes
it as rules and recipes:

```java
// aci-index: Rules — the scenario's judgement, in one place
if (ef < m)      verboten("V1", "efConstruction=" + ef + " < m=" + m);
if (nlist > cap) verboten("V2", "nlist=" + nlist + " > capacity=" + cap);
if (m > 64)      suboptimal("S1", "m=" + m + " > 64",                 "16–64");
if (ef < 2 * m)  suboptimal("S2", "efConstruction=" + ef + " < 2·m=" + 2*m, "≥ " + 2*m);
// … and the subject's own validation as the floor: whatever it rejects is verboten here.

// aci-index: Recipe — the configurations the builder actually recommends
HNSW_HIGH_RECALL("hnsw.highRecall", HNSW) { fixed: metric=cosine, m=32, efConstruction=400; free: dimensions, capacity }
IVF_STANDARD    ("ivf.standard",    IVF)  { fixed: metric=cosine, nlist=⌈√capacity⌉;        free: dimensions, capacity }
```

A recipe fixes most parameters and names the few that remain free. That is
the builder's way of saying *"start here"* — and because a hand-built
configuration that exactly matches a recipe is recognized as prescriptive
too, users who don't know the recipe exists still get told when they've
landed on one.

### What the builder gets

The fluent layer can never hand the subject a spec it would reject: the
subject's own validation is the verboten floor (R-M8), and the tests prove
every recipe and every acceptable example creates in `MockCatalog`. The
builder writes the parameter declarations, the rules, and the recipes once;
they never write a configuration parser, a per-type validator, or a
"which options are valid here" table.

---

## 2. A user of the vector APIs

*Programs against the index system in Java. Wants the compiler on their side
and never wants to guess.* — living example:
`ApiUserPerspectiveTest`

### Configuring: the type first, then only what fits

```java
IndexConfig hnsw = IndexConfig.hnsw()
    .dimensions(768)
    .metric(Metric.COSINE)
    .m(32)
    .efConstruction(400)
    .build();

IndexConfig ivf = IndexConfig.ivf()
    .dimensions(768)
    .nlist(1024)
    .build();
```

The index type is chosen first, and it selects the vocabulary: `.nlist()` is
not a method of the `hnsw` builder, `.m()` is not a method of the `ivf`
builder. A disjoint parameter on the wrong type is a **compile error**, not
a runtime refusal. The common parameters — `dimensions`, `metric`,
`capacity` — are the shared stage both builders inherit.

### The easy path: recipes

```java
IndexConfig c = IndexConfig.recipe(Recipe.HNSW_HIGH_RECALL).dimensions(768).build();
c.describe();
// hnsw{dimensions=768, metric=cosine, capacity=10000, m=32, efConstruction=400} — PRESCRIPTIVE (hnsw.highRecall)
```

A recipe's builder exposes **only** its free parameters. There is nothing
else to set, so there is nothing to get wrong.

### The verdict, always

Every descriptor carries its assessment, and `describe()` says it in one
line:

```
hnsw{…, m=32, efConstruction=400} — PRESCRIPTIVE (hnsw.highRecall)
ivf{…, capacity=10000, nlist=120} — ACCEPTABLE
hnsw{…, m=96, efConstruction=150} — SUBOPTIMAL: S1 m=96 > 64 (suggest 16–64); S2 efConstruction=150 < 2·m=192 (suggest ≥ 192)
```

Verboten never becomes a descriptor. `build()` throws, with every finding at
once:

```java
IndexConfig.ivf().dimensions(768).capacity(1000).nlist(5000).build();
// VerbotenConfiguration: ivf{dimensions=768, metric=cosine, capacity=1000, nlist=5000} — VERBOTEN: V2 nlist=5000 > capacity=1000
```

And for a UI that wants the verdict as the user types, `builder.assess()`
returns it without building — empty means "this would be refused".

### Using the index — and the reader

```java
Catalog catalog = …;                                   // the subject's catalog
Index index = catalog.create(c.toIndexSpec("products"));
index.upsert(Entry.vector("doc-1", embedding));

Reader<?> reader = Readers.forIndex(index);            // matches the concrete type — the caller holds only Index
Results r = reader.vector(q).limit(10).search();       // the common machinery
```

The user holds `Index`, the contract type; the backing is hidden. When they
ask for a reader they get the one whose machinery matches what the index
*is* — an `HnswReader` with `efSearch`, or an `IvfReader` with `nprobe` —
decided from the index's descriptor, never by the caller. When they want that
concrete machinery, the `Reader` hierarchy is sealed, so a `switch` is
exhaustive and cast-free:

```java
Results tuned = switch (reader) {
    case HnswReader h -> h.vector(q).limit(10).efSearch(128).search();
    case IvfReader  i -> i.vector(q).limit(10).nprobe(16).search();
};
```

`search()` executes through the subject's own `index.query()`. There is no
second path: an invalid option is refused by the subject with the subject's
message, and the type-specific option shows up in the index's journal.

### What the API user never has to know

Which backing an index is (the reader tells them); which parameters exist
for a type (the builder only offers the right ones); whether a configuration
is good (the verdict says); what the subject would reject (it's verboten
before they get there).

---

## 3. A system integrator

*Embeds the index system into a larger host. Configuration arrives as data,
indexes come and go, and operations people need to see what's what.* — living
example:
`IntegratorPerspectiveTest`

### Configuration as data: the loader

The integrator's configurations come from files, a config service, an
operator's hands — not from Java code. The descriptor's JSON form is the
contract:

```json
{"type": "hnsw", "dimensions": 768, "metric": "cosine", "capacity": 10000, "m": 32, "efConstruction": 400}
```

```java
IndexConfig c = IndexConfig.fromJson(json);   // exactly the builder's checks, exactly its verdict
```

The loader is the same machine as the builder, so the integrator gets the
same guarantees the API user got from the compiler — one step later, with
messages written for someone looking at a file rather than at code:

```
not an index descriptor: "nlist" is not a parameter of hnsw (it belongs to ivf); valid: [dimensions, metric, capacity, m, efConstruction]
```

Every structural problem is reported at once; then the verboten rules apply
exactly as at build. Omitted parameters take defaults; `toJson()` writes
every parameter **with defaults filled in**, so a descriptor saved today
still means the same thing after the builder changes a default next year.

### Many indexes, type-agnostic code

```java
for (String name : catalog.names()) {
    Index index = catalog.find(name);                  // by name; that's all the integrator knows
    Reader<?> reader = Readers.forIndex(index);        // the right machinery, every time
    …
}
```

The integrator's code names no backings. If the vector system builder adds a
third type, it appears as a new `IndexType`, a new builder, and a new sealed
`Reader` subtype — and the compiler points at every `switch` in the
integrator's code that now needs a case. That is the sealed hierarchy doing
the integrator's regression testing for them.

### Surfacing verdicts where operators look

Because the verdict is a value on the descriptor, the integrator decides
where it shows:

- at **load**: refuse verboten (the exception already carries every finding);
  log suboptimal findings *with their suggestions* so the operator knows what
  to change; note the recipe name for prescriptive ones;
- in **status / describe** endpoints: `describe()` is the one line to print;
- in **audit**: the subject's `journal()` holds the effective configuration
  at creation, every tune, and every query with its effective options —
  the record of what actually ran, not what was intended.

```java
switch (c.assessment()) {
    case Assessment.Prescriptive p -> log.info("{} uses recipe {}", name, p.recipe().key());
    case Assessment.Suboptimal s   -> s.findings().forEach(f -> log.warn("{}: {}", name, f.render()));
    case Assessment.Acceptable a   -> { }
}
```

### What the integrator never has to know

Anything about a specific backing; how to validate a configuration; what a
"good" configuration is. They carry JSON in, hand `Index` handles out, and
relay verdicts.

---

## 4. An outside user, through an embedding system

*Never sees Java, aci, or the subject. Talks to another system — say, a
database that embeds the index system — in that system's own language.* —
living example:
`OutsideUserPerspectiveTest`
over the stand-in
`EmbeddingDatabase`

This perspective is the reason the previous three are shaped the way they
are. The database is a **system integrator** (§3); its end user is someone
who writes statements. Nothing in this section's syntax is part of aci — it
is what an embedding database *could* say, shown to make the flow concrete.

### Creating an index

```sql
-- illustrative: the embedding database's own syntax
CREATE VECTOR INDEX products ON docs(embedding)
  USING hnsw WITH OPTIONS {"dimensions": 768, "m": 32, "efConstruction": 400};

-- or, the easy path the builder recommended:
CREATE VECTOR INDEX products ON docs(embedding)
  USING RECIPE 'hnsw.highRecall' WITH OPTIONS {"dimensions": 768};
```

Under the statement, the database does what §3 does: the `OPTIONS` object
plus the `USING` type *is* the descriptor's JSON, so it calls
`IndexConfig.fromJson(…)` (or `IndexConfig.recipe(…)` for the recipe form),
then `catalog.create(c.toIndexSpec("products"))`. What comes back to the
outside user is the **verdict, translated into the database's idiom**:

| The descriptor was… | The outside user sees… |
|---|---|
| verboten | the statement fails: `ERROR: VERBOTEN: V2 nlist=5000 > capacity=1000` — every finding, none hidden |
| suboptimal | the statement succeeds with a notice: `WARNING: S1 m=96 > 64 (suggest 16–64)` |
| prescriptive | success, and `DESCRIBE INDEX products` reports `recipe: hnsw.highRecall` |
| structurally wrong | `ERROR: "nlist" is not a parameter of hnsw (it belongs to ivf); valid: [dimensions, metric, capacity, m, efConstruction]` |

The outside user gets exactly the three-way clarity the scenario requires —
without the database having written a single rule of its own.

### Querying

```sql
-- illustrative
SELECT id FROM docs ORDER BY embedding <-> $q LIMIT 10 WITH SEARCH OPTIONS {"efSearch": 128};
```

The database finds the index by name, calls `Readers.forIndex`, and switches
on the sealed reader to apply `efSearch` — or, if the index turns out to be
`ivf`, it can pass the option straight to the subject's query and relay the
subject's own message: *"`efSearch` does not apply to backing ivf (applies
to backing hnsw or backing hybrid)"*. The user never named a backing; the
system told them what fits.

### What the outside user never has to know

That there is a Java API; that there are two index types with different
vocabularies; which one they have. They learn a configuration is verboten,
suboptimal, or prescriptive in their own tool's words — because the verdict
was computed once, at the layer that knows, and carried out unchanged.

---

## The through-line

| Perspective | Writes | Reads | Never touches |
|---|---|---|---|
| **1. vector system builder** | parameter declarations, rules, recipes | — | configuration parsers, per-type validators |
| **2. API user** | fluent builders, readers | verdicts, results | backing-specific types, option tables |
| **3. system integrator** | JSON in, handles out | verdicts, journals | backings, validation, "good" configurations |
| **4. outside user** | statements in another system's language | the same verdicts, in that language | any of the above |

One declaration of the parameter space (the builder's), one place for
judgement (rules and recipes), one seam (`IndexSpec`), one verdict — and
each perspective meets it through the surface that fits them: the compiler,
a JSON file, or a statement in some other system. That is what "an
effective configuration interface for both programmatic users and system
embedders" means in practice, and the tests in `aci-index` (named by the
requirement each verifies, `rM1_…` through `rM9_…`) are where each claim
above is pinned.

*Sources of truth: SRD 02 Subject System,
SRD 03 The Minimum Scenario. Every code
block above is a compiled, tested example under
`aci-index/src/test/java/io/nosqlbench/aci/index/guide/`; run them with
`mvn -pl aci-index test -Dtest='*PerspectiveTest'`. This guide is also rendered
as [`four_perspectives.html`](four_perspectives.html) and
[`four_perspectives.pdf`](four_perspectives.pdf).*


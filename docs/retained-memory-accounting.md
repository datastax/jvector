# Retained memory accounting

`ramBytesUsed()` reports the memory an object retains: its Java heap, plus native memory
for ASH. Callers such as a database's memory limits can read it often: reads allocate
nothing, and their cost does not grow with the number of vectors (a PQ or NVQ model adds
a loop over its subspaces).

## What is counted

- **On-disk graph index:** graph metadata, cached upper layers and fused sources, and each
  feature, including any model read from the index header (fused PQ codebooks, NVQ
  parameters). A model is counted by the feature that owns it.
- **PQ and NVQ models:** the whole model, including caches created later on first use
  (for example the PQ cosine table).
- **PQ vectors:** their code storage plus the model, and, for immutable vectors, each
  scoring table once it exists.
- **ASH:** models can hold native memory (their native projections). `ramBytesUsed()`
  includes it and `nativeBytesUsed()` reports that part, for the index, the `FusedASH`
  feature and the model. See [Native ASH](native-ash-preparation.md#retained-memory-accounting).

Not counted: per-thread scratch, the caller's reader supplier, JVM-global objects, and
memory-mapped file pages or the OS page cache, which are not Java heap.

Totals can grow after loading, as caches are created on first use. A caller that adds a
total when it opens an index and subtracts it at close must subtract the value it added,
not a fresh read.

## Accuracy

Totals match the JVM's measured sizes exactly, except fixed per-index metadata, which is
estimated within 256 bytes and does not grow with the index, and the Java handles of an
ASH model's native segments, estimated within 512 bytes per model. Immutable PQ chunks that
share backing storage are each charged for it, so those totals can err high.

Sizes follow the running JVM's object layout (compressed or wide references and class
pointers, compact headers, object alignment). No `sun.misc.Unsafe`, Java agent or
`--add-opens` flag is needed.

## Validation

With `JAVA_HOME` pointing to JDK 22 or newer (and the native build set up as in the
installation instructions), run:

```bash
mvn -Pjdk21 -pl jvector-tests -am test \
  -Dtest=TestRetainedMemoryAccounting,TestFusedPQMemoryAccounting,TestMemoryAccountingCost \
  -Dsurefire.failIfNoSpecifiedTests=false \
  -Djvector.vectorization_provider=native
```

Use `-Pjdk20` with provider `panama`, or `-Pjdk11` with provider `default`, for the other
providers. The tests compare totals with sizes measured in forked JVMs across every object
layout the JVM supports, and check that reads allocate nothing.

# Retained memory accounting

`OnDiskGraphIndex.ramBytesUsed()` reports the immutable index's shared heap: a total
cached at load for graph metadata, containers at their allocated capacity, upper-layer
adjacency arrays and cached fused sources with their backing storage, plus each
feature's live `Feature.ramBytesUsed()`. A model read from the index header (fused PQ
codebooks, NVQ parameters) is held only by the index, so its feature counts it,
including caches the model creates later.

`ProductQuantization` and `NVQuantization` report their whole retained model: for PQ,
codebooks, global centroid, subspace layout, centroid norms and the cosine magnitude
table once created. Float vectors include their array and, on the native provider, the
segment view created on first native access. These totals are read live and allocate
nothing.

The reader supplier belongs to the caller. Per-worker scratch is excluded. JVM-global
format, enum, class, and cached-integer objects are excluded. Mapped file pages and the operating system page cache
are not Java heap and are not included.

## Initialization and updates

Resident source buffers and adjacency arrays are fresh allocations made by the index
loader. Their sizes are accumulated while loading, using actual array lengths and map
capacities, so this part is exact. Fixed per-index metadata (the index object, the
header's feature map and layer list, the entry node) is added up explicitly. The JDK
collections that hold it are sized from how the header loaders build them, without
inspecting JDK internals, so this part is an estimate; it does not grow with the index.
No object graph is traversed.

Retained collection views used by the exposed caches are prepared and counted during
loading. Feature metadata maps and sets are read-only. The internal primitive source
cache returned to scoring code is also read-only by contract: callers must not modify
it. There is no supported insertion or deletion of nodes in an already loaded on-disk
index. Subsequent memory reads are a field read, with no traversal or allocation.

Mutable PQ storage updates its cached total under the existing chunk-allocation lock.
The total changes when a chunk or a larger chunk-reference array is allocated. Writing,
zeroing, or logically deleting a vector inside existing capacity does not change the
retained allocation. Spare chunk-reference capacity is included. Readers use one volatile
field read; they do not acquire that lock or scan chunks.

Immutable PQ storage owns a copy of its chunk-reference array. Its total is computed
once: the chunk array plus each chunk's `ramBytesUsed()`. Chunks that share backing
storage are each charged for it, so such a total errs high. PQ vector totals add the
model and, for immutable vectors, each similarity function's scoring table once created.
The tables are held one slot per similarity function, so they are counted without
allocating.

Byte sequences report the storage retained by that sequence when considered alone.
An array slice retains its backing sequence and full array. A heap-segment slice retains
its segment, keep-alive scope, and full backing array. Native byte sequences reuse their
existing integer field to remember backing-array length, and cache JVM layout constants
once; reporting creates no `Optional` objects or new per-vector accounting fields.

Object layout is determined from the running JVM's flags. Compressed object references,
compressed class pointers, compact headers, and object alignment are handled
independently, as are array element offsets, which depend on element width on JDK 23
and later and with compact headers. Field sizes are summed per class; before JDK 15,
HotSpot's per-class field placement and its padding are modeled. Layout support
targets HotSpot-compatible JVMs. Production uses no `sun.misc.Unsafe`, Java agent, or
`--add-opens` flags.

## Validation

With `JAVA_HOME` pointing to JDK 22 or JDK 25, run:

```bash
mvn -Pjdk21 -pl jvector-tests -am test \
  -Dtest=TestRetainedMemoryAccounting,TestFusedPQMemoryAccounting,TestMemoryAccountingCost \
  -Dsurefire.failIfNoSpecifiedTests=false \
  -Djvector.vectorization_provider=native
```

`jdk21` is the repository's native test-profile name; use JDK 22 or newer to build the
native module. Initialize Highway and the native build as described in the installation
instructions. Use `-Pjdk20` with provider `panama`, or `-Pjdk11` with provider `default`,
to run the corresponding provider tests.

The footprint test builds a temporary startup agent, forks the current JDK, and sums
the JVM's instrumentation sizes with a separate reflective traversal. Buffers, fused
sources, PQ models, PQ storage and its lazily created tables must match exactly. Graphs must match within 256 bytes, the bound
for estimated metadata; the largest metadata error per layout is printed. It checks ordinary
and wide references and class pointers at both 8-byte and 16-byte alignment. JDK 25 adds
compact headers with ordinary and wide references at both alignments: eight layouts on
JDK 22 and twelve on JDK 25. Earlier HotSpot releases test only combinations their VM
can enable; JDK 11 verifies six layouts. Requested settings are checked explicitly.

Fixtures include code sizes through 768 bytes and alignment boundaries, empty and
nonempty graphs, formats 2 through 6, hierarchy depths through 32, multiple node counts
and degrees, and all supported fused-PQ reranking layouts. PQ chunk-boundary growth,
spare reference capacity, reuse/zeroing, invalid ordinals, concurrent growth, and
immutable sliced buffers are compared with independent JVM measurements.

The cost regression asserts that repeated built-in graph, PQ, byte-sequence, and slice
memory reads allocate zero bytes after warmup in at least one of five samples; a per-read
allocation would appear in every sample, while a one-time JVM event (seen on JDK 11) lands
in one. It prints diagnostic timing and graph-load
allocation measurements without timing thresholds. Its optional `main` accepts an
iteration count for longer exercises. Timings include loop, dispatch, and volatile receiver overhead; they
are not graph-query throughput benchmarks. Temporary agents, logs, and index files are
removed by the tests.

Long diagnostic exercises warm the measured call site for at least 250 ms per receiver
type before sampling. Cold graph-loading time and allocated bytes are reported
separately; they are not hidden in the reporting warmup.

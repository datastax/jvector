# Retained memory accounting

`OnDiskGraphIndex.ramBytesUsed()` returns a cached total for the immutable index's
shared heap. It includes graph metadata, containers at their allocated capacity,
upper-layer adjacency arrays, and cached fused sources with their backing storage.

The reader supplier belongs to the caller. Compressor models and per-worker scratch
are excluded from the shared-graph total. JVM-global format, enum, class, cached-integer, and collection
sentinel objects are excluded. Mapped file pages and the operating system page cache
are not Java heap and are not included.

## Initialization and updates

Resident source buffers and adjacency arrays are fresh allocations made by the index
loader. Their sizes are accumulated while loading, using actual array lengths and map
capacities. The small graph/header metadata (the layer list, the feature map and its
views, each feature's own objects, and the inline-offset map) is summed explicitly from
known containers; nothing is traversed reflectively, and no resident source buffer or
adjacency array is walked.

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

Immutable PQ storage owns a copy of its chunk-reference array. Its storage contribution
is summed once from each allocated chunk's own report; chunks that share backing storage
are counted per chunk, and the built-in loaders always allocate distinct chunks. PQ
vector totals include the compressor's
existing reported estimate; derived scoring-table workspaces remain outside this
estimate, as before.

Byte sequences report the storage retained by that sequence when considered alone.
An array slice retains its backing sequence and full array. A heap-segment slice retains
its segment, keep-alive scope, and full backing array. Native byte sequences reuse their
existing integer field to remember backing-array length, and cache JVM layout constants
once; reporting creates no `Optional` objects or new per-vector accounting fields.

Object layout constants come from the running JVM's HotSpot diagnostic options.
Compressed object references, compressed class pointers, compact headers, and object
alignment are handled independently. Array headers follow HotSpot's rule for the running
release: rounded up to a heap word before JDK 23, and only to the element width from JDK 23
on. A shallow instance size is the header plus the
declared field widths, aligned, which matches HotSpot's field packing from JDK 15 on;
earlier releases can pad superclass field boundaries by a heap word. No `sun.misc.Unsafe`
method, Java agent, or `--add-opens` flag is used, so accounting keeps working when the
terminally deprecated Unsafe memory-access methods are denied or removed (JEP 471).

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
the JVM's instrumentation sizes with a separate reflective traversal. It checks ordinary
and wide references and class pointers at both 8-byte and 16-byte alignment. JDK 25 adds
compact headers with ordinary and wide references at both alignments: eight layouts on
JDK 22 and twelve on JDK 25. Earlier HotSpot releases test only combinations their VM
can enable; JDK 11 verifies six layouts. Requested settings are checked explicitly.

Fixtures include code sizes through 768 bytes and alignment boundaries, empty and
nonempty graphs, formats 2 through 6, hierarchy depths through 32, multiple node counts
and degrees, and all supported fused-PQ reranking layouts. PQ chunk-boundary growth,
spare reference capacity, reuse/zeroing, invalid ordinals, concurrent growth, and
immutable sliced buffers are compared with independent JVM measurements. The comparison
is exact on JDK 15 and later; on earlier releases the report must not exceed the
measurement and may fall short by at most 8 bytes per counted object.

The cost regression asserts that repeated built-in graph, PQ, byte-sequence, and slice
memory reads allocate zero bytes after warmup. It prints diagnostic timing and graph-load
allocation measurements without timing thresholds. Its optional `main` accepts an
iteration count for longer exercises. Timings include loop, dispatch, and volatile receiver overhead; they
are not graph-query throughput benchmarks. Temporary agents, logs, and index files are
removed by the tests.

Long diagnostic exercises warm the measured call site for at least 250 ms per receiver
type before sampling. Cold graph-loading time and allocated bytes are reported
separately; they are not hidden in the reporting warmup.

# Retained memory accounting

`OnDiskGraphIndex.ramBytesUsed()` reports the shared graph heap. It includes the graph
object, graph metadata, collection storage at its allocated capacity, upper-layer
adjacency arrays, and cached fused sources with their backing storage. It counts each
object identity once, including storage shared by multiple sources or slices.

The reader supplier belongs to the caller. Compressor models and per-worker scratch
are accounted separately, as with compressed-vector accounting. JVM-global format,
enum, class, cached-integer, and collection-sentinel objects are excluded. Mapped file pages and the operating
system page cache are not Java heap and are not included.

The estimate follows the allocated objects rather than graph degree, code size, active
map entries, or an assumed map load factor. Removing entries does not free a map's backing
arrays; compacting or growing the map changes the reported capacity. Collection views
are included only when allocated. Accounting itself does not create retained graph views.

Byte sequences report the storage retained by that sequence when considered alone.
An array slice retains its backing sequence and full array. A heap-segment slice retains
its segment, keep-alive scope, and full backing array. When summing multiple sequences,
use identity-based accounting to avoid charging their shared objects repeatedly.

Object layout is determined from the running JVM. Compressed object references,
compressed class pointers, compact headers, field offsets, and object alignment are
handled independently. As with the existing sizing utilities, layout support targets
HotSpot-compatible JVMs. Inspecting encapsulated collection fields uses the same
`sun.misc.Unsafe` facility already used by JVector; production does not need a Java agent
or `--add-opens` flags.

Graph accounting traverses resident shared objects and uses temporary identity-tracking
storage proportional to that object count. It is a memory-reporting operation and is
not called by the scoring kernels. Callers must not externally mutate graph caches
while a measurement is in progress. Individual byte-sequence and fused-source estimates
remain constant-time operations.

## Validation

With `JAVA_HOME` pointing to JDK 22 or JDK 25, run:

```bash
mvn -Pjdk21 -pl jvector-tests -am test \
  -Dtest=TestRetainedMemoryAccounting,TestFusedPQMemoryAccounting \
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
JDK 22 and twelve on JDK 25. It asserts that the requested settings are
actually enabled. No production accounting formulas or `ramBytesUsed()` methods are
used to derive expected footprints.

Older HotSpot releases test only the reference/class-pointer combinations their VM
can enable; JDK 11 verifies six layouts.

Fixtures include code sizes through 768 bytes and alignment boundaries, empty and
nonempty graphs, formats 2 through 6, hierarchy depths through 32, multiple node counts
and degrees, and all supported fused-PQ reranking layouts. Cache views, clearing,
resizing, compaction, duplicate sources, shared slices, and cyclic references are
checked. The tests remove their temporary agent jars, logs, and index files.

# Quick start

## Add the dependency

JVector is published to Maven Central as `io.github.jbellis:jvector`:

```xml
<dependency>
  <groupId>io.github.jbellis</groupId>
  <artifactId>jvector</artifactId>
  <version>4.1.0</version>
</dependency>
```

With Gradle: `implementation("io.github.jbellis:jvector:4.1.0")`. The library runs on Java 11 and
newer; ASH requires 4.1.0 or later.

## Enable SIMD

Run your application with these JVM options so JVector can use the Vector API and its native
kernels:

```text
--add-modules jdk.incubator.vector
--enable-native-access=ALL-UNNAMED
```

On Java 22 and later JVector prefers its native library (bundled for Linux x86-64 and AArch64),
on Java 20 and 21 it uses the Panama Vector API, and older runtimes use plain Java.
[Building and installing](building-and-installing.md) explains each backend and how to check
which one is in use.

## Build and search an index

This builds an in-memory graph over a set of vectors and finds the ten nearest neighbors of a
query:

```java
RandomAccessVectorValues vectors = ...;           // your vectors
VectorSimilarityFunction similarity = VectorSimilarityFunction.DOT_PRODUCT;
VectorFloat<?> query = ...;

try (var builder = GraphIndexBuilder.builder(vectors, similarity, 32)   // max degree
             .withBeamWidth(100)
             .withNeighborOverflow(1.2f)
             .withAlpha(1.2f)
             .build();
     var graph = builder.build(vectors)) {
    SearchResult result = GraphSearcher.search(query, 10, vectors, similarity, graph, Bits.ALL);
    for (var match : result.getNodes()) {
        System.out.printf("vector %d  score %.6f%n", match.node, match.score);
    }
}
```

The complete program is
[HelloVectorWorld](../jvector-examples/src/main/java/io/github/jbellis/jvector/example/HelloVectorWorld.java).
It downloads a small public dataset on first use; the
[examples README](../jvector-examples/README.md) shows how to build and run it and the other
hello-world programs, including HelloASHWorld, which writes a compressed on-disk index and
searches it.

## Next steps

1. [Tutorial 1](tutorials/1-intro-tutorial.md): the building blocks (vectors, graphs, searchers).
2. [Tutorial 2](tutorials/2-disk-tutorial.md): writing an index to disk and searching it.
3. [Tutorial 3](tutorials/3-larger-than-memory-tutorial.md): compressed vectors and reranking for
   indexes larger than memory.
4. [Choosing an index](choosing-an-index.md) and [Tuning](tuning.md) for production settings.

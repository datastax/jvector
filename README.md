## Introduction to approximate nearest neighbor search

Exact nearest neighbor search (k-nearest-neighbor or KNN) is prohibitively expensive at higher dimensions, because approaches to segment the search space that work in 2D or 3D like quadtree or k-d tree devolve to linear scans at higher dimensions.  This is one aspect of what is called “[the curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).”

With larger datasets, it is almost always more useful to get an approximate answer in logarithmic time, than the exact answer in linear time.  This is abbreviated as ANN (approximate nearest neighbor) search.

There are two broad categories of ANN index:
* Partition-based indexes, like [LSH or IVF](https://www.datastax.com/guides/what-is-a-vector-index) or [SCANN](https://github.com/google-research/google-research/tree/master/scann)
* Graph indexes, like [HNSW](https://arxiv.org/abs/1603.09320) or [DiskANN](https://www.microsoft.com/en-us/research/project/project-akupara-approximate-nearest-neighbor-search-for-large-scale-semantic-search/)

Graph-based indexes tend to be simpler to implement and faster, but more importantly they can be constructed and updated incrementally.  This makes them a much better fit for a general-purpose index than partitioning approaches that only work on static datasets that are completely specified up front.  That is why all the major commercial vector indexes use graph approaches.

JVector is a graph index that merges the DiskANN and HNSW family trees.
JVector borrows the hierarchical structure from HNSW, and uses Vamana (the algorithm behind DiskANN) within each layer.


## JVector Architecture

JVector is a graph-based index that builds on the HNSW and DiskANN designs with composable extensions.

JVector implements a multi-layer graph with nonblocking concurrency control, allowing construction to scale linearly with the number of cores:
![JVector scales linearly as thread count increases](https://github.com/jbellis/jvector/assets/42158/f0127bfc-6c45-48b9-96ea-95b2120da0d9)

The upper layers of the hierarchy are represented by an in-memory adjacency list per node. This allows for quick navigation with no IOs.
The bottom layer of the graph is represented by an on-disk adjacency list per node. JVector uses additional data stored inline to support two-pass searches, with the first pass powered by lossily compressed representations of the vectors kept in memory, and the second by a more accurate representation read from disk.  The first pass can be performed with
* Product quantization (PQ), optionally with [anisotropic weighting](https://arxiv.org/abs/1908.10396)
* [Binary quantization](https://huggingface.co/blog/embedding-quantization) (BQ)
* Fused PQ, where PQ codebooks are written inline with the graph adjacency list

The second pass can be performed with
* Full resolution float32 vectors
* NVQ, which uses a non-uniform technique to quantize vectors with high-accuracy

[This two-pass design reduces memory usage and reduces latency while preserving accuracy](https://thenewstack.io/why-vector-size-matters/).  

Additionally, JVector is unique in offering the ability to construct the index itself using two-pass searches, allowing larger-than-memory indexes to be built:
![Much larger indexes](https://github.com/jbellis/jvector/assets/42158/34cb8094-68fa-4dc3-b3ce-4582fdbd77e1)

This is important because it allows you to take advantage of logarithmic search within a single index, instead of spilling over to linear-time merging of results from multiple indexes.


## Getting started with JVector

Introductory tutorials for JVector are available in [docs/tutorials](./docs/tutorials/). Start with the [basic tutorial](./docs/tutorials/1-intro-tutorial.md) or review [VectorIntro.java](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/tutorial/VectorIntro.java) for a simple example using JVector.

The older step-by-step guide for JV can be found [here](./docs/legacy/jvector-step-by-step.md). New users should start with the tutorials mentioned earler, but the step-by-step guide contains useful commentary for advanced users.


## The research behind the algorithms

* Foundational work: [HNSW](https://ieeexplore.ieee.org/abstract/document/8594636) and [DiskANN](https://suhasjs.github.io/files/diskann_neurips19.pdf) papers, and [a higher level explainer](https://www.datastax.com/guides/hierarchical-navigable-small-worlds)
* [Anisotropic PQ paper](https://arxiv.org/abs/1908.10396)
* [Quicker ADC paper](https://arxiv.org/abs/1812.09162)
* [NVQ paper](https://arxiv.org/abs/2509.18471)

## Developing and Testing
This project is organized as a [multimodule Maven build](https://maven.apache.org/guides/mini/guide-multiple-modules.html). The intent is to produce a multirelease jar suitable for use as
a dependency from any Java 11 code. When run on a Java 20+ JVM with the Vector module enabled, optimized vector
providers will be used. In general, the project is structured to be built with JDK 20+, but when `JAVA_HOME` is set to
Java 11 -> Java 19, certain build features will still be available.

Base code is in [jvector-base](./jvector-base) and will be built for Java 11 releases, restricting language features and APIs
appropriately. Code in [jvector-twenty](./jvector-twenty) will be compiled for Java 20 language features/APIs and included in the final
multirelease jar targeting supported JVMs. [jvector-multirelease](./jvector-multirelease) packages [jvector-base](./jvector-base) and [jvector-twenty](./jvector-twenty) as a
multirelease jar for release. [jvector-examples](./jvector-examples) is an additional sibling module that uses the reactor-representation of
jvector-base/jvector-twenty to run example code. [jvector-tests](./jvector-tests) contains tests for the project, capable of running against
both Java 11 and Java 20+ JVMs.

To run tests, use `mvn test`. To run tests against Java 20+, use `mvn test`. To run tests against Java 11, use `mvn -Pjdk11 test`.
To run a single test class, use the Maven Surefire test filtering capability, e.g.,
`mvn -Dsurefire.failIfNoSpecifiedTests=false -Dtest=TestNeighborArray test`.
You may also use method-level filtering and patterns, e.g.,
`mvn -Dsurefire.failIfNoSpecifiedTests=false -Dtest=TestNeighborArray#testRetain* test`.
(The `failIfNoSpecifiedTests` option works around a quirk of surefire: it is happy to run `test` with submodules with empty test sets,
but as soon as you supply a filter, it wants at least one match in every submodule.)

You can run `SiftSmall` and `Bench` directly to get an idea of what all is going on here. `Bench` will automatically download required datasets to the `dataset_cache` directories.
The files used by `SiftSmall` can be found in the [siftsmall directory](./siftsmall) in the project root.

To run either class, you can use the Maven exec-plugin via the following incantations:

> `mvn compile exec:exec@bench`

or for Sift:

> `mvn compile exec:exec@sift`

`Bench` takes an optional `benchArgs` argument that can be set to a list of whitespace-separated regexes. If any of the
provided regexes match within a dataset name, that dataset will be included in the benchmark. For example, to run only the glove
and nytimes datasets, you could use:

> `mvn compile exec:exec@bench -DbenchArgs="glove nytimes"`

To run Sift/Bench without the JVM vector module available, you can use the following invocations:

> `mvn -Pjdk11 compile exec:exec@bench`

> `mvn -Pjdk11 compile exec:exec@sift`

The `... -Pjdk11` invocations will also work with `JAVA_HOME` pointing at a Java 11 installation.

For more information on running benchmarks, go through [docs/benchmarking.md](./docs/benchmarking.md).

To release, configure `~/.m2/settings.xml` to point to OSSRH and run `mvn -Prelease clean deploy`.

---

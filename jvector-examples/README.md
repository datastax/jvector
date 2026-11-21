# JVector Examples

JVector comes with the following sample programs to try. Build them once from the repository root,
using JDK 22 or newer:

```bash
mvn -pl jvector-examples -am -DskipTests package -Dnative.onednn=true
JAR=$(ls jvector-examples/target/jvector-examples-*-jar-with-dependencies.jar)
```

`-Dnative.onednn=true` adds the oneDNN library described in the
[native build guide](../jvector-native/src/main/native/README.md#recommended-onednn-build-and-package);
it needs CMake. The programs download public datasets on first use and cache them under
`dataset_cache/` in the current directory; set `DATASET_CACHE_DIR` to cache them elsewhere (see
[datasets](yaml-configs/dataset-catalogs/datasets.md)).

### Hello vector world

[HelloVectorWorld](src/main/java/io/github/jbellis/jvector/example/HelloVectorWorld.java) builds an
in-memory graph over the public `ada002-100k` dataset (1,536 dimensions, about 650 MB to download)
and prints the ten nearest neighbors of one query:

```bash
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp "$JAR" io.github.jbellis.jvector.example.HelloVectorWorld
```

### Hello ASH world

Two self-contained introductions build and query a fused ASH disk index over the
public `e5-small-v2-100k` dataset (downloaded and cached automatically):

- [HelloASHWorldInt8](src/main/java/io/github/jbellis/jvector/example/HelloASHWorldInt8.java):
  INT8 query projection, for x86-64 processors with AVX-512 VNNI or AVX-VNNI.
- [HelloASHWorldFp16](src/main/java/io/github/jbellis/jvector/example/HelloASHWorldFp16.java):
  native FP16 query projection, for x86-64 processors with AVX-512 FP16 (Intel Sapphire Rapids or
  newer); it also needs the oneDNN add-on.

Both teach the same six steps: load vectors, train/encode ASH, construct the graph,
write fused ASH neighborhoods with NVQ reranking vectors, reopen and search
synchronously, then submit individual queries through the queued API.
They use ASH(b=2,d=192) for 384-dimensional inputs. Stored-code precision (`b`)
is independent of query-projection precision; training and encoding remain FP32.
NVQ's mean is collected during ASH encoding, avoiding another input scan.

With the jar built as above:

```bash
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g \
  -cp "$JAR" io.github.jbellis.jvector.example.HelloASHWorldInt8
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g \
  -cp "$JAR" io.github.jbellis.jvector.example.HelloASHWorldFp16
```

Run the program matching your CPU. On Linux,
`grep -o -m1 -w 'avx512_fp16\|avx512_vnni\|avx_vnni' /proc/cpuinfo` lists the features that
apply. An unsupported precision fails instead of silently changing precision; HelloASHWorldFp16
reports this only after it has built and saved its index. Each saves `ash-int8.graph` or `ash-fp16.graph`
and prints top-ten vector ordinals, similarity scores and visited-node counts
for the first held-out query using both synchronous and queued search. An optional
argument selects a new output path;
existing files are never overwritten, so delete the index or pass a new path to run a
program again. Original vectors are not needed to reopen the finished index.

The synchronous section introduces preparation and graph search for one query.
The queued section submits eight independent queries using two application-assigned
workers, admission 16 and maximum preparation batch 8. The shared
[HelloASHQueuedSearch](src/main/java/io/github/jbellis/jvector/example/HelloASHQueuedSearch.java)
shows the host integration explicitly: compatible-query preparation, worker-owned
searchers, completion futures, admission rejection and orderly shutdown. Callers
submit individual requests; the queue prepares currently waiting compatible work
together, without a fill timer. Batch sizes depend on what is waiting at that moment.
Each request owns its input snapshot and prepared state through completion.

For clarity, this small example allocates request records and preparation arrays.
Services can reuse bounded storage as described in
[bounded query batching and state ownership](../docs/ash-query-integration.md).
For more hardware choices, see [projection precision](../docs/ash-projection-precision.md).

### Batched file-backed vectors

Two hello-world programs use public-catalog `e5-small-v2-100k`, or accept an existing
`.fvecs` file:

```bash
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp "$JAR" io.github.jbellis.jvector.example.HelloBatchedVectors
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp "$JAR" io.github.jbellis.jvector.example.HelloBatchedQuantization
```

To use your own file, pass its path (`/path/to/vectors.fvecs`) as the argument. The first
demonstrates scans, selections and vector ownership; the second trains and encodes with PQ and NVQ. See
[Batched input vector access](../docs/batched-vector-access.md) for configuration,
source lifecycle, and integration guidance.

### SiftSmall 
A simple benchmark for the sift dataset located in the [siftsmall](../siftsmall) directory in the project root.

> `mvn compile exec:exec@sift`

### Bench
Builds and measures indexes from YAML descriptions of datasets, index parameters and
measurements (BenchYAML). With no arguments it runs the default set of small public datasets,
downloading them on first use:

> `mvn compile exec:exec@bench`

See [Benchmarking with BenchYAML](../README.md#benchmarking-with-benchyaml) for selecting datasets
and configurations, and [How to run benchmarks](../docs/benchmarking.md) for the details.

To run `SiftSmall`/`Bench` without the JVM vector module available, you can use the following invocations:

> `mvn -Pjdk11 compile exec:exec@bench`

> `mvn -Pjdk11 compile exec:exec@sift`

### IPCService

A simple service for adding / querying vectors over a unix socket.

Install [socat](http://www.dest-unreach.org/socat/) using Homebrew on mac or apt/rpm on linux

Mac:
  > `brew install socat`

Linux:
  > `apt-get install socat`

Start the service with:
  > `mvn compile exec:exec@ipcserve`

Now you can interact with the service
```bash
socat - unix-client:/tmp/jvector.sock

CREATE 3 DOT_PRODUCT 2 20
OK
WRITE [0.1,0.15,0.3]
OK
WRITE [0.2,0.83,0.05]
OK
WRITE [0.5,0.5,0.5]
OK
OPTIMIZE
OK
SEARCH 20 3 [0.15,0.1,0.1]
RESULT [2,1,0]
```

#### Commands
  All commands are completed with `\n`. 
  
  No spaces are allowed inside vector brackets.

  * `CREATE {dimensions} {similarity-function} {M} {searchDepthConstruction}`
    * Creates a new index for this session  
      
  * `WRITE [N,N,N] ... [N,N,N]`
    * Add one or more vectors to the index 
  * `OPTIMIZE`
    * Call when indexing is complete  
  * `MEMORY`
    * Get the in memory size of index  
  * `SEARCH {overquerySearch} {top-k} [N,N,N] ... [N,N,N]` 
    * Search index for the top-k closest vectors (ordinals of indexed values returned per query)
  * `BULKLOAD {localpath}`
    * Bulk loads a local file in numpy format Rows x Columns
    

### Vector prefetching benchmark

Compare preload, synchronous demand reads, and the buffered source with
`io.github.jbellis.jvector.example.VectorPrefetchBenchmark`. Use `--samples N` for training sample size;
each launch samples, trains, and encodes the entire candidate set, running the
three arms sequentially in fresh JVMs. PQ/NVQ and named quantization parameters are selectable.

See [command-line and IntelliJ instructions](../docs/vector-prefetch-benchmark.md)
for cold-file-cache verification, timing boundaries and future vector/loader adapters.

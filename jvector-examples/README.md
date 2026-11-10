# JVector Examples

JVector comes with the following sample programs to try:

### Hello ASH world

Two self-contained introductions build and query a fused ASH disk index over the
public `e5-small-v2-100k` dataset (downloaded and cached automatically):

- [HelloASHWorldInt8](src/main/java/io/github/jbellis/jvector/example/HelloASHWorldInt8.java):
  INT8 query projection for Intel Ice Lake and other supported INT8 processors.
- [HelloASHWorldFp16](src/main/java/io/github/jbellis/jvector/example/HelloASHWorldFp16.java):
  native FP16 query projection for Intel Sapphire Rapids and newer FP16 processors.

Both teach the same six steps: load vectors, train/encode ASH, construct the graph,
write fused ASH neighborhoods with NVQ reranking vectors, reopen and search
synchronously, then submit individual queries through the queued API.
They use ASH(b=2,d=192) for 384-dimensional inputs. Stored-code precision (`b`)
is independent of query-projection precision; training and encoding remain FP32.
NVQ's mean is collected during ASH encoding, avoiding another input scan.

Build the examples with the native provider and oneDNN support described in the
[native build guide](../jvector-native/src/main/native/README.md#recommended-onednn-build-and-package).
From the repository root, using JDK 22 or newer:

```bash
mvn package -DskipTests
java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g \
  -cp jvector-examples/target/jvector-examples-4.1.0-SNAPSHOT-jar-with-dependencies.jar \
  io.github.jbellis.jvector.example.HelloASHWorldInt8
java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g \
  -cp jvector-examples/target/jvector-examples-4.1.0-SNAPSHOT-jar-with-dependencies.jar \
  io.github.jbellis.jvector.example.HelloASHWorldFp16
```

Run the program matching your CPU; explicit unsupported precisions fail instead
of silently changing precision. Each saves `ash-int8.graph` or `ash-fp16.graph`
and prints top-ten vector ordinals, similarity scores and visited-node counts
for the first held-out query using both synchronous and queued search. An optional
argument selects a new output path;
existing files are never overwritten. Original vectors are not needed to reopen
the finished index.

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
`.fvecs` file through `-DvectorFile=/path/to/vectors.fvecs`:

```bash
mvn -Pjdk20 -pl jvector-examples -am compile exec:exec@hello-batched-vectors
mvn -Pjdk20 -pl jvector-examples -am compile exec:exec@hello-batched-quantization
```

Run from the repository root with JDK 23. The first demonstrates scans, selections
and vector ownership; the second trains and encodes with PQ and NVQ. See
[Batched input vector access](../docs/batched-vector-access.md) for configuration,
source lifecycle, and integration guidance.

### SiftSmall 
A simple benchmark for the sift dataset located in the [siftsmall](../siftsmall) directory in the project root.

> `mvn compile exec:exec@sift`

### Bench
Performs grid search across the `GraphIndexBuilder` parameter space to find
the best tradeoffs between recall and throughput.  

This benchmark requires `fvecs' versions of datasets from [https://github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks/blob/main/README.md#data-sets) to be downloaded to `dataset_cache` 
directory under the project root. 

> `mvn compile exec:exec@bench`

Some sample KNN datasets for testing based on ada-002 embeddings generated on wikipedia data are available in ivec/fvec format for testing at:

```
aws s3 ls s3://astra-vector/wikipedia_squad/ --no-sign-request 
                           PRE 100k/
                           PRE 1M/
                           PRE 4M/
```

Bench automatically downloads the 100k dataset to the `./fvec` directory .

To run `SiftSmall`/`Bench` without the JVM vector module available, you can use the following invocations:

> `mvn -Pjdk11 compile exec:exec@bench`

> `mvn -Pjdk11 compile exec:exec@sift`

### IPCService

A simple service for adding / querying vectors over a unix socket.

Install [socat]() using homebrew on mac or apt/rpm on linux

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

Compare preload, synchronous demand reads, and the PR's buffered source with
`io.github.jbellis.jvector.example.VectorPrefetchBenchmark`. Use `--samples N` for training sample size;
each launch samples, trains, and encodes the entire candidate set, running the
three arms sequentially in fresh JVMs. PQ/NVQ and named quantization parameters are selectable.

See [command-line and IntelliJ instructions](../docs/vector-prefetch-benchmark.md)
for cold-file-cache verification, timing boundaries and future vector/loader adapters.

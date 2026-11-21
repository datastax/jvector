# JVector Examples

JVector comes with the following sample programs to try:

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
A simple benchmark for the sift dataset located in the [siftsmall](./siftsmall) directory in the project root.

> `mvn compile exec:exec@sift`

### Bench
Performs grid search across the `GraphIndexBuilder` parameter space to find
the best tradeoffs between recall and throughput.  

This benchmark requires `fvecs' versions of datasets from [https://github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks/blob/main/README.md#data-sets) to be downloaded to `dataset_cache` 
directory under the project root. 

You can use [`plot_output.py`](./plot_output.py) to graph the [pareto-optimal points](https://en.wikipedia.org/wiki/Pareto_efficiency) found by `Bench`.

> `mvn compile exec:exec@bench`

YAML-driven runs (`BenchYAML`, `AutoBenchYAML`) take their search sweep from the `search:` section of
`yaml-configs/index-parameters/*.yml`: `topKOverquery` maps each topK to overquery factors
(rerankK = topK * overquery), `useSearchPruning` lists pruning settings, and the optional `adaptiveGamma`
list (e.g. `adaptiveGamma: [0.02, 0.05, 0.1]`) additionally runs each topK/overquery cell with
[Adaptive Beam Search](https://arxiv.org/abs/2505.15636) termination for each gamma, where rerankK is the
number of results the stopping rule waits for instead of a beam width. Larger gamma searches more and recalls more;
useful values are small (on 1024-dimension embeddings, 0.02 to 0.07 spans recall@10 0.95 to 0.996).
Adaptive rows show a `Gamma` column in the console and fill the `adaptiveGamma` column of `experiments.csv`
(blank for beam-search rows). Without the key, output is beam search only, as before.

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

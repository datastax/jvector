# JVector Examples

JVector comes with the following sample programs to try:

### SiftSmall 
A simple benchmark for the sift dataset located in the [siftsmall](./siftsmall) directory in the project root.

> `mvn compile exec:exec@sift`

### Bench
Performs grid search across the `GraphIndexBuilder` parameter space to find
the best tradeoffs between recall and throughput.  

This benchmark requires datasets from [https://github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks/blob/main/README.md#data-sets) to be downloaded to hdf5 and fvec 
directories `hdf5` or `fvec` under the project root depending on the dataset format. 

You can use [`plot_output.py`](./plot_output.py) to graph the [pareto-optimal points](https://en.wikipedia.org/wiki/Pareto_efficiency) found by `Bench`.

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
    

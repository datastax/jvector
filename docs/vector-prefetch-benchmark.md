# Vector prefetching benchmark

`VectorPrefetchBenchmark` retrieves a reproducible training sample, trains a new
quantizer, calls `encodeAll` on the **entire base candidate set**, then scores the
encoded vectors against one query as a sink. This isolates the input operations
that buffered access can benefit: arbitrary sample retrieval and full-set encoding.
Training and scoring are timed separately; scoring uses encoded resident output,
not the input prefetcher. This is not an index build or recall measurement.

Each launch compares three arms **one at a time**, in fresh JVMs:

1. **preload**: the usual `SiftLoader.readFvecs` full resident preload
   **outside comparison timing**; the sample is then selected
   from resident vectors inside timing. This is the ideal resident-input baseline.
2. **demand**: synchronous positional reads of requested records, without a cache,
   I/O executor, batching or speculative read-ahead. Float decoding matches the PR source.
3. **prefetch**: unmodified `FvecFileVectorValues.open(path)` defaults from this PR,
   through production `VectorAccess` and quantizer APIs. This comparison includes
   the PR's batching/scheduling as well as prefetching; it is not a read-ahead-only ablation.

Catalog lookup/downloads, selecting ordinal IDs and preparing one query happen
before timing. Each fresh JVM initially has no resident base vectors in Java. The
preload arm loads its complete base set before its comparison timer starts; file
arms start timing before opening their source and include required input reads. By default,
64-bit Linux `posix_fadvise(POSIX_FADV_DONTNEED)` evicts only the base input file;
`mincore` verifies zero resident pages before preload preparation or file-arm timing. No privileged global-cache
drop is used. Verification failure stops the comparison, without retries. Run with
no other readers of this input. Do not use eviction on a live database's files.
After the initial cold start, later passes naturally reuse OS cache pages.

`--cache uncontrolled` explicitly skips eviction (also usable on other platforms).
It is labeled **not cold**. Neither policy makes JVM startup, code compilation or
storage-controller caches identical across arms: these are fresh-process pipeline
measurements, not warmed JMH kernel measurements. Repeat in different arm orders
using `--mode` for performance conclusions.

### Command line

Build with JDK 23 from the repository root:

```bash
mvn -Pjdk20 -pl jvector-examples -am -DskipTests package
```

Use the resulting `jvector-examples/target/jvector-examples-*-jar-with-dependencies.jar`
as `$JV_EXAMPLES`. Select an appropriate caller-controlled heap for the resident
baseline; no machine-specific heap or core count is built into these benchmarks.

```bash
java --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -cp "$JV_EXAMPLES" io.github.jbellis.jvector.example.VectorPrefetchBenchmark \
  --dataset cohere-english-v3-10M --samples 128000 --quantizer pq
```

Use any name in your dataset catalog, or `--file /path/to/base.fvecs` instead of
`--dataset`. The default is the public catalog's `e5-small-v2-100k`; sampling defaults
to 128,000 training vectors, or the candidate count if smaller. `DATASET_CACHE_DIR` applies to catalog preparation. Downloaded
files and existing caches are never deleted or rewritten. Files must be immutable.

### IntelliJ

Create an Application configuration using the main class
`io.github.jbellis.jvector.example.VectorPrefetchBenchmark`.

Select the `jvector-examples` module classpath and JDK 23. Use the repository root
as working directory. VM options:

```text
--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

Program arguments are the same as the command line, e.g.
`--dataset cohere-english-v3-10M --samples 128000 --quantizer pq`.
Heap and provider settings from the launching JVM are inherited by each arm;
debugger/profiler agents are not inherited, avoiding port conflicts. Child JVM
output appears in the same IntelliJ console. Stopping the launcher also stops its
own active child. Start only one microbenchmark configuration at a time.

### Options and output

- `--mode all|preload|demand|prefetch`: all is the default; individual modes let you
  choose another order or avoid preload when it cannot fit in memory.
- `--quantizer nvq|pq`: NVQ is the default, with one subvector. PQ uses the production
  training policy, width approximately eight, 256 clusters, no global centering and
  no anisotropic weighting. It needs at least 256 selected candidates. Its internal
  random training is independent across arms, so nearest-vector results can differ.
- `--metric DOT_PRODUCT|EUCLIDEAN|COSINE`: dot product is the default. Values are
  used as stored; no normalization or scrubbing is performed.
- `--query-file /path/to/queries.fvecs`: use its first query. Otherwise, use the last
  base vector as a held-out query and exclude it from all candidate sets. Thus the
  default full scan encodes N−1 candidates, and avoids a trivial self-match.
- `--samples N`, `--seed 42`: training sample count and deterministic ordinal seed; all
  arms use identical ordinals in identical order. The complete ordinal array is
  supplied to `VectorAccess.copySelected` up front; the PR schedules bounded
  asynchronous read-ahead across concurrent selection cursors, rather than
  enqueueing the entire set at once. Count must fit the candidate set.
- `--progress-seconds 30`: phase transitions and infrequent elapsed-time updates;
  `0` disables periodic updates. There is no per-vector logging or monitoring scan.

Console tables show excluded preload preparation, sample retrieval, training, encoding,
sink and comparison total elapsed seconds,
the winning original ordinal, its score and a sum of all scores. For preload, totals
start **after the entire base set is resident**; its file-load preparation is reported
separately and excluded. For demand and prefetch, totals start before file-source
open and include input reads. All totals finish when the sink completes; JVM launch,
download, ordinal generation, cold-cache preparation, and resource closure are excluded.
This measures the penalty of file-backed input relative to already resident input,
not the time to load and process the dataset from scratch. Sample retrieval includes materialization and the loader's source-open work for
file arms. Training operates on those resident samples. Encoding always operates
on the complete candidate source, so file-arm encoding includes required input I/O.
Source-open time alone is not the time to load the dataset. NVQ's `compute` stage
computes the selected training set's mean; per-vector parameter learning is in
full-set encoding. Use `--samples` equal to the candidate count when evaluating
NVQ's usual full-base mean. Retained training vectors are separate from the
prefetcher's bounded I/O payload budget.
Encoded vectors remain resident for the sink; this is not a bounded-total-memory
encoding pipeline. Preload additionally retains the complete raw base set.

Source counters report **logical** reads/bytes, not SSD traffic. No graph is built,
no saved quantizer is reused, and the benchmark does not measure index recall or QPS.

### Quantizer parameters and extension points

Pass named parameters without changing the harness:

```text
--quantizer pq --quantizer-options subspaces=128,clusters=256,center=false,anisotropic=-1
--quantizer nvq --quantizer-options subvectors=2,learn=true
```

Full precision is not a built-in benchmark arm yet. Once the required input hooks
are supported by a full-precision vector implementation, an adapter can materialize
or stream those vectors through `VectorAccess` and supply its defined JVector distance
function as the sink. Do not treat current quantized results as measurements of that
future path.

`--loader-options ioThreads=8,batchVectors=64,readAhead=3,maxBufferBytes=67108864`
configures the existing PR source when using `--mode prefetch`. With no loader
options, the exact PR defaults are used. Run individual modes when passing loader
options because preload/demand reject unused parameters rather than ignore them.

A future quantizer can supply a public no-argument adapter implementing
`VectorPrefetchBenchmark.Quantizer`:

```java
public class MyQuantizerBenchmark implements VectorPrefetchBenchmark.Quantizer {
    public VectorPrefetchBenchmark.Encoder train(RandomAccessVectorValues source,
                                                Map<String, String> parameters,
                                                ForkJoinPool executor) {
        return VectorPrefetchBenchmark.compressed(
                MyQuantizer.compute(source, parameters, executor));
    }
}
```

Run with `--quantizer my.package.MyQuantizerBenchmark`. The existing
`VectorCompressor.encodeAll` and `CompressedVectors.scoreFunctionFor` perform
encoding and scoring; there is no algorithm copied into the benchmark. ASH or a
future representation that does not implement those interfaces can instead return
its own `Encoder` and `Encoded`: encoding accepts the source and compute pool;
`Encoded` supplies `count()` and a per-ordinal `scorer(query, metric)`. The adapter
calls that representation's production methods. The score must be finite, with
larger meaning nearer, as for JVector similarities. This also provides the extension
point for full precision once its required hooks are supported.

Similarly, a public no-argument `VectorPrefetchBenchmark.Loader` adapter opens a
source and returns `VectorPrefetchBenchmark.Input`, exposing `values()` and `close()`.
Use `--mode my.package.MyLoaderBenchmark --loader-options key=value,...`.
The source can implement `BatchedVectorValues` or ordinary `RandomAccessVectorValues`;
the benchmark preserves that capability when restricting a held-out query. A custom
loader must preserve the file's vector count, dimension and ordinal/value mapping.
Adapters are example-harness extension points, not new core-library requirements.

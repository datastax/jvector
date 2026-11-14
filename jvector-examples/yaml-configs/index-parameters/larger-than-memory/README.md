# Larger-than-memory index

Builds an index on disk without holding the source vectors in memory, then serves queries from disk.

**Use case:** Collections whose index does not fit in memory, where each query reads graph nodes from
storage.

**Pipeline:** onDisk: build ASH symmetric -> search fused ASH -> rerank NVQ

The configuration combines the settings that matter at this scale:

- **Construction:** `useDatasetPrefetching` streams base vectors in order, symmetric ASH scoring keeps
  only compact codes in memory, and the parallel writer adds the fused ASH and NVQ features.
- **Layout:** `pageAlignment: 4096` keeps every node record within whole pages, so reading a node costs
  one page read when its record fits in a page. Choose ASH bits and degree so the record (node id +
  NVQ + degree × ASH code + neighbor list) fits in a page, as this file does; the build prints the
  record layout and its padding.
- **Queries:** queued execution with one request per worker, a batch size of 1, and a sweep of worker
  counts. Each worker waits on its own disk reads, so more workers than cores raise throughput until the
  device reaches its read limit; past that, added workers only add latency.

To measure cold queries on a cold index, use the
[larger-than-memory run configuration](../../run-config-larger-than-memory.yml): it evicts the index from
the page cache before its first search (cold index), warms the JIT with dataset vectors rather than the
measured queries, and times one pass in which each held-out query runs once (cold queries). Later search
cases in the same JVM repeat those queries on a warmer index, so for a cold measurement of each worker count,
set one value in `search.workers` and run each in its own JVM.

Run from the repository root:

```bash
java -Djvector.bench.runConfig=jvector-examples/yaml-configs/run-config-larger-than-memory.yml \
  -cp "$BENCH_JAR" io.github.jbellis.jvector.example.BenchYAML \
  jvector-examples/yaml-configs/index-parameters/larger-than-memory/mpnet-10m.yml
```

See [Building and serving larger-than-memory indexes](../../../../docs/larger-than-memory-indexes.md) for
sizing, JVM settings and the corresponding library calls.

# Tuning construction and search

Most of an index's recall, speed and size come from a handful of settings. Start from the
defaults below, measure on your own data with [BenchYAML](benchyaml-guide.md), and change one
setting at a time.

## Construction settings

| Setting | API | BenchYAML | Typical start | What it does |
| --- | --- | --- | --- | --- |
| Max degree | `M` in `GraphIndexBuilder.builder(..., M)` | `outDegree` | 32 | The most neighbors a node keeps. Higher raises recall and the cost of each search step, and makes the index larger and slower to build. |
| Beam width | `withBeamWidth` | `efConstruction` | 100 | How widely each insertion searches for neighbors. Higher builds a better graph more slowly. |
| Neighbor overflow | `withNeighborOverflow` | `neighborOverflow` | 1.2 | Extra neighbors a node may hold temporarily during construction. Larger values build more efficiently but use more memory. Must be at least 1.0. |
| Alpha | `withAlpha` | — | 1.2 | How aggressively neighbors are pruned for diversity. Above 1.0 keeps some longer edges, which helps search cross the graph; 1.0 is usually too strict. |
| Hierarchy | `GraphIndexBuilderConfig` | `addHierarchy` | on | Adds HNSW-style upper layers for faster entry into the graph. |
| Refinement | `GraphIndexBuilderConfig` | `refineFinalGraph` | on | A final pass over every node that improves its neighbors. |

Construction scoring can use full vectors or compact codes. Building from ASH codes
(`ashBuildScoreProvider`, or `scoring: symmetric` in BenchYAML) needs neither the source vectors
nor query projection during insertion, which keeps build memory small; see
[Larger-than-memory indexes](larger-than-memory-indexes.md#construction) and
[ASH graph construction](ash-configuration.md#graph-construction).

## Search settings

A search returns `topK` results. With compressed traversal it first collects `rerankK`
candidates using the codes, then rescores them with the reranking vectors and keeps the best
`topK`:

```java
int topK = 10;
float overquery = 5.0f;                          // try 1, 2, 5, 10
int rerankK = (int) (topK * overquery);
SearchResult sr = searcher.search(scoreProvider, topK, rerankK, 0.0f, 0.0f, Bits.ALL);
```

| Setting | API | BenchYAML | Effect |
| --- | --- | --- | --- |
| Overquery (rerank depth) | `rerankK` = `topK` × overquery | `topKOverquery` | The main recall knob. More candidates raise recall and lower throughput. |
| Threshold | `threshold` | — | Stops once more results above a similarity are unlikely; useful for "all matches above X" queries. Use `Float.NEGATIVE_INFINITY` with unnormalized `DOT_PRODUCT`. |
| Filter | `acceptOrds` (`Bits`) | — | Restricts results to accepted nodes. Very selective filters make searches visit more of the graph. |
| Search pruning | `GraphSearcher` option | `useSearchPruning` | Prunes traversal for supported scorers and similarities. |
| Queued execution (ASH) | `QueryPreparer`, `BatchingExecutor` | `queryExecution: queued` | Prepares waiting ASH queries together and bounds the work admitted; see [Integrating ASH queries](ash-query-integration.md). |

A sweep such as this one measures recall, throughput and latency at several rerank depths from
one build:

```yaml
search:
  topKOverquery:
    10: [1.0, 2.0, 5.0, 10.0]
    100: [1.0, 2.0]
```

## A tuning order that works

1. **Fix the code size first.** It sets the memory budget and the accuracy ceiling; see
   [Choosing an index](choosing-an-index.md) and [Sizing](sizing.md).
2. **Sweep overquery** to find the recall you need at the lowest rerank depth.
3. **If recall tops out too low**, raise degree (for example 32 → 48 or 64) or beam width, or use
   a larger code; rebuild and sweep again.
4. **If builds are too slow**, lower beam width, build from compact codes, or give construction
   more cores.
5. **For indexes served from disk**, use page-aligned records and queued execution with one
   request per worker; see
   [Larger-than-memory indexes](larger-than-memory-indexes.md#serving-queries).

Report recall together with throughput and latency, measured with queries the index has not seen,
so the numbers reflect production behavior; see [Running benchmarks](benchmarking.md).

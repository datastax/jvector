# Graph construction

JVector builds a Vamana-style graph (the algorithm behind DiskANN): each new node searches the
graph for its nearest neighbors, keeps a diverse subset of them, and links back to them. On top of
that base layer it can add an HNSW-style hierarchy. Construction is concurrent and incremental:
many threads can insert at once, and a graph can keep growing after it is first searched.

## Ways to build

| Approach | API | Use it when |
| --- | --- | --- |
| **Batch** | `builder.build(vectors)` | All vectors are available up front; inserts them in parallel and cleans up |
| **Incremental** | `builder.addGraphNode(node, ...)` from any number of threads, then `builder.cleanup()` | Vectors arrive over time, or you control the threads; `markNodeDeleted` and cleanup remove nodes |
| **Streaming ASH** | `StreamingASHIndex` | A database's live index: build and search while vectors arrive, holding only the graph and ASH codes, and move to newer ASH models without rebuilding |
| **Merge** | `OnDiskGraphIndexCompactor` | Combine several on-disk indexes into one; see [Compaction](compaction.md) |
| **Continue a graph** | `GraphIndexBuilder.builder(scoreProvider, dimension, existingGraph)` | Add to a graph that was built or loaded earlier |

`cleanup()` must run before a graph is written or handed to readers: it removes deleted nodes,
optionally refines the graph (below) and trims every neighbor list back to the maximum degree.

## What construction scores with

Every insertion is a search, so the representation used for scoring sets much of the build's
speed and memory.

| Scoring | Provider | Memory | Notes |
| --- | --- | --- | --- |
| Full vectors | `BuildScoreProvider.randomAccessScoreProvider` (or `builder(vectors, similarity, M)`) | The vectors, unless read from a file-backed source | Exact scores; simplest |
| PQ codes | `BuildScoreProvider.pqBuildScoreProvider` | PQ codes | Builds under a memory limit; see [Tutorial 3](tutorials/3-larger-than-memory-tutorial.md) |
| BQ codes | `BuildScoreProvider.bqBuildScoreProvider` | BQ codes | Cheapest scoring, coarsest graph |
| ASH codes, symmetric | `BuildScoreProvider.ashBuildScoreProvider`, inserting with `provider.searchProviderFor(node)` | ASH codes | Compares stored codes with each other; needs neither the source vectors nor query projection during insertion. The fastest ASH option |
| ASH codes, asymmetric | `ashBuildScoreProvider`, inserting with the original vector | ASH codes, plus each inserted vector while it is inserted | Scores the original vector against codes |

[ASH graph construction](ash-configuration.md#graph-construction) lists every ASH scoring policy,
and [Larger-than-memory indexes](larger-than-memory-indexes.md#construction) shows a complete
symmetric build that streams vectors from a file.

## Hierarchical or flat

| | Hierarchical (default) | Flat |
| --- | --- | --- |
| Layers | A base layer with every node, plus upper layers holding a random, shrinking sample of nodes, as in HNSW | The base layer only |
| Where a search starts | The top layer's entry node; each layer narrows the start for the next | One fixed entry node (the first node inserted) |
| Final refinement | Available (`refineFinalGraph`) | Not run: refinement applies only to graphs with upper layers |
| On disk | Upper layers are loaded into memory; the base layer can stay on disk | Nothing extra to load |

Each node's top layer is drawn at random, so the number of nodes in each upper layer falls by
about a factor of the degree per layer, and the upper layers add little memory. Searches start
closer to the query's region, so they usually take fewer steps. Keep the hierarchy unless you have
measured that a flat graph does as well on your data.

Hierarchy and refinement are set JVM-wide through `GraphIndexBuilderConfig` (also exposed over
JMX), or per builder through the deprecated constructors that take `addHierarchy` and
`refineFinalGraph`. In BenchYAML they are `addHierarchy` and `refineFinalGraph`, and
`upperLayerDegree` sets a different maximum degree for the upper layers; in the API,
`GraphIndexBuilder.builder(scoreProvider, dimension, maxDegrees)` takes one degree per layer.

## Settings

Degree, beam width, neighbor overflow and alpha trade build time and size against recall; see
[Tuning construction and search](tuning.md#construction-settings).

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.ForkJoinPool;

public class InMemoryGraphIndexBuilder {
    private static final Logger logger = LoggerFactory.getLogger(InMemoryGraphIndexBuilder.class);

    private int beamWidth = 100;
    private int maxConnections = 32;
    private int dimension;
    private float neighborOverflow = 1.2f;
    private float alpha = 1.2f;
    private boolean addHierarchy = true;
    private boolean refineFinalGraph = true;
    private List<Integer> maxDegrees = List.of();
    //TODO: set up build score provider
    private BuildScoreProvider scoreProvider;

    ForkJoinPool simdExecutor = new ForkJoinPool(Runtime.getRuntime().availableProcessors() / 2, ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, false);
    ForkJoinPool parallelExecutor = new ForkJoinPool(Runtime.getRuntime().availableProcessors(), ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, false);

    public InMemoryGraphIndexBuilder() {
    }

    public InMemoryGraphIndexBuilder withMaxConnections(int maxConnections) {
        if (maxConnections <= 0) {
            throw new IllegalArgumentException("max connections must be positive");
        }
        this.maxConnections = maxConnections;
        return this;
    }

    public InMemoryGraphIndexBuilder withMaxDegrees(List<Integer> maxDegrees) {
        this.maxDegrees = maxDegrees;
        return this;
    }

    public InMemoryGraphIndexBuilder withBeamWidth(int beamWidth) {
        if (beamWidth <= 0) {
            throw new IllegalArgumentException("beamWidth must be positive");
        }
        this.beamWidth = beamWidth;
        return this;
    }

    public InMemoryGraphIndexBuilder withDimension(int dimension) {
        this.dimension = dimension;
        return this;
    }

    public InMemoryGraphIndexBuilder withNeighborOverflow(float neighborOverflow) {
        if (neighborOverflow < 1.0f) {
            throw new IllegalArgumentException("neighborOverflow must be >= 1.0");
        }
        this.neighborOverflow = neighborOverflow;
        return this;
    }

    public InMemoryGraphIndexBuilder withAlpha(float alpha) {
        if (alpha <= 0) {
            throw new IllegalArgumentException("alpha must be positive");
        }
        this.alpha = alpha;
        return this;
    }

    public InMemoryGraphIndexBuilder withAddHierarchy(boolean addHierarchy) {
        this.addHierarchy = addHierarchy;
        return this;
    }

    public InMemoryGraphIndexBuilder withRefineFinalGraph(boolean refineFinalGraph) {
        this.refineFinalGraph = refineFinalGraph;
        return this;
    }

    public InMemoryGraphIndexBuilder withScoreProvider(BuildScoreProvider scoreProvider) {
        this.scoreProvider = scoreProvider;
        return this;
    }

    public InMemoryGraphIndexBuilder withSimdExecutor(ForkJoinPool simdExecutor) {
        this.simdExecutor = simdExecutor;
        return this;
    }

    public InMemoryGraphIndexBuilder withParallelExecutor(ForkJoinPool parallelExecutor) {
        this.parallelExecutor = parallelExecutor;
        return this;
    }

    public InMemoryGraphIndex build() {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive");
        }
        return new OnHeapGraphIndex(!maxDegrees.isEmpty() ? maxDegrees : List.of(maxConnections),
                dimension,
                neighborOverflow,
                new VamanaDiversityProvider(scoreProvider, alpha),
                addHierarchy);
    }

//    public MemoryGraphIndex buildAndPopulate(RandomAccessVectorValues ravv) {
//        MemoryGraphIndex index = build();
//        populateGraph(index, ravv);
//        return index;
//    }
//
//    public MemoryGraphIndex populateGraph(MemoryGraphIndex index, RandomAccessVectorValues ravv) {
//        var vv = ravv.threadLocalSupplier();
//        int size = ravv.size();
//
//        simdExecutor.submit(() -> {
//            IntStream.range(0, size).parallel().forEach(node -> {
//                addGraphNode(node, vv.get().getVector(node));
//            });
//        }).join();
//
//        cleanup();
//        return index;
//    }
//
//    public long addGraphNode(int node, VectorFloat<?> vector) {
//        var ssp = scoreProvider.searchProviderFor(vector);
//        return addGraphNode(node, ssp);
//    }
//
//    public long addGraphNode(int node, SearchScoreProvider searchScoreProvider) {
//        var nodeLevel = new ImmutableGraphIndex.NodeAtLevel(getRandomGraphLevel(), node);
//        // do this before adding to in-progress, so a concurrent writer checking
//        // the in-progress set doesn't have to worry about uninitialized neighbor sets
//        graph.addNode(nodeLevel);
//
//        insertionsInProgress.add(nodeLevel);
//        var inProgressBefore = insertionsInProgress.clone();
//        try (var gs = searchers.get()) {
//            var view = graph.getView();
//            gs.setView(view); // new snapshot
//            var naturalScratchPooled = naturalScratch.get();
//            var concurrentScratchPooled = concurrentScratch.get();
//
//            var bits = new GraphIndexBuilder.ExcludingBits(nodeLevel.node);
//            var entry = view.entryNode();
//            SearchResult result;
//            if (entry == null) {
//                result = new SearchResult(new SearchResult.NodeScore[] {}, 0, 0, 0, 0, 0);
//            } else {
//                gs.initializeInternal(searchScoreProvider, entry, bits);
//
//                // Move downward from entry.level to 1
//                for (int lvl = entry.level; lvl > 0; lvl--) {
//                    if (lvl > nodeLevel.level) {
//                        gs.searchOneLayer(searchScoreProvider, 1, 0.0f, lvl, gs.getView().liveNodes());
//                    } else {
//                        gs.searchOneLayer(searchScoreProvider, beamWidth, 0.0f, lvl, gs.getView().liveNodes());
//                        SearchResult.NodeScore[] neighbors = new SearchResult.NodeScore[gs.approximateResults.size()];
//                        AtomicInteger index = new AtomicInteger();
//                        // TODO extract an interface that lets us avoid the copy here and in toScratchCandidates
//                        gs.approximateResults.foreach((neighbor, score) -> {
//                            neighbors[index.getAndIncrement()] = new SearchResult.NodeScore(neighbor, score);
//                        });
//                        Arrays.sort(neighbors);
//                        updateNeighborsOneLayer(lvl, nodeLevel.node, neighbors, naturalScratchPooled, inProgressBefore, concurrentScratchPooled, searchScoreProvider);
//                    }
//                    gs.setEntryPointsFromPreviousLayer();
//                }
//
//                // Now do the main search at layer 0
//                result = gs.resume(beamWidth, beamWidth, 0.0f, 0.0f);
//            }
//
//            updateNeighborsOneLayer(0, nodeLevel.node, result.getNodes(), naturalScratchPooled, inProgressBefore, concurrentScratchPooled, searchScoreProvider);
//
//            graph.markComplete(nodeLevel);
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        } finally {
//            insertionsInProgress.remove(nodeLevel);
//        }
//
//        return IntStream.rangeClosed(0, nodeLevel.level).mapToLong(graph::ramBytesUsedOneNode).sum();
//    }
//
//    public void cleanup() {
//        if (graph.size(0) == 0) {
//            return;
//        }
//        validateEntryNode(); // sanity check before we start
//
//        // purge deleted nodes.
//        // backlinks can cause neighbors to soft-overflow, so do this before neighbors cleanup
//        removeDeletedNodes();
//
//        if (graph.size(0) == 0) {
//            // After removing all the deleted nodes, we might end up with an empty graph.
//            // The calls below expect a valid entry node, but we do not have one right now.
//            return;
//        }
//
//        if (refineFinalGraph && graph.getMaxLevel() > 0) {
//            // improve connections on everything in L1 & L0.
//            // It may be helpful for 2D use cases, but empirically it seems unnecessary for high-dimensional vectors.
//            // It may bring a slight improvement in recall for small maximum degrees,
//            // but it can be easily be compensated by using a slightly larger neighborOverflow.
//            parallelExecutor.submit(() -> {
//                graph.nodeStream(1).parallel().forEach(this::improveConnections);
//            }).join();
//        }
//
//        // clean up overflowed neighbor lists
//        parallelExecutor.submit(() -> {
//            IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(id -> {
//                for (int level = 0; level <= graph.getMaxLevel(); level++) {
//                    graph.enforceDegree(id);
//                }
//            });
//        }).join();
//
//        graph.setAllMutationsCompleted();
//    }
//
//    public int getDegree(int level) {
//        if (level >= maxDegrees.size()) {
//            return maxDegrees.get(maxDegrees.size() - 1);
//        }
//        return maxDegrees.get(level);
//    }

//    private int getRandomGraphLevel(MemoryGraphIndex graph) {
//        double ml;
//        double randDouble;
//        if (addHierarchy) {
//            ml = graph.getDegree(0) == 1 ? 1 : 1 / log(1.0 * graph.getDegree(0));
//            do {
//                randDouble = this.rng.nextDouble();  // avoid 0 value, as log(0) is undefined
//            } while (randDouble == 0.0);
//        } else {
//            ml = 0;
//            randDouble = 0;
//        }
//        return ((int) (-log(randDouble) * ml));
//    }

}

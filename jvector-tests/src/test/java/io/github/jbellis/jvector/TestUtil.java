/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector;

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedADC;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.carrotsearch.randomizedtesting.RandomizedTest.getRandom;
import static org.junit.Assert.assertEquals;

public class TestUtil {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** min .. max inclusive on both ends, to match Lucene's */
    public static int nextInt(Random random, int min, int max) {
        return min + random.nextInt(1 + max - min);
    }

    public static BufferedRandomAccessWriter openBufferedWriter(Path outputPath) throws IOException {
        return new BufferedRandomAccessWriter(outputPath);
    }

    public static DataOutputStream openDataOutputStream(Path outputPath) throws IOException {
        return new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(outputPath)));
    }

    /**
     * Deletes target Path and its contents (if applicable). Never throws an exception. Only suitable for tests.
     * Visits all levels of file tree and does not follow symlinks.
     * Exceptions will terminate the walk and print errors on stderr.
     * @param targetPath Path to delete
     */
    public static void deleteQuietly(Path targetPath) {
        try {
            Files.walkFileTree(targetPath, new FileVisitor<>() {
                @Override
                public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    Files.delete(file);
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFileFailed(Path file, IOException exc) {
                    System.err.println("deleteQuietly encountered an Exception when visiting file: " + exc.toString());
                    return FileVisitResult.TERMINATE;

                }

                @Override
                public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                    if (exc != null) {
                        System.err.println("deleteQuietly encountered an Exception after visiting directory: " + exc);
                        return FileVisitResult.TERMINATE;
                    }
                    Files.delete(dir);
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException e) {
            System.err.println("deleteQuietly encountered an Exception: " + e);
        }
    }

    public static VectorFloat<?> randomVector(Random random, int dim) {
        var vec = vectorTypeSupport.createFloatVector(dim);
        for (int i = 0; i < dim; i++) {
            vec.set(i, random.nextFloat());
            if (random.nextBoolean()) {
                vec.set(i, -vec.get(i));
            }
        }
        VectorUtil.l2normalize(vec);
        return vec;
    }

    public static List<VectorFloat<?>> createRandomVectors(int count, int dimension) {
        return IntStream.range(0, count).mapToObj(i -> TestUtil.randomVector(getRandom(), dimension)).collect(Collectors.toList());
    }

    public static VectorFloat<?> normalRandomVector(Random random, int dim) {
        var vec = vectorTypeSupport.createFloatVector(dim);
        for (int i = 0; i < dim; i++) {
            vec.set(i, (float) random.nextGaussian());
        }
        return vec;
    }

    public static List<VectorFloat<?>> createNormalRandomVectors(int count, int dimension) {
        return IntStream.range(0, count).mapToObj(i -> TestUtil.normalRandomVector(getRandom(), dimension)).collect(Collectors.toList());
    }

    public static void writeGraph(GraphIndex graph, RandomAccessVectorValues ravv, Path outputPath) throws IOException {
        OnDiskGraphIndex.write(graph, ravv, outputPath);
    }

    public static void writeFusedGraph(GraphIndex graph, RandomAccessVectorValues ravv, PQVectors pqv, Path outputPath) throws IOException {
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, outputPath)
                .with(new InlineVectors(ravv.dimension()))
                .with(new FusedADC(graph.maxDegree(), pqv.getCompressor())).build())
        {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            suppliers.put(FeatureId.INLINE_VECTORS, ordinal -> new InlineVectors.State(ravv.getVector(ordinal)));
            suppliers.put(FeatureId.FUSED_ADC, ordinal -> new FusedADC.State(graph.getView(), pqv, ordinal));
            writer.write(suppliers);
        }
    }

    public static Set<Integer> getNeighborNodes(GraphIndex.View g, int level, int node) {
      Set<Integer> neighbors = new HashSet<>();
      for (var it = g.getNeighborsIterator(level, node); it.hasNext(); ) {
        int n = it.nextInt();
        neighbors.add(n);
      }
      return neighbors;
    }

    static List<Integer> sortedNodes(GraphIndex h, int level) {
          var graphNodes = h.getNodes(level); // TODO
          List<Integer> nodes = new ArrayList<>();
          while (graphNodes.hasNext()) {
              nodes.add(graphNodes.next());
          }
          Collections.sort(nodes);
          return nodes;
      }

    public static void assertGraphEquals(GraphIndex g, GraphIndex h) {
        // construct these up front since they call seek which will mess up our test loop
        String prettyG = GraphIndex.prettyPrint(g);
        String prettyH = GraphIndex.prettyPrint(h);
        assertEquals(String.format("the number of nodes in the graphs are different:%n%s%n%s",
                                   prettyG,
                                   prettyH),
                     g.size(),
                     h.size());

        assertEquals(g.getView().entryNode(), h.getView().entryNode());
        for (int level = 0; level <= g.getMaxLevel(); level++) {
            // assert equal nodes in each graph
            List<Integer> hNodes = sortedNodes(h, level);
            List<Integer> gNodes = sortedNodes(g, level);
            assertEquals(String.format("nodes in the graphs are different:%n%s%n%s",
                                       prettyG,
                                       prettyH),
                         gNodes,
                         hNodes);

            // assert equal nodes' neighbours in each graph
            NodesIterator gNodesIterator = g.getNodes(level);
            var gv = g.getView();
            var hv = h.getView();
            while (gNodesIterator.hasNext()) {
                int node = gNodesIterator.nextInt();
                assertEqualsLazy(() -> String.format("arcs differ for node %d%n%s%n%s",
                                                     node,
                                                     prettyG,
                                                     prettyH),
                                 getNeighborNodes(gv, level, node),
                                 getNeighborNodes(hv, level, node));
            }
        }
    }

    /**
     * For when building the failure message is expensive
     */
    public static void assertEqualsLazy(Supplier<String> f, Set<Integer> s1, Set<Integer> s2) {
        if (!s1.equals(s2)) {
            throw new AssertionError(f.get());
        }
    }

    public static OnHeapGraphIndex buildSequentially(GraphIndexBuilder builder, RandomAccessVectorValues vectors) {
        for (var i = 0; i < vectors.size(); i++) {
            builder.addGraphNode(i, vectors.getVector(i));
        }
        builder.cleanup();
        return builder.getGraph();
    }

    public static class FullyConnectedGraphIndex implements GraphIndex {
        private final int entryNode;
        private final List<Integer> layerSizes;

        public FullyConnectedGraphIndex(int entryNode, int size) {
            this(entryNode, List.of(size));
        }

        public FullyConnectedGraphIndex(int entryNode, List<Integer> layerSizes) {
            this.entryNode = entryNode;
            this.layerSizes = layerSizes;
        }

        @Override
        public int size(int level) {
            return layerSizes.get(level);
        }

        @Override
        public int maxDegree() {
            return layerSizes.stream().mapToInt(i -> i).max().orElseThrow();
        }

        @Override
        public NodesIterator getNodes(int level) {
            int n = layerSizes.get(level);
            return new NodesIterator.ArrayNodesIterator(IntStream.range(0, n).toArray(), n);
        }

        @Override
        public View getView() {
            return new FullyConnectedGraphIndexView();
        }

        @Override
        public int getDegree(int level) {
            return layerSizes.get(level) - 1;
        }

        @Override
        public int getMaxLevel() {
            return layerSizes.size() - 1;
        }

        @Override
        public void close() { }

        private class FullyConnectedGraphIndexView implements View {
            @Override
            public NodesIterator getNeighborsIterator(int level, int node) {
                return new NodesIterator.ArrayNodesIterator(IntStream.range(0, layerSizes.get(level))
                                                                    .filter(i -> i != node).toArray(),
                                                            layerSizes.get(level) - 1);
            }

            @Override
            public int size() {
                return FullyConnectedGraphIndex.this.size(0);
            }

            @Override
            public NodeAtLevel entryNode() {
                return new NodeAtLevel(layerSizes.size() - 1, entryNode);
            }

            @Override
            public Bits liveNodes() {
                return Bits.ALL;
            }

            @Override
            public void close() { }
        }

        @Override
        public long ramBytesUsed() {
            throw new UnsupportedOperationException();
        }
    }

    public static class RandomlyConnectedGraphIndex implements GraphIndex {
        private final List<CommonHeader.LayerInfo> layerInfo;
        private final List<Map<Integer, int[]>> layerAdjacency;
        private final int entryNode;

        public RandomlyConnectedGraphIndex(List<CommonHeader.LayerInfo> layerInfo, Random random) {
            this.layerInfo = layerInfo;
            this.layerAdjacency = new ArrayList<>(layerInfo.size());
            
            // Build adjacency for each layer
            for (int level = 0; level < layerInfo.size(); level++) {
                int size = layerInfo.get(level).size;
                int maxNeighbors = layerInfo.get(level).degree;
                Map<Integer, int[]> adjacency = new ConcurrentHashMap<>();

                // Generate node IDs in random order
                var nodeIds = IntStream.range(0, size).boxed().collect(Collectors.toCollection(ArrayList::new));
                Collections.shuffle(nodeIds, random);

                // Fill adjacency
                for (int i = 0; i < size; i++) {
                    Set<Integer> neighborSet = new HashSet<>();
                    while (neighborSet.size() < maxNeighbors) {
                        int neighborIdx = random.nextInt(size);
                        if (neighborIdx != i) {
                            neighborSet.add(nodeIds.get(neighborIdx));
                        }
                    }
                    adjacency.put(nodeIds.get(i), neighborSet.stream().mapToInt(Integer::intValue).toArray());
                }
                layerAdjacency.add(adjacency);
            }
            
            // Pick an entry node from the top layer
            this.entryNode = random.nextInt(layerInfo.get(layerInfo.size() - 1).size);
        }

        public RandomlyConnectedGraphIndex(int size, int M, Random random) {
            this(List.of(new CommonHeader.LayerInfo(size, M)), random);
        }

        @Override
        public int getMaxLevel() {
            return layerInfo.size() - 1;
        }

        @Override
        public int size(int level) {
            return layerInfo.get(level).size;
        }

        @Override
        public NodesIterator getNodes(int level) {
            int sz = layerInfo.get(level).size;
            return new NodesIterator.ArrayNodesIterator(IntStream.range(0, sz).toArray(), sz);
        }

        @Override
        public View getView() {
            return new RandomlyConnectedGraphIndexView();
        }

        @Override
        public int getDegree(int level) {
            return layerInfo.get(level).degree;
        }

        @Override
        public int maxDegree() {
            return layerInfo.stream().mapToInt(li -> li.degree).max().orElseThrow();
        }

        @Override
        public void close() { }

        private class RandomlyConnectedGraphIndexView implements View {
            @Override
            public NodesIterator getNeighborsIterator(int level, int node) {
                var adjacency = layerAdjacency.get(level);
                return new NodesIterator.ArrayNodesIterator(adjacency.get(node));
            }

            public int size() {
                return layerInfo.get(0).size;
            }

            @Override
            public NodeAtLevel entryNode() {
                return new NodeAtLevel(getMaxLevel(), entryNode);
            }

            @Override
            public Bits liveNodes() {
                return Bits.ALL;
            }

            @Override
            public void close() { }
        }

        @Override
        public long ramBytesUsed() {
            throw new UnsupportedOperationException();
        }
    }
}

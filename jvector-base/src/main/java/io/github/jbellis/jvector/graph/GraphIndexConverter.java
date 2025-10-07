package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.graph.diversity.VamanaDiversityProvider;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.max;

class GraphIndexConverter {
    /**
     * Converts an OnDiskGraphIndex to an OnHeapGraphIndex by copying all nodes, their levels, and neighbors,
     * along with other configuration details, from disk-based storage to heap-based storage.
     *
     * @param immutableGraphIndex the disk-based index to be converted
     * @param bsp The build score provider to be used for
     * @param overflowRatio usually 1.2f
     * @param alpha usually 1.2f
     * @return an OnHeapGraphIndex that is equivalent to the provided OnDiskGraphIndex but operates in heap memory
     * @throws IOException if an I/O error occurs during the conversion process
     */
    public static MutableGraphIndex convertToHeap(ImmutableGraphIndex immutableGraphIndex,
                                                  BuildScoreProvider bsp,
                                                  float overflowRatio,
                                                  float alpha) throws IOException {

        // Create a new OnHeapGraphIndex with the appropriate configuration
        List<Integer> maxDegrees = new ArrayList<>();
        for (int level = 0; level <= immutableGraphIndex.getMaxLevel(); level++) {
            maxDegrees.add(immutableGraphIndex.getDegree(level));
        }

        MutableGraphIndex index = new OnHeapGraphIndex(
                maxDegrees,
                overflowRatio, // overflow ratio
                new VamanaDiversityProvider(bsp, alpha) // diversity provider - can be null for basic usage
        );

        // Copy all nodes and their connections from disk to heap
        try (var view = immutableGraphIndex.getView()) {
            // Copy nodes level by level
            for (int level = 0; level <= immutableGraphIndex.getMaxLevel(); level++) {
                final NodesIterator nodesIterator = immutableGraphIndex.getNodes(level);

                while (nodesIterator.hasNext()) {
                    int nodeId = nodesIterator.next();

                    var sf = bsp.searchProviderFor(nodeId).scoreFunction();

                    var neighborsIterator = view.getNeighborsIterator(level, nodeId);

                    NodeArray nodeArray = new NodeArray(neighborsIterator.size());
                    while(neighborsIterator.hasNext()) {
                        int neighbor = neighborsIterator.nextInt();
                        float score = sf.similarityTo(neighbor);
                        nodeArray.addInOrder(neighbor, score);
                    }

                    // Add the node with its neighbors
                    index.connectNode(level, nodeId, nodeArray);
                    index.markComplete(new ImmutableGraphIndex.NodeAtLevel(level, nodeId));
                }
            }

            // Set the entry point
            index.updateEntryNode(view.entryNode());
        }

        return index;
    }
}

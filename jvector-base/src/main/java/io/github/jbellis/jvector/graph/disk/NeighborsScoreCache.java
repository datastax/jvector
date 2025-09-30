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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Cache containing pre-computed neighbor scores, organized by levels and nodes.
 * <p>
 * This cache bridges the gap between {@link OnDiskGraphIndex} and {@link OnHeapGraphIndex}:
 * <ul>
 * <li>{@link OnDiskGraphIndex} stores only neighbor IDs (not scores) for space efficiency</li>
 * <li>{@link OnHeapGraphIndex} requires neighbor scores for pruning operations</li>
 * </ul>
 * <p>
 * When converting from disk to heap representation, this cache avoids expensive score 
 * recomputation by providing pre-calculated neighbor scores for all graph levels.
 *
 * @see OnHeapGraphIndex#convertToHeap(OnDiskGraphIndex, NeighborsScoreCache, BuildScoreProvider, float, float)
 *
 * This is particularly useful when merging new nodes into an existing graph.
 * @see GraphIndexBuilder#buildAndMergeNewNodes(OnDiskGraphIndex, NeighborsScoreCache, RandomAccessVectorValues, BuildScoreProvider, int, int[], int, float, float, boolean)
 */
public class NeighborsScoreCache {
    private final Map<Integer, Map<Integer, NodeArray>> perLevelNeighborsScoreCache;

    public NeighborsScoreCache(OnHeapGraphIndex graphIndex) throws IOException {
        try (OnHeapGraphIndex.FrozenView view = graphIndex.getFrozenView()) {
            final Map<Integer, Map<Integer, NodeArray>> perLevelNeighborsScoreCache = new HashMap<>(graphIndex.getMaxLevel() + 1);
            for (int level = 0; level <= graphIndex.getMaxLevel(); level++) {
                final Map<Integer, NodeArray> levelNeighborsScores = new HashMap<>(graphIndex.size(level) + 1);
                final NodesIterator nodesIterator = graphIndex.getNodes(level);
                while (nodesIterator.hasNext()) {
                    final int nodeId = nodesIterator.nextInt();

                    ConcurrentNeighborMap.NeighborIterator neighborIterator = (ConcurrentNeighborMap.NeighborIterator) view.getNeighborsIterator(level, nodeId);
                    final NodeArray neighbours = neighborIterator.merge(new NodeArray(neighborIterator.size()));
                    levelNeighborsScores.put(nodeId, neighbours);
                }

                perLevelNeighborsScoreCache.put(level, levelNeighborsScores);
            }

            this.perLevelNeighborsScoreCache = perLevelNeighborsScoreCache;
        }
    }

    public NeighborsScoreCache(RandomAccessReader in) throws IOException {
        final int numberOfLevels = in.readInt();
        perLevelNeighborsScoreCache = new HashMap<>(numberOfLevels);
        for (int i = 0; i < numberOfLevels; i++) {
            final int level = in.readInt();
            final int numberOfNodesInLevel = in.readInt();
            final Map<Integer, NodeArray> levelNeighborsScores = new HashMap<>(numberOfNodesInLevel);
            for (int j = 0; j < numberOfNodesInLevel; j++) {
                final int nodeId = in.readInt();
                final int numberOfNeighbors = in.readInt();
                final NodeArray nodeArray = new NodeArray(numberOfNeighbors);
                for (int k = 0; k < numberOfNeighbors; k++) {
                    final int neighborNodeId = in.readInt();
                    final float neighborScore = in.readFloat();
                    nodeArray.insertSorted(neighborNodeId, neighborScore);
                }
                levelNeighborsScores.put(nodeId, nodeArray);
            }
            perLevelNeighborsScoreCache.put(level, levelNeighborsScores);
        }
    }

    public void write(IndexWriter out) throws IOException {
        out.writeInt(perLevelNeighborsScoreCache.size()); // write the number of levels
        for (Map.Entry<Integer ,Map<Integer, NodeArray>> levelNeighborsScores : perLevelNeighborsScoreCache.entrySet()) {
            final int level = levelNeighborsScores.getKey();
            out.writeInt(level);
            out.writeInt(levelNeighborsScores.getValue().size()); // write the number of nodes in the level
            // Write the neighborhoods for each node in the level
            for (Map.Entry<Integer, NodeArray> nodeArrayEntry : levelNeighborsScores.getValue().entrySet()) {
                final int nodeId = nodeArrayEntry.getKey();
                out.writeInt(nodeId);
                final NodeArray nodeArray = nodeArrayEntry.getValue();
                out.writeInt(nodeArray.size()); // write the number of neighbors for the node
                // Write the nodeArray(neighbors)
                for (int i = 0; i < nodeArray.size(); i++) {
                    out.writeInt(nodeArray.getNode(i));
                    out.writeFloat(nodeArray.getScore(i));
                }
            }
        }
    }

    public Map<Integer, NodeArray> getNeighborsScoresInLevel(int level) {
        return perLevelNeighborsScoreCache.get(level);
    }


}

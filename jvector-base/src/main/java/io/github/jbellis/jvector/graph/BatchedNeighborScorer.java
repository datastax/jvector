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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.ArrayUtil;

/**
 * Implements {@link ImmutableGraphIndex.View#processNeighbors} by scoring all unvisited neighbors of a node with one
 * {@link ScoreFunction#similarityTo(int[], int, float[])} call, then handing them to the processor in neighbor order.
 * The processor sees the same nodes, scores and order as it would if each neighbor were scored as it was found.
 * <p>
 * Keeps scratch space, so each View needs its own, like the View itself.
 */
@Experimental
public final class BatchedNeighborScorer {
    private int[] nodes = new int[32];
    private float[] scores = new float[32];

    /**
     * Marks each of {@code neighbors} visited, scores the ones that were not visited before in one bulk call, and
     * passes them with their scores to {@code neighborProcessor} in the order the iterator returned them.
     */
    public void processNeighbors(NodesIterator neighbors, ScoreFunction scoreFunction, ImmutableGraphIndex.IntMarker visited,
                                 ImmutableGraphIndex.NeighborProcessor neighborProcessor) {
        int count = 0;
        while (neighbors.hasNext()) {
            int friendOrd = neighbors.nextInt();
            if (visited.mark(friendOrd)) {
                if (count == nodes.length) {
                    nodes = ArrayUtil.grow(nodes, count + 1);
                }
                nodes[count++] = friendOrd;
            }
        }
        if (count == 0) {
            return;
        }
        if (scores.length < count) {
            scores = new float[nodes.length];
        }
        scoreFunction.similarityTo(nodes, count, scores);
        for (int i = 0; i < count; i++) {
            neighborProcessor.process(nodes[i], scores[i]);
        }
    }
}

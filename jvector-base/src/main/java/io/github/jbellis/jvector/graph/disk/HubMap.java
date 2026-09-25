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

import java.util.*;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.Int2IntHashMap;

/**
 * Resident copy of the largest source's upper layers (levels 1 and above): adjacency and vectors
 * of every upper-layer node, so region keys and cells can be assigned by a greedy descent and a
 * beam over the hierarchy without touching the source file per query.
 */
final class HubMap {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    final VectorSimilarityFunction similarityFunction;
    final boolean cosine;
    final int dimension;
    final int topLevel;
    int entryNode;
    // per level (index 0 unused): node ids, adjacency with stride degree[level] (-1 padded), and
    // the nodes' vectors position-major, all indexed by the node's position in nodes[level]
    final int[] degree;
    final int[][] nodes;
    final int[][] adjacency;
    final VectorFloat<?>[] vectors;
    final float[][] norms;          // COSINE only: |v| per position
    // node id -> position: a flat array for level 1 (the bulk of the upper layers), maps above it
    int[] level1Position;
    final Int2IntHashMap[] upperPosition;   // levels >= 2: node id -> position (level 1 uses level1Position)

    HubMap(OnDiskGraphIndex hub, VectorSimilarityFunction similarityFunction, int dimension) {
        this.similarityFunction = similarityFunction;
        this.cosine = similarityFunction == VectorSimilarityFunction.COSINE;
        this.dimension = dimension;
        this.topLevel = hub.getMaxLevel();
        this.degree = new int[topLevel + 1];
        this.nodes = new int[topLevel + 1][];
        this.adjacency = new int[topLevel + 1][];
        this.vectors = new VectorFloat<?>[topLevel + 1];
        this.norms = new float[topLevel + 1][];
        this.upperPosition = new Int2IntHashMap[topLevel + 1];
    }

    /** Per-thread query state. */
    final class Scorer {
        private final VectorFloat<?> query = vectorTypeSupport.createFloatVector(dimension);
        private float queryNorm;

        void setQuery(VectorFloat<?> q) {
            query.copyFrom(q, 0, 0, dimension);
            queryNorm = cosine ? (float) Math.sqrt(VectorUtil.dotProduct(q, q)) : 1f;
        }

        /** Higher is closer. */
        float score(int level, int position) {
            VectorFloat<?> v = vectors[level];
            int o = position * dimension;
            switch (similarityFunction) {
                case EUCLIDEAN:
                    return -VectorUtil.squareL2Distance(query, 0, v, o, dimension);
                case COSINE:
                    return VectorUtil.dotProduct(query, 0, v, o, dimension) / Math.max(1e-12f, norms[level][position] * queryNorm);
                default:
                    return VectorUtil.dotProduct(query, 0, v, o, dimension);
            }
        }
    }

    Scorer scorer() {
        return new Scorer();
    }

    int position(int level, int node) {
        if (level == 1) {
            return node >= 0 && node < level1Position.length ? level1Position[node] : -1;
        }
        return upperPosition[level].get(node);   // missing value is -1
    }

    /**
     * Greedy descent from the entry node down to level 1 under the scorer's current query.
     *
     * @return the level-1 node the descent lands on
     */
    int descend(Scorer scorer) {
        int current = entryNode;
        for (int level = topLevel; level >= 1; level--) {
            int position = position(level, current);
            if (position < 0) {
                break;
            }
            float currentScore = scorer.score(level, position);
            int stride = degree[level];
            int[] adj = adjacency[level];
            boolean improved = true;
            while (improved) {
                improved = false;
                for (int j = 0; j < stride; j++) {
                    int neighbor = adj[position * stride + j];
                    if (neighbor < 0) {
                        break;
                    }
                    int neighborPosition = position(level, neighbor);
                    if (neighborPosition < 0) {
                        continue;
                    }
                    float score = scorer.score(level, neighborPosition);
                    if (score > currentScore) {
                        currentScore = score;
                        current = neighbor;
                        position = neighborPosition;
                        improved = true;
                    }
                }
            }
        }
        return current;
    }

    /**
     * Breadth-first walk over the level-1 graph, restarting at the lowest unvisited position
     * when a component is exhausted.
     *
     * @return walk position per node id (Integer.MAX_VALUE for nodes not on level 1)
     */
    int[] walkPositions() {
        int[] l1 = nodes[1];
        int n = l1.length;
        int stride = degree[1];
        int[] adj = adjacency[1];
        int[] positionOf = new int[level1Position.length];
        Arrays.fill(positionOf, Integer.MAX_VALUE);
        boolean[] seen = new boolean[n];
        int[] queue = new int[n];
        int head = 0, tail = 0, emitted = 0, nextUnseen = 0;
        while (emitted < n) {
            if (head == tail) {
                while (nextUnseen < n && seen[nextUnseen]) {
                    nextUnseen++;
                }
                if (nextUnseen >= n) {
                    break;
                }
                seen[nextUnseen] = true;
                queue[tail++] = nextUnseen;
            }
            int x = queue[head++];
            positionOf[l1[x]] = emitted++;
            for (int j = 0; j < stride; j++) {
                int neighbor = adj[x * stride + j];
                if (neighbor < 0) {
                    break;
                }
                int y = position(1, neighbor);
                if (y >= 0 && !seen[y]) {
                    seen[y] = true;
                    queue[tail++] = y;
                }
            }
        }
        return positionOf;
    }
}

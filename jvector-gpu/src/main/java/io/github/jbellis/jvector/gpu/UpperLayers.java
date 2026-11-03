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

package io.github.jbellis.jvector.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The layers above the base layer of a hierarchical graph (HNSW-style). Every node gets a level from JVector's own
 * distribution; layer {@code L} holds the nodes with level {@code >= L}, a few percent of the nodes for layer 1 and
 * fewer for each layer above. Each layer is built like the base layer, from exact candidates: they are cheap at
 * these sizes. The layers use the wide candidate list, as their long edges connect distant parts of the graph.
 * <p>
 * The levels are known before the build starts, so the build can copy the upper-layer vectors out early
 * ({@link #gather}) and compute the layers ({@link #build}) while the base layer is being installed.
 */
final class UpperLayers {

    /** One layer: its nodes, and their neighbor lists by position in {@code members}. */
    record Layer(int level, int[] members, GpuGraphBuilder.Adjacency adjacency) {
    }

    /** The nodes on layer 1 or above, in increasing order, with their levels and a copy of their rows. */
    record Nodes(int[] ids, int[] levels, MemorySegment rows) {
    }

    private UpperLayers() {
    }

    /** Copies out the rows of every node with level 1 or above. */
    static Nodes gather(MemorySegment rows, int[] levels, int dim, Arena arena) {
        int[] ids = IntStream.range(0, levels.length).filter(i -> levels[i] > 0).toArray();
        int[] upperLevels = Arrays.stream(ids).map(i -> levels[i]).toArray();
        return new Nodes(ids, upperLevels, HostRows.gather(rows, ids, dim, arena));
    }

    /**
     * Builds every upper layer, bottom up.
     *
     * @param nodes      from {@link #gather}
     * @param candidates exact candidates per node (fewer when a layer is smaller)
     */
    static List<Layer> build(Nodes nodes, int dim, Candidates.Metric metric, boolean normalise, boolean euclidean,
                             int candidates, int maxDegree, float alpha, Arena arena) {
        int top = Arrays.stream(nodes.levels()).max().orElse(0);
        List<Layer> layers = new ArrayList<>(top);
        for (int level = 1; level <= top; level++) {
            int lvl = level;
            // positions in nodes.ids() of this layer's members
            int[] positions = IntStream.range(0, nodes.ids().length).filter(x -> nodes.levels()[x] >= lvl).toArray();
            int[] members = Arrays.stream(positions).map(x -> nodes.ids()[x]).toArray();
            int m = members.length;
            if (m == 1) {
                layers.add(new Layer(lvl, members, GpuGraphBuilder.Adjacency.empty(1)));
                continue;
            }
            MemorySegment rows = lvl == 1 ? nodes.rows() : HostRows.gather(nodes.rows(), positions, dim, arena);
            var exact = Candidates.exact(rows, m, dim, Math.min(candidates, m - 1), metric);
            var adjacency = GpuGraphBuilder.build(PackedVectors.of(rows, m, dim, normalise, euclidean), m, exact, maxDegree, alpha);
            layers.add(new Layer(lvl, members, adjacency));
        }
        return layers;
    }
}

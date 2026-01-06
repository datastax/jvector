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

import io.github.jbellis.jvector.graph.similarity.SimilarityFunction;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorRepresentation;

import java.io.Closeable;

/**
 * Encapsulates the state of a graph for searching.  Re-usable across search calls,
 * but each thread needs its own.
 */
public interface GraphIndexView<Primary extends VectorRepresentation, Secondary extends VectorRepresentation> extends Closeable {
    /**
     * The steps needed to process a neighbor during a search. That is, adding it to the priority queue, etc.
     */
    interface NeighborProcessor {
        void process(int friendOrd, float similarity);
    }

    /**
     * Serves as an abstract interface for marking nodes as visited
     */
    @FunctionalInterface
    interface IntMarker {
        /**
         * Marks the node and returns true if it was not marked previously. Returns false otherwise
         */
        boolean mark(int value);
    }

    /**
     * Iterator over the neighbors of a given node.  Only the most recently instantiated iterator
     * is guaranteed to be valid.
     */
    NodesIterator getNeighborsIterator(int level, int node);

    /**
     * Iterates over the neighbors of a given node if they have not been visited yet.
     * For each non-visited neighbor, it computes its similarity and processes it using the given processor.
     * The processor is meant to add the best nodes to a queue, for example.
     */
    void processNeighbors(int level, int node, SimilarityFunction<Primary> similarityFunction, IntMarker visited, NeighborProcessor neighborProcessor);

    /**
     * This method is deprecated as most View usages should not need size.
     * Where they do, they could access the graph.
     * @return the number of nodes in the graph
     */
    @Deprecated
    int size();

    /**
     * @return the node of the graph to start searches at
     */
    ImmutableGraphIndex.NodeAtLevel entryNode();

    /**
     * Return a Bits instance indicating which nodes are live.  The result is undefined for
     * ordinals that do not correspond to nodes in the graph.
     */
    Bits liveNodes();

    /**
     * @return the largest ordinal id in the graph.  May be different from size() if nodes have been deleted.
     */
    default int getIdUpperBound() {
        return size();
    }

    Primary getPrimaryRepresentation(int node);

    Secondary getSecondaryRepresentation(int node);

    /**
     * Updates the view with the latest changes to the underlying graph if any.
     * To establish a point of consistency in the graph, this method needs to be called before each search.
     */
    void refresh();
}

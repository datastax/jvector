/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.index;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.Closeable;
import java.io.IOException;
import java.util.Objects;

/**
 * Represents a graph-based vector index.  Nodes are represented as ints, and edges are
 * represented as adjacency lists.
 * <p>
 * Mostly this applies to any graph index, but a few methods (e.g. getVector()) are
 * specifically included to support the DiskANN-based design of OnDiskGraphIndex.
 * <p>
 * All methods are threadsafe.  Operations that require persistent state are wrapped
 * in a View that should be created per accessing thread.
 */
public interface Index extends AutoCloseable, Accountable {
    VectorSimilarityFunction getSimilarityFunction();

    Searcher getSearcher();

    IndexWriter getWriter();

    int size();

    /**
     * Encapsulates the state of a graph for searching.  Re-usable across search calls,
     * but each thread needs its own.
     */
    interface View extends Closeable {
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
        GraphIndex.NodeAtLevel entryNode();

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
    }
}

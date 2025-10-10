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

package io.github.jbellis.jvector.graph;

import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;

/**
 * Iterator over graph nodes that includes the size –- the total
 * number of nodes to be iterated over. The nodes are NOT guaranteed to be presented in any
 * particular order.
 */
public interface NodesIterator extends PrimitiveIterator.OfInt {
    /**
     * Returns the number of elements in this iterator.
     *
     * @return the size of this iterator
     */
    int size();

    /**
     * Creates a NodesIterator from a primitive iterator and size.
     *
     * @param iterator the primitive iterator to wrap
     * @param size the number of elements
     * @return a NodesIterator wrapping the given iterator
     */
    static NodesIterator fromPrimitiveIterator(PrimitiveIterator.OfInt iterator, int size) {
        return new NodesIterator() {
            @Override
            public int size() {
                return size;
            }

            @Override
            public int nextInt() {
                return iterator.nextInt();
            }

            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }
        };
    }

    /**
     * An iterator over an array of node IDs.
     */
    class ArrayNodesIterator implements NodesIterator {
        private final int[] nodes;
        private int cur = 0;
        private final int size;

        /**
         * Constructs an iterator based on an integer array representing nodes.
         *
         * @param nodes the array of node IDs
         * @param size the number of valid elements in the array
         */
        public ArrayNodesIterator(int[] nodes, int size) {
            assert nodes != null;
            assert size <= nodes.length;
            this.size = size;
            this.nodes = nodes;
        }

        @Override
        public int size() {
            return size;
        }

        /**
         * Constructs an iterator for the entire array.
         *
         * @param nodes the array of node IDs
         */
        public ArrayNodesIterator(int[] nodes) {
            this(nodes, nodes.length);
        }

        @Override
        public int nextInt() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            if (nodes == null) {
                return cur++;
            } else {
                return nodes[cur++];
            }
        }

        @Override
        public boolean hasNext() {
            return cur < size;
        }
    }

    /**
     * A singleton empty node iterator.
     */
    EmptyNodeIterator EMPTY_NODE_ITERATOR = new EmptyNodeIterator();

    /**
     * An empty node iterator implementation.
     */
    class EmptyNodeIterator implements NodesIterator {
        /** Package-private constructor. */
        EmptyNodeIterator() {
        }
        @Override
        public int size() {
            return 0;
        }

        @Override
        public int nextInt() {
            throw new NoSuchElementException();
        }

        @Override
        public boolean hasNext() {
            return false;
        }
    }
}

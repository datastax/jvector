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
     * Returns the total number of elements that this iterator will produce.
     * @return the count of elements in this iterator
     */
    int size();

    /**
     * Creates a NodesIterator from a standard primitive integer iterator with a known size.
     * @param iterator the source iterator providing node IDs
     * @param size the total number of elements the iterator will produce
     * @return a NodesIterator wrapping the provided iterator
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
     * Implementation of NodesIterator that iterates over a subset of an integer array.
     */
    class ArrayNodesIterator implements NodesIterator {
        private final int[] nodes;
        private int cur = 0;
        private final int size;

        /**
         * Constructs an iterator over a portion of the given array.
         * @param nodes the array containing node IDs
         * @param size the number of elements to iterate over (must be &lt;= nodes.length)
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
         * Constructs an iterator over the entire array.
         * @param nodes the array containing node IDs to iterate over
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
     * Shared instance of an empty iterator that contains no elements.
     */
    EmptyNodeIterator EMPTY_NODE_ITERATOR = new EmptyNodeIterator();

    /**
     * An iterator implementation that contains no elements. Always returns false for hasNext()
     * and throws NoSuchElementException for nextInt().
     */
    class EmptyNodeIterator implements NodesIterator {
        /**
         * Constructs an empty node iterator with no elements.
         */
        public EmptyNodeIterator() {
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

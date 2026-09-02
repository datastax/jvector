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

import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Bits;

import java.io.IOException;

/**
 * One source's base-layer adjacency, resident, built from its {@link NodeTokenStream}: what a
 * cross-source search needs to walk the graph without touching a record. Upper levels are
 * already resident in the {@link OnDiskGraphIndex} (loaded whole at open), so a
 * {@link #view(OnDiskGraphIndex.View) view} serves level 0 from these arrays and delegates every
 * other question — upper-level edges, the entry node, liveness, containment — to the source's
 * own view. Scoring is the searcher's business: with a score function over the merge's
 * pre-encode cache ({@code AdcScorer}), a search over this view reads nothing from disk.
 *
 * <p>Layout: per node {@code degree + 1} ints — the edge count, then the edges, {@code -1}
 * padded — in chunks of {@code 2^26} ints, so a source of any size fits without a single array
 * above 256 MB. {@link #bytes()} is what it costs; the compactor budgets sources into residency
 * largest-first, since the largest sources are the search targets.
 */
final class ResidentGraph {
    private static final int CHUNK_BITS = 26;
    private static final int CHUNK_INTS = 1 << CHUNK_BITS;
    private static final int CHUNK_MASK = CHUNK_INTS - 1;

    final int nodeCount;
    final int degree;
    final int stride;
    private final int[][] chunks;

    private ResidentGraph(int nodeCount, int degree) {
        this.nodeCount = nodeCount;
        this.degree = degree;
        this.stride = degree + 1;
        long ints = (long) nodeCount * stride;
        int n = (int) ((ints + CHUNK_INTS - 1) >>> CHUNK_BITS);
        chunks = new int[Math.max(n, 1)][];
        for (int c = 0; c < chunks.length; c++) {
            long remaining = ints - ((long) c << CHUNK_BITS);
            chunks[c] = new int[(int) Math.max(0, Math.min(CHUNK_INTS, remaining))];
        }
    }

    /** Bytes the resident form of a source with these dimensions would take. */
    static long bytesFor(int nodeCount, int degree) {
        return (long) nodeCount * (degree + 1) * Integer.BYTES;
    }

    long bytes() {
        return bytesFor(nodeCount, degree);
    }

    /** Builds from the source's token stream; the source must carry one. */
    static ResidentGraph fromStream(OnDiskGraphIndex source) throws IOException {
        try (NodeTokenStream.Reader ts = source.openTokenStream()) {
            ResidentGraph g = new ResidentGraph(ts.nodeCount, ts.degree);
            while (ts.next()) {
                int node = ts.ordinal();
                int count = ts.live() ? ts.neighborCount() : 0;
                if (count > g.degree) {
                    throw new IOException("node " + node + " has " + count + " edges, degree " + g.degree);
                }
                long base = (long) node * g.stride;
                g.set(base, count);
                int[] nbs = ts.neighbors();
                for (int k = 0; k < count; k++) {
                    g.set(base + 1 + k, nbs[k]);
                }
                for (int k = count; k < g.degree; k++) {
                    g.set(base + 1 + k, -1);
                }
            }
            return g;
        }
    }

    private void set(long index, int value) {
        chunks[(int) (index >>> CHUNK_BITS)][(int) (index & CHUNK_MASK)] = value;
    }

    private int get(long index) {
        return chunks[(int) (index >>> CHUNK_BITS)][(int) (index & CHUNK_MASK)];
    }

    int neighborCount(int node) {
        return get((long) node * stride);
    }

    int neighbor(int node, int k) {
        return get((long) node * stride + 1 + k);
    }

    /** A view for one thread: level 0 from these arrays, everything else from {@code delegate}. */
    ImmutableGraphIndex.View view(OnDiskGraphIndex.View delegate) {
        return new View(delegate);
    }

    final class View implements ImmutableGraphIndex.View {
        private final OnDiskGraphIndex.View delegate;
        private final int[] buffer = new int[degree];

        View(OnDiskGraphIndex.View delegate) {
            this.delegate = delegate;
        }

        @Override
        public NodesIterator getNeighborsIterator(int level, int node) {
            if (level != 0) {
                return delegate.getNeighborsIterator(level, node);
            }
            int count = neighborCount(node);
            for (int k = 0; k < count; k++) {
                buffer[k] = neighbor(node, k);
            }
            return new NodesIterator.ArrayNodesIterator(buffer, count);
        }

        @Override
        public void processNeighbors(int level, int node, ScoreFunction scoreFunction, ImmutableGraphIndex.IntMarker visited,
                                     ImmutableGraphIndex.NeighborProcessor neighborProcessor) {
            if (level != 0) {
                delegate.processNeighbors(level, node, scoreFunction, visited, neighborProcessor);
                return;
            }
            int count = neighborCount(node);
            for (int k = 0; k < count; k++) {
                int friend = neighbor(node, k);
                if (visited.mark(friend)) {
                    neighborProcessor.process(friend, scoreFunction.similarityTo(friend));
                }
            }
        }

        @Override
        public int size() {
            return delegate.size();
        }

        @Override
        public ImmutableGraphIndex.NodeAtLevel entryNode() {
            return delegate.entryNode();
        }

        @Override
        public Bits liveNodes() {
            return delegate.liveNodes();
        }

        @Override
        public int getIdUpperBound() {
            return delegate.getIdUpperBound();
        }

        @Override
        public boolean contains(int level, int node) {
            return level == 0 ? node >= 0 && node < nodeCount : delegate.contains(level, node);
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }
    }
}

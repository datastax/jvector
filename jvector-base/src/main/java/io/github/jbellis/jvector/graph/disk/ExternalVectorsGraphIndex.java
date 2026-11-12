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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2IntHashMap;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.function.Supplier;

/**
 * An {@link OnDiskGraphIndex} whose full-precision vectors come from the caller instead of from the
 * graph file. Topology, features and fused codes are still read from the file; only
 * {@link View#getVector}, {@link View#getVectorInto} and the dimension are answered by the supplied
 * {@link RandomAccessVectorValues}, indexed by this graph's own ordinals.
 * <p>
 * Lets {@link OnDiskGraphIndexCompactor} merge graphs that were written without their vectors
 * (inline vectors of dimension 0), or whose stored vectors the caller overrides. Every vector read of
 * the merge, PQ retraining and the cross-source searches goes through a view, so wrapping the source
 * is enough. Each view takes its own instance from {@link RandomAccessVectorValues#threadLocalSupplier()},
 * and a view is confined to one thread, like the reader it holds.
 * <p>
 * Only live nodes are ever read from the caller. A store that keeps its vectors elsewhere has usually
 * let go of a deleted one, yet the cross-source searches still walk through deleted nodes and score
 * them to find their way. A deleted node is therefore answered by a live node of its neighbourhood: its
 * first live neighbour, else the first live neighbour of one of its neighbours, else any live node of
 * the graph. Its neighbours are the nodes it was closest to when it was inserted, so the walk keeps
 * heading the right way. Measured on random 64-dimensional data with a third of the nodes deleted, this
 * costs about 0.4 points of recall@10 against the deleted vectors themselves, where a fixed placeholder
 * vector costs 1.6.
 */
final class ExternalVectorsGraphIndex extends OnDiskGraphIndex {
    private static final int NO_STAND_IN = -1;

    private final Supplier<RandomAccessVectorValues> vectors;
    private final int vectorDimension;
    private final FixedBitSet live;
    // deleted node -> the live node read in its place; NO_STAND_IN only when the graph has no live node
    private final Int2IntHashMap standIns;
    // what a deleted node answers when the graph has no live node at all: any finite vector will do,
    // since nothing of such a graph reaches the output
    private final VectorFloat<?> noLiveNode;

    ExternalVectorsGraphIndex(OnDiskGraphIndex graph, RandomAccessVectorValues vectors, FixedBitSet live) {
        super(graph);
        this.vectors = vectors.threadLocalSupplier();
        this.vectorDimension = vectors.dimension();
        this.live = live;
        this.standIns = standInsForDeletedNodes(live);
        this.noLiveNode = vectorTypeSupport.createFloatVector(vectorDimension);
        for (int d = 0; d < vectorDimension; d++) {
            noLiveNode.set(d, 1.0f);
        }
    }

    private Int2IntHashMap standInsForDeletedNodes(FixedBitSet live) {
        int deleted = idUpperBound - live.cardinality();
        var map = new Int2IntHashMap(Math.max(8, deleted * 2), 0.65f, NO_STAND_IN);
        if (deleted == 0) {
            return map;
        }
        int anyLive = live.nextSetBit(0);
        if (anyLive == DocIdSetIterator.NO_MORE_DOCS) {
            anyLive = NO_STAND_IN;
        }
        try (var view = super.getView()) {
            int[] neighbours = new int[maxDegree()];
            for (int node = 0; node < idUpperBound; node++) {
                if (live.get(node)) {
                    continue;
                }
                int standIn = NO_STAND_IN;
                int count = 0;
                var it = view.getNeighborsIterator(0, node);
                while (it.hasNext() && standIn == NO_STAND_IN) {
                    int nb = it.nextInt();
                    if (live.get(nb)) {
                        standIn = nb;
                    } else if (count < neighbours.length) {
                        neighbours[count++] = nb;
                    }
                }
                // Every neighbour is deleted as well: one hop further. The iterator reuses the view's
                // buffer, which is why the first hop was copied out.
                for (int i = 0; i < count && standIn == NO_STAND_IN; i++) {
                    var second = view.getNeighborsIterator(0, neighbours[i]);
                    while (second.hasNext()) {
                        int nb = second.nextInt();
                        if (live.get(nb)) {
                            standIn = nb;
                            break;
                        }
                    }
                }
                map.put(node, standIn != NO_STAND_IN ? standIn : anyLive);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return map;
    }

    @Override
    public int getDimension() {
        return vectorDimension;
    }

    @Override
    public View getView() {
        try {
            return new ExternalVectorsView(readerSupplier.get());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    final class ExternalVectorsView extends View {
        private final RandomAccessVectorValues values;

        ExternalVectorsView(RandomAccessReader reader) {
            super(reader);
            this.values = vectors.get();
        }

        @Override
        public int dimension() {
            return vectorDimension;
        }

        @Override
        public VectorFloat<?> getVector(int node) {
            VectorFloat<?> vec = vectorTypeSupport.createFloatVector(vectorDimension);
            getVectorInto(node, vec, 0);
            return vec;
        }

        @Override
        public void getVectorInto(int node, VectorFloat<?> vector, int offset) {
            if (live.get(node)) {
                values.getVectorInto(node, vector, offset);
                return;
            }
            int standIn = standIns.get(node);
            if (standIn != NO_STAND_IN) {
                values.getVectorInto(standIn, vector, offset);
            } else {
                vector.copyFrom(noLiveNode, 0, offset, vectorDimension);
            }
        }
    }
}

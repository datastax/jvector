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

import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.graph.similarity.QueryPreparer;
import io.github.jbellis.jvector.quantization.FusedPQDecoder;
import io.github.jbellis.jvector.quantization.NVQScorer;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Objects;

/**
 * Prepares queries for an on-disk graph that scores with fused PQ, NVQ or inline float vectors, by the similarity
 * function the graph records. Preparation does once what scoring otherwise does at the start of every search:
 * fused PQ's lookup table, and NVQ's per-query state. For a graph with only inline vectors it just snapshots the
 * query. Fused ASH graphs use {@link io.github.jbellis.jvector.quantization.ASHQueryPreparer} instead.
 */
final class OnDiskQueryPreparer implements QueryPreparer {
    private final OnDiskGraphIndex graph;
    private final VectorSimilarityFunction similarity;
    private final ProductQuantization pq;
    private final NVQ nvq;

    OnDiskQueryPreparer(OnDiskGraphIndex graph, VectorSimilarityFunction similarity) {
        this.graph = graph;
        this.similarity = Objects.requireNonNull(similarity, "similarity");
        var fusedPQ = (FusedPQ) graph.getFeatures().get(FeatureId.FUSED_PQ);
        this.pq = fusedPQ == null ? null : fusedPQ.getPQ();
        // inline vectors rerank in preference to NVQ, as they do for unprepared queries
        this.nvq = graph.getFeatures().containsKey(FeatureId.INLINE_VECTORS) ? null
                : (NVQ) graph.getFeatures().get(FeatureId.NVQ_VECTORS);
    }

    @Override
    public PreparedQuery newState() {
        return new State(this);
    }

    @Override
    public boolean supportsBatchPreparation() {
        return false;
    }

    @Override
    public void prepare(VectorFloat<?> query, PreparedQuery preparedState) {
        State state = owned(preparedState);
        validate(query);
        state.ready = false;
        state.query.copyFrom(query, 0, 0, graph.getDimension());
        if (pq != null) {
            state.queryMagnitudeSquared = FusedPQDecoder.fillPartialSums(pq, state.query, similarity, state.partialSums);
        }
        if (nvq != null) {
            state.nvqFunction = nvq.scoreFunctionFor(state.query, similarity);
        }
        state.ready = true;
    }

    @Override
    public void prepareBatch(VectorFloat<?>[] queries, PreparedQuery[] preparedStates, int count) {
        Objects.requireNonNull(queries, "queries");
        Objects.requireNonNull(preparedStates, "preparedStates");
        Objects.checkFromIndexSize(0, count, queries.length);
        Objects.checkFromIndexSize(0, count, preparedStates.length);
        // Structural validation precedes modification.
        for (int i = 0; i < count; i++) {
            State state = owned(preparedStates[i]);
            validate(queries[i]);
            for (int j = 0; j < i; j++) {
                if (preparedStates[j] == state) {
                    throw new IllegalArgumentException("Batch output states must be distinct");
                }
            }
        }
        for (int i = 0; i < count; i++) {
            prepare(queries[i], preparedStates[i]);
        }
    }

    private void validate(VectorFloat<?> query) {
        Objects.requireNonNull(query, "query");
        if (query.length() != graph.getDimension()) {
            throw new IllegalArgumentException("Query dimension " + query.length() + " does not match the index's "
                    + graph.getDimension());
        }
    }

    private State owned(PreparedQuery query) {
        return ownedBy(graph, query);
    }

    private static State ownedBy(OnDiskGraphIndex graph, PreparedQuery query) {
        Objects.requireNonNull(query, "preparedQuery");
        if (!(query instanceof State) || ((State) query).graph != graph) {
            throw new IllegalArgumentException("Prepared query belongs to another index");
        }
        return (State) query;
    }

    /** {@code query} as a state prepared for {@code graph}. */
    static State prepared(OnDiskGraphIndex graph, PreparedQuery query) {
        State state = ownedBy(graph, query);
        if (!state.ready) {
            throw new IllegalStateException("Prepared query is not ready; prepare it first");
        }
        return state;
    }

    @Override
    public long ramBytesUsed() {
        return RamUsageEstimator.shallowSizeOf(this);
    }

    static final class State implements PreparedQuery {
        final OnDiskGraphIndex graph;
        final VectorSimilarityFunction similarity;
        final VectorFloat<?> query;
        /** Fused PQ's lookup table, or null without fused PQ. */
        final VectorFloat<?> partialSums;
        /** The query's squared magnitude for a COSINE fused PQ table, otherwise 0. */
        float queryMagnitudeSquared;
        /** NVQ's per-query scorer, or null without NVQ. */
        NVQScorer.NVQScoreFunction nvqFunction;
        boolean ready;

        private State(OnDiskQueryPreparer preparer) {
            var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
            this.graph = preparer.graph;
            this.similarity = preparer.similarity;
            this.query = vts.createFloatVector(graph.getDimension());
            this.partialSums = preparer.pq == null ? null
                    : vts.createFloatVector(preparer.pq.getSubspaceCount() * preparer.pq.getClusterCount());
        }

        @Override
        public VectorFloat<?> queryVector() {
            return query;
        }

        @Override
        public long ramBytesUsed() {
            // NVQ's per-query state is rebuilt by each preparation and not counted
            return RamUsageEstimator.shallowSizeOf(this) + query.ramBytesUsed()
                    + (partialSums == null ? 0 : partialSums.ramBytesUsed());
        }
    }
}

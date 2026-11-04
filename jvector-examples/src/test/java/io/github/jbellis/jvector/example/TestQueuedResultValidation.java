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
package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.benchmarks.QueuedQueryExecution;
import io.github.jbellis.jvector.example.benchmarks.datasets.SimpleDataSet;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.similarity.*;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.Assert.*;

/** Separate numerical batch differences from corrupted prepared-state handoffs. */
public class TestQueuedResultValidation {
    private static VectorFloat<?> vector(float value) {
        return VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(new float[]{value});
    }

    private static final class State implements PreparedQuery {
        VectorFloat<?> query;
        float projected;
        int searches;
        public VectorFloat<?> queryVector() { return query; }
        public long ramBytesUsed() { return 0; }
    }

    private static final class Preparer implements QueryPreparer {
        public PreparedQuery newState() { return new State(); }
        public boolean supportsBatchPreparation() { return true; }
        public long ramBytesUsed() { return 0; }
        public void prepare(VectorFloat<?> query, PreparedQuery output) {
            State state = (State) output;
            state.query = query; state.projected = query.get(0); state.searches = 0;
        }
        public void prepareBatch(VectorFloat<?>[] inputs, PreparedQuery[] outputs, int count) {
            for (int i = 0; i < count; i++) {
                prepare(inputs[i], outputs[i]);
                // Deliberately distinguish batch arithmetic from singleton
                // arithmetic. Validation must compare the same prepared state.
                ((State) outputs[i]).projected += .00001f;
            }
        }
    }

    private static final class Graph implements ImmutableGraphIndex {
        final AtomicInteger bindings = new AtomicInteger();
        final boolean corrupt;
        Graph(boolean corrupt) { this.corrupt = corrupt; }
        public NodesIterator getNodes(int level) { return new NodesIterator.ArrayNodesIterator(new int[]{0}); }
        public int maxDegree() { return 1; }
        public List<Integer> maxDegrees() { return List.of(1); }
        public int getDimension() { return 1; }
        public boolean isHierarchical() { return false; }
        public int getMaxLevel() { return 0; }
        public int getDegree(int level) { return 1; }
        public double getAverageDegree(int level) { return 0; }
        public int size(int level) { return 1; }
        public long ramBytesUsed() { return 0; }
        public void close() {}
        public ScoringView getView() {
            return new ScoringView() {
                public NodesIterator getNeighborsIterator(int level, int node) { return NodesIterator.EMPTY_NODE_ITERATOR; }
                public void processNeighbors(int level, int node, ScoreFunction score, IntMarker visited, NeighborProcessor processor) {}
                public int size() { return 1; }
                public NodeAtLevel entryNode() { return new NodeAtLevel(0, 0); }
                public Bits liveNodes() { return Bits.ALL; }
                public boolean contains(int level, int node) { return level == 0 && node == 0; }
                public void close() {}
                public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> query, VectorSimilarityFunction similarity) {
                    return node -> query.get(0);
                }
                public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarity) {
                    return node -> query.get(0);
                }
                public SearchScoreProvider scoreProviderFor(PreparedQuery prepared, boolean rerank) {
                    State state = (State) prepared;
                    bindings.incrementAndGet();
                    float score = state.projected + (corrupt && state.searches++ > 0 ? .1f : 0);
                    return new DefaultSearchScoreProvider((ScoreFunction.ExactScoreFunction) node -> score);
                }
            };
        }
    }

    private Grid.ConfiguredSystem system(Graph graph, int workers) {
        var queries = new ArrayList<VectorFloat<?>>();
        for (int i = 0; i < 97; i++) queries.add(vector(.1f + i / 1000f));
        var data = new SimpleDataSet("queued-validation", VectorSimilarityFunction.DOT_PRODUCT,
                List.of(vector(1)), queries, java.util.Collections.nCopies(queries.size(), List.of(0)));
        return new Grid.ConfiguredSystem(data, graph, null, Set.of()) {
            final Preparer preparer = new Preparer();
            @Override public int queryWorkerCount() { return workers; }
            @Override public QueryPreparer queryPreparer() { return preparer; }
        };
    }

    @Test(timeout=20000)
    public void identicalPreparedMathValidatesAcrossWorkersAndNormalPassDoesNotSearchTwice() throws Exception {
        for (int workers : new int[]{1, 3}) {
            var graph = new Graph(false);
            try (var cs = system(graph, workers); var queued = new QueuedQueryExecution(cs, 1, 1, false)) {
                var queries = cs.getDataSet().getQueryVectors();
                queued.verifyConcurrentResults(queries.size(), queries::get);
                assertEquals(2 * queries.size(), graph.bindings.get());
                graph.bindings.set(0);
                var pass = queued.runPass(queries.size(), queries::get, null, true);
                assertEquals(queries.size(), graph.bindings.get());
                for (int i = 0; i < queries.size(); i++)
                    assertEquals(queries.get(i).get(0) + .00001f, pass.results[i].getNodes()[0].score, 0);
                queued.verifyConcurrentResults(0, i -> { throw new AssertionError("Empty validation fetched input"); });
                queued.beginMeasurements();
                assertThrows(IllegalStateException.class, () -> queued.verifyConcurrentResults(0, queries::get));
            }
        }
    }

    @Test(timeout=10000)
    public void corruptedPreparedStateStillFailsValidation() throws Exception {
        try (var cs = system(new Graph(true), 2); var queued = new QueuedQueryExecution(cs, 1, 1, false)) {
            var queries = cs.getDataSet().getQueryVectors();
            var error = assertThrows(IllegalStateException.class,
                    () -> queued.verifyConcurrentResults(queries.size(), queries::get));
            assertTrue(error.getMessage().startsWith("Concurrent prepared query mismatch at "));
        }
    }
}

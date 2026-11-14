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

import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.PreparedQuery;
import io.github.jbellis.jvector.util.BatchingExecutor;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;

/** The small host-side queue adapter shared by the two ASH hello-world programs. */
final class HelloASHQueuedSearch {
    private HelloASHQueuedSearch() {}

    // Inputs and prepared buffers belong to one request until its callback completes.
    private static final class Request {
        final VectorFloat<?> query;
        final PreparedQuery prepared;
        final CompletableFuture<SearchResult> completion = new CompletableFuture<>();
        SearchResult result;

        Request(VectorFloat<?> query, PreparedQuery prepared) {
            this.query = query.copy();
            this.prepared = prepared;
        }
    }

    static List<SearchResult> search(OnDiskGraphIndex graph, VectorSimilarityFunction similarity,
                                     List<VectorFloat<?>> queries, int searchCores) throws Exception {
        var preparer = graph.queryPreparer(similarity).orElseThrow();
        int maximumBatch = 8, admission = 8 * searchCores;

        // The application supplies execution threads. Each worker owns its searcher.
        // Close the queue first (drains callbacks), then searchers, then the executor.
        var executor = Executors.newFixedThreadPool(searchCores);
        try (var searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(graph));
             var queue = new BatchingExecutor<Request>(searchCores, admission, maximumBatch,
                     executor, new BatchingExecutor.Operation<Request>() {
                 public Object batchKey(Request request) {
                     return preparer; // Only queries for this same model can share preparation.
                 }

                 public void prepareBatch(List<Request> batch) {
                     // These small arrays keep the example simple; each call owns its arrays.
                     var inputs = new VectorFloat<?>[batch.size()];
                     var states = new PreparedQuery[batch.size()];
                     for (int i = 0; i < batch.size(); i++) {
                         inputs[i] = batch.get(i).query;
                         states[i] = batch.get(i).prepared;
                     }
                     if (batch.size() == 1) preparer.prepare(inputs[0], states[0]);
                     else preparer.prepareBatch(inputs, states, batch.size());
                 }

                 public void execute(Request request) {
                     var searcher = searchers.get();
                     var view = (ImmutableGraphIndex.ScoringView) searcher.getView();
                     request.result = searcher.search(view.scoreProviderFor(request.prepared, true),
                             10, 20, 0.0f, 0.0f, Bits.ALL);
                 }

                 public void complete(Request request, Throwable failure) {
                     if (failure == null) request.completion.complete(request.result);
                     else request.completion.completeExceptionally(failure);
                 }
             })) {
            var requests = new ArrayList<Request>();
            // Submit individual queries. The queue batches currently waiting work;
            // it never waits for a timer or requires the caller to assemble a batch.
            for (var query : queries) {
                var request = new Request(query, preparer.newState());
                if (!queue.trySubmit(request)) {
                    // Rejected requests remain caller-owned: reject or apply backpressure.
                    throw new RejectedExecutionException("Query admission is full");
                }
                requests.add(request);
            }
            // Waiting happens on the caller, outside the executor needed to run searches.
            // Futures preserve submission order even if callbacks arrive out of order.
            var results = new ArrayList<SearchResult>();
            for (var request : requests) results.add(request.completion.join());
            return results;
        } finally {
            executor.shutdown(); // Accepted work was drained by queue.close(), never discarded.
        }
    }
}

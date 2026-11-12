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

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;

import io.github.jbellis.jvector.graph.GraphBuildAccelerator;
import io.github.jbellis.jvector.graph.GraphBuildAccelerators;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/** Graph builds through the {@link GraphBuildAccelerator} selected by {@code -Djvector.graph.accelerator}, if any. */
final class AcceleratedBuild {

    private AcceleratedBuild() {
    }

    /**
     * Starts the graph build on the selected accelerator, in the background.
     *
     * @return the running build, or null when no accelerator is selected or it does not support the vectors
     */
    static CompletableFuture<Void> start(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity) {
        var accelerator = GraphBuildAccelerators.find();
        if (accelerator.isEmpty() || !accelerator.get().supports(builder, vectors, similarity)) {
            return null;
        }
        System.out.format("Building the graph with accelerator %s%n", accelerator.get().name());
        var thread = Executors.newSingleThreadExecutor();
        try {
            return CompletableFuture.runAsync(() -> accelerator.get().build(builder, vectors, similarity), thread);
        } finally {
            thread.shutdown();
        }
    }
}

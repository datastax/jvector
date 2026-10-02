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

import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/**
 * An {@link HnswIndexBuilder} that builds from a {@link RandomAccessVectorValues} and a
 * {@link VectorSimilarityFunction}. Returned by {@code Indexes.hnswBuilder(RandomAccessVectorValues,
 * VectorSimilarityFunction)}.
 * <p>
 * {@link #build()} derives the build score provider from the vectors according to
 * {@link #withCompressionType}: exact comparisons for {@link CompressionType#NONE}, or the vectors
 * are first compressed with product quantization ({@link CompressionType#PQ}, sized from
 * {@code GraphIndexBuilderConfig}) or binary quantization ({@link CompressionType#BQ}) and the graph
 * is built with the compressed scores.
 */
public class RavvHnswBuilder extends HnswIndexBuilder {
    private final RandomAccessVectorValues vectorValues;
    private final VectorSimilarityFunction similarityFunction;
    private CompressionType compressionType = CompressionType.NONE;

    /**
     * Creates a builder for a graph over {@code vectorValues}, compared with {@code similarityFunction}.
     */
    public RavvHnswBuilder(RandomAccessVectorValues vectorValues, VectorSimilarityFunction similarityFunction) {
        this.vectorValues = vectorValues;
        this.similarityFunction = similarityFunction;
    }

    @Override
    public HnswIndexBuilder withCompressionType(CompressionType compressionType) {
        this.compressionType = compressionType;
        return this;
    }

    @Override
    public PersistableGraphIndex buildAndPopulate() {
        return populateGraph(vectorValues);
    }

    @Override
    CompressionType compressionType() {
        return compressionType;
    }

    /**
     * {@inheritDoc}
     * <p>
     * With {@link CompressionType#PQ} or {@link CompressionType#BQ}, this first trains the
     * quantization on, and encodes, all of the vectors, using this builder's executors, which can take
     * significant time.
     *
     * @throws IllegalArgumentException if the compression type is not supported
     */
    @Override
    protected GraphIndexBuilder createGraphBuilder() {
        BuildScoreProvider scoreProvider = GraphIndexBuilder.buildScoreProvider(vectorValues, similarityFunction,
                compressionType, simdExecutor, parallelExecutor);
        return newGraphBuilder(scoreProvider, vectorValues.dimension());
    }

}

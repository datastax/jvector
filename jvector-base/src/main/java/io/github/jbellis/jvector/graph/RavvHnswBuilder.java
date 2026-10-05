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
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.Objects;

/**
 * An {@link HnswIndexBuilder} that builds from a {@link RandomAccessVectorValues} and a
 * {@link VectorSimilarityFunction}. Returned by {@code Indexes.hnswBuilder(RandomAccessVectorValues,
 * VectorSimilarityFunction)}, which is the supported way to obtain one; the constructor is public only
 * because {@code Indexes} is in another package.
 * <p>
 * {@link #build()} derives the build score provider from the vectors according to
 * {@link #withCompressionType}: exact comparisons for {@link CompressionType#NONE}, or the vectors
 * are first compressed with product quantization ({@link CompressionType#PQ}, with
 * {@link #withPqSubspaces} subspaces, by default one per 4 dimensions) or binary quantization
 * ({@link CompressionType#BQ}) and the graph is built with the compressed scores.
 */
public class RavvHnswBuilder extends HnswIndexBuilder {
    private final RandomAccessVectorValues vectorValues;
    private final VectorSimilarityFunction similarityFunction;
    private CompressionType compressionType = CompressionType.NONE;
    private int pqSubspaces;
    // Written once, under the base class's lock, before the volatile graphBuilder is published.
    private CompressedVectors compressedVectors;

    /**
     * Creates a builder for a graph over {@code vectorValues}, compared with {@code similarityFunction}.
     * Prefer {@code Indexes.hnswBuilder(vectorValues, similarityFunction)}, which returns the same
     * builder.
     */
    public RavvHnswBuilder(RandomAccessVectorValues vectorValues, VectorSimilarityFunction similarityFunction) {
        this.vectorValues = Objects.requireNonNull(vectorValues, "vectorValues");
        this.similarityFunction = Objects.requireNonNull(similarityFunction, "similarityFunction");
        this.pqSubspaces = Math.max(1, vectorValues.dimension() / 4);
    }

    @Override
    public HnswIndexBuilder withCompressionType(CompressionType compressionType) {
        if (ignoredAfterBuild("withCompressionType", compressionType)) {
            return this;
        }
        this.compressionType = compressionType;
        return this;
    }

    @Override
    public HnswIndexBuilder withPqSubspaces(int pqSubspaces) {
        int dimension = vectorValues.dimension();
        if (pqSubspaces < 1 || pqSubspaces > dimension) {
            throw new IllegalArgumentException(String.format(
                    "pqSubspaces must be between 1 and the vector dimension, %d (was %d)", dimension, pqSubspaces));
        }
        if (ignoredAfterBuild("withPqSubspaces", pqSubspaces)) {
            return this;
        }
        this.pqSubspaces = pqSubspaces;
        return this;
    }

    /** The number of PQ subspaces this builder will use. */
    int pqSubspaces() {
        return pqSubspaces;
    }

    @Override
    public PersistableGraphIndex buildAndPopulate() {
        return populateGraph(vectorValues);
    }

    /**
     * {@inheritDoc}
     * <p>
     * For {@link CompressionType#PQ} these are {@link PQVectors}, whose {@code getCompressor()} is the
     * trained {@code ProductQuantization}; for {@link CompressionType#BQ}, {@code BQVectors}.
     *
     * @throws IllegalStateException if the graph hasn't been built yet and compression is enabled
     */
    @Override
    public CompressedVectors getCompressedVectors() {
        if (graphBuilder == null) {
            if (compressionType == CompressionType.NONE) {
                return null;
            }
            throw new IllegalStateException("The vectors are compressed when the graph is built; "
                    + "call build(), buildAndPopulate() or populateGraph() first");
        }
        return compressedVectors;
    }

    @Override
    int dimension() {
        return vectorValues.dimension();
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
        compressedVectors = GraphIndexBuilder.compress(vectorValues, compressionType, pqSubspaces, simdExecutor, parallelExecutor);
        BuildScoreProvider scoreProvider = GraphIndexBuilder.buildScoreProvider(vectorValues, similarityFunction,
                compressedVectors);
        return newGraphBuilder(scoreProvider, vectorValues.dimension());
    }

}

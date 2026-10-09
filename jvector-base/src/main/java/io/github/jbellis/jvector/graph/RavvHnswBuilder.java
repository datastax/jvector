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
import io.github.jbellis.jvector.index.HnswRecipe;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer;
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
 * are first compressed with product quantization ({@link CompressionType#PQ}, configured with
 * {@link #withPqSubspaces}, {@link #withPqGlobalCentering} and {@link #withPqAnisotropicThreshold},
 * whose defaults match Cassandra's), binary quantization ({@link CompressionType#BQ}) or asymmetric
 * hashing ({@link CompressionType#ASH}, configured with {@link #withAshProjectedDimensions} and
 * {@link #withAshBitsPerDimension}, and for {@link VectorSimilarityFunction#DOT_PRODUCT} only), and the
 * graph is built with the compressed scores.
 */
public class RavvHnswBuilder extends HnswIndexBuilder {
    private final RandomAccessVectorValues vectorValues;
    private final VectorSimilarityFunction similarityFunction;
    private CompressionType compressionType = CompressionType.NONE;
    private int pqSubspaces;
    private boolean pqGlobalCentering = false;
    private float pqAnisotropicThreshold = KMeansPlusPlusClusterer.UNWEIGHTED;
    private int ashProjectedDimensions;
    private int ashBitsPerDimension = AsymmetricHashing.DEFAULT_BITS_PER_DIMENSION;
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
        this.pqSubspaces = defaultPqSubspaces(vectorValues.dimension());
        this.ashProjectedDimensions = vectorValues.dimension();
    }

    @Override
    public HnswIndexBuilder withCompressionType(CompressionType compressionType) {
        Objects.requireNonNull(compressionType, "compressionType");
        if (compressionType == CompressionType.ASH && similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new IllegalArgumentException("CompressionType.ASH supports DOT_PRODUCT similarity only, but this builder "
                    + "compares vectors with " + similarityFunction + "; for unit-length vectors DOT_PRODUCT ranks "
                    + "neighbors the same way COSINE and EUCLIDEAN do");
        }
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
    public HnswIndexBuilder withPqGlobalCentering(boolean pqGlobalCentering) {
        if (ignoredAfterBuild("withPqGlobalCentering", pqGlobalCentering)) {
            return this;
        }
        this.pqGlobalCentering = pqGlobalCentering;
        return this;
    }

    @Override
    public HnswIndexBuilder withPqAnisotropicThreshold(float pqAnisotropicThreshold) {
        if (Float.isNaN(pqAnisotropicThreshold) || pqAnisotropicThreshold < -1.0f || pqAnisotropicThreshold >= 1.0f) {
            throw new IllegalArgumentException(
                    "pqAnisotropicThreshold must be in [-1.0, 1.0) (was " + pqAnisotropicThreshold + ")");
        }
        if (ignoredAfterBuild("withPqAnisotropicThreshold", pqAnisotropicThreshold)) {
            return this;
        }
        this.pqAnisotropicThreshold = pqAnisotropicThreshold;
        return this;
    }

    @Override
    public HnswIndexBuilder withAshProjectedDimensions(int projectedDimensions) {
        int dimension = vectorValues.dimension();
        if (projectedDimensions < 1 || projectedDimensions > dimension) {
            throw new IllegalArgumentException(String.format(
                    "ashProjectedDimensions must be between 1 and the vector dimension, %d (was %d)", dimension, projectedDimensions));
        }
        if (ignoredAfterBuild("withAshProjectedDimensions", projectedDimensions)) {
            return this;
        }
        this.ashProjectedDimensions = projectedDimensions;
        return this;
    }

    @Override
    public HnswIndexBuilder withAshBitsPerDimension(int bitsPerDimension) {
        if (bitsPerDimension < 1 || bitsPerDimension > 9) {
            throw new IllegalArgumentException("ashBitsPerDimension must be between 1 and 9 (was " + bitsPerDimension + ")");
        }
        if (ignoredAfterBuild("withAshBitsPerDimension", bitsPerDimension)) {
            return this;
        }
        this.ashBitsPerDimension = bitsPerDimension;
        return this;
    }

    /** The number of ASH projected dimensions this builder will use. */
    int ashProjectedDimensions() {
        return ashProjectedDimensions;
    }

    /** The number of ASH bits per projected dimension this builder will use. */
    int ashBitsPerDimension() {
        return ashBitsPerDimension;
    }

    boolean pqGlobalCentering() {
        return pqGlobalCentering;
    }

    float pqAnisotropicThreshold() {
        return pqAnisotropicThreshold;
    }

    /**
     * The default number of PQ subspaces (bytes per code) for vectors of {@code dimension}: Cassandra's
     * rule for vectors from an unknown model ({@code VectorSourceModel.defaultPQBytesFor}), which
     * OpenSearch uses too. It keeps the code size strictly increasing with the dimension.
     */
    static int defaultPqSubspaces(int dimension) {
        if (dimension <= 32) {
            return Math.max(1, dimension);
        } else if (dimension <= 64) {
            return 32;
        } else if (dimension <= 200) {
            return (int) (dimension * 0.5);
        } else if (dimension <= 400) {
            return 100;
        } else if (dimension <= 768) {
            return (int) (dimension * 0.25);
        } else if (dimension <= 1536) {
            return 192;
        } else {
            return (int) (dimension * 0.125);
        }
    }

    @Override
    public PersistableGraphIndex buildAndPopulate() {
        return populateGraph(vectorValues);
    }

    /**
     * {@inheritDoc}
     * <p>
     * This builder scores every node, by ordinal, against the vectors it was created with (or their
     * compressed codes), so {@code ravv} must hold those same vectors at the same ordinals: usually the
     * same {@link RandomAccessVectorValues}, which {@link #buildAndPopulate()} passes for you. Different
     * vectors would build a graph scored against the wrong data.
     *
     * @throws IllegalArgumentException if {@code ravv}'s size or dimension differs from the vectors this
     *         builder was created with
     */
    @Override
    public PersistableGraphIndex populateGraph(RandomAccessVectorValues ravv) {
        Objects.requireNonNull(ravv, "ravv");
        if (ravv != vectorValues && ravv.size() != vectorValues.size()) {
            throw new IllegalArgumentException(String.format(
                    "populateGraph() was given %d vectors, but this builder was created with %d; it scores nodes "
                            + "against the vectors it was created with, so it must be populated with the same vectors "
                            + "(buildAndPopulate() does that)", ravv.size(), vectorValues.size()));
        }
        return super.populateGraph(ravv);
    }

    /**
     * {@inheritDoc}
     * <p>
     * For {@link CompressionType#PQ} these are {@link PQVectors}, whose {@code getCompressor()} is the
     * trained {@code ProductQuantization}; for {@link CompressionType#BQ}, {@code BQVectors}; for
     * {@link CompressionType#ASH}, {@link ASHVectors}, whose {@code getCompressor()} is the trained
     * {@code AsymmetricHashing}.
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
    boolean scoresExactly() {
        return compressionType == CompressionType.NONE;
    }

    @Override
    void applyCompressionRecipe(HnswRecipe recipe) {
        if (recipe.has(HnswRecipe.Param.PQ_SUBSPACES)) {
            int subspaces = recipe.get(HnswRecipe.Param.PQ_SUBSPACES);
            if (subspaces == 0) {
                pqSubspaces = defaultPqSubspaces(vectorValues.dimension());
            } else {
                withPqSubspaces(subspaces);
            }
        }
        if (recipe.has(HnswRecipe.Param.PQ_GLOBAL_CENTERING)) {
            pqGlobalCentering = recipe.get(HnswRecipe.Param.PQ_GLOBAL_CENTERING);
        }
        if (recipe.has(HnswRecipe.Param.PQ_ANISOTROPIC_THRESHOLD)) {
            withPqAnisotropicThreshold(recipe.get(HnswRecipe.Param.PQ_ANISOTROPIC_THRESHOLD));
        }
        if (recipe.has(HnswRecipe.Param.ASH_PROJECTED_DIMENSIONS)) {
            int dimensions = recipe.get(HnswRecipe.Param.ASH_PROJECTED_DIMENSIONS);
            if (dimensions == 0) {
                ashProjectedDimensions = vectorValues.dimension();
            } else {
                withAshProjectedDimensions(dimensions);
            }
        }
        if (recipe.has(HnswRecipe.Param.ASH_BITS_PER_DIMENSION)) {
            withAshBitsPerDimension(recipe.get(HnswRecipe.Param.ASH_BITS_PER_DIMENSION));
        }
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
     * With {@link CompressionType#PQ}, {@link CompressionType#BQ} or {@link CompressionType#ASH}, this first trains the
     * quantization on, and encodes, all of the vectors, using this builder's executors, which can take
     * significant time.
     *
     * @throws IllegalArgumentException if the compression type is not supported
     */
    @Override
    protected GraphIndexBuilder createGraphBuilder() {
        compressedVectors = GraphIndexBuilder.compress(vectorValues, compressionType, pqSubspaces,
                pqGlobalCentering, pqAnisotropicThreshold, ashProjectedDimensions, ashBitsPerDimension,
                buildExecutor, maintenanceExecutor);
        BuildScoreProvider scoreProvider = GraphIndexBuilder.buildScoreProvider(vectorValues, similarityFunction,
                compressedVectors);
        return newGraphBuilder(scoreProvider, vectorValues.dimension());
    }

}

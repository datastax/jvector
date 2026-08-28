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

package io.github.jbellis.jvector.index.graph;

import io.github.jbellis.jvector.index.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * Fluent, validating builder for {@link GraphIndexBuilder}.
 * <p>
 * {@link GraphIndexBuilder} itself keeps its telescoping constructors and {@code final} fields
 * unchanged; this class only collects configuration via chainable {@code withXxx} methods,
 * applies the same default values that {@link GraphIndexBuilder}'s convenience constructors do,
 * and validates that everything required is present before delegating to the appropriate
 * {@link GraphIndexBuilder} constructor in {@link #build()}.
 * <p>
 * Two mutually exclusive ways to supply scoring are accepted, mirroring
 * {@link GraphIndexBuilder}'s own constructor overloads:
 * <ul>
 *     <li>{@link #withVectorValues} + {@link #withSimilarityFunction} (dimension is derived
 *     automatically from the vector values), or</li>
 *     <li>{@link #withScoreProvider} directly (in which case {@link #withDimension} is required).</li>
 * </ul>
 * Similarly, two mutually exclusive ways to supply the graph shape are accepted:
 * <ul>
 *     <li>{@link #withMaxDegree}/{@link #withMaxDegrees} + {@link #withAddHierarchy}, to build a
 *     new graph from scratch, or</li>
 *     <li>{@link #withExistingGraph}, to continue building on top of an already-loaded
 *     {@link MutableGraphIndex} (see {@link GraphIndexBuilder}'s {@code @Experimental} constructor
 *     of the same shape).</li>
 * </ul>
 */
public class HnswIndexBuilder {
    private BuildScoreProvider scoreProvider;
    private RandomAccessVectorValues vectorValues;
    private VectorSimilarityFunction similarityFunction;
    private Integer dimension;
    private List<Integer> maxDegrees;
    private Integer beamWidth;
    private Float neighborOverflow;
    private Float alpha;
    private Boolean addHierarchy;
    private MutableGraphIndex existingGraph;

    // Defaults matching GraphIndexBuilder's convenience constructors.
    private boolean refineFinalGraph = true;
    private ForkJoinPool simdExecutor = PhysicalCoreExecutor.pool();
    private ForkJoinPool parallelExecutor = ForkJoinPool.commonPool();

    public HnswIndexBuilder() {
    }

    /**
     * Supplies the score provider directly. Mutually exclusive with {@link #withVectorValues}/
     * {@link #withSimilarityFunction}. Requires {@link #withDimension} to also be set.
     */
    public HnswIndexBuilder withScoreProvider(BuildScoreProvider scoreProvider) {
        this.scoreProvider = scoreProvider;
        return this;
    }

    /**
     * Supplies the vectors to build the graph from. Must be paired with
     * {@link #withSimilarityFunction}; dimension is derived from this automatically.
     * Mutually exclusive with {@link #withScoreProvider}.
     */
    public HnswIndexBuilder withVectorValues(RandomAccessVectorValues vectorValues) {
        this.vectorValues = vectorValues;
        return this;
    }

    /** The similarity metric to use during construction. Paired with {@link #withVectorValues}. */
    public HnswIndexBuilder withSimilarityFunction(VectorSimilarityFunction similarityFunction) {
        this.similarityFunction = similarityFunction;
        return this;
    }

    /**
     * The vector dimension. Only needed (and required) when using {@link #withScoreProvider}
     * directly; when using {@link #withVectorValues}, the dimension is derived from it and does
     * not need to be set here.
     */
    public HnswIndexBuilder withDimension(int dimension) {
        this.dimension = dimension;
        return this;
    }

    /** Sets a single max degree for all layers. Equivalent to {@code withMaxDegrees(List.of(maxDegree))}. */
    public HnswIndexBuilder withMaxDegree(int maxDegree) {
        this.maxDegrees = List.of(maxDegree);
        return this;
    }

    /**
     * The maximum number of connections a node can have in each layer; if fewer entries are
     * specified than the number of layers, the last entry is used for all remaining layers.
     */
    public HnswIndexBuilder withMaxDegrees(List<Integer> maxDegrees) {
        this.maxDegrees = maxDegrees;
        return this;
    }

    /** The size of the beam search to use when finding nearest neighbors. */
    public HnswIndexBuilder withBeamWidth(int beamWidth) {
        this.beamWidth = beamWidth;
        return this;
    }

    /**
     * The ratio of extra neighbors to allow temporarily when inserting a node. Larger values
     * will build more efficiently, but use more memory.
     */
    public HnswIndexBuilder withNeighborOverflow(float neighborOverflow) {
        this.neighborOverflow = neighborOverflow;
        return this;
    }

    /**
     * How aggressive pruning diverse neighbors should be. Set alpha &gt; 1.0 to allow longer
     * edges. If alpha = 1.0 then the equivalent of the lowest level of an HNSW graph will be
     * created, which is usually not what you want.
     */
    public HnswIndexBuilder withAlpha(float alpha) {
        this.alpha = alpha;
        return this;
    }

    /**
     * Whether to add an HNSW-style hierarchy on top of the Vamana index. Not used (and not
     * required) when building on top of an {@link #withExistingGraph existing graph}, since its
     * hierarchy is already fixed.
     */
    public HnswIndexBuilder withAddHierarchy(boolean addHierarchy) {
        this.addHierarchy = addHierarchy;
        return this;
    }

    /**
     * Whether to do a second pass over each node in the graph to refine its connections.
     * Defaults to {@code true}, matching {@link GraphIndexBuilder}'s convenience constructors.
     */
    public HnswIndexBuilder withRefineFinalGraph(boolean refineFinalGraph) {
        this.refineFinalGraph = refineFinalGraph;
        return this;
    }

    /**
     * ForkJoinPool instance for SIMD operations. Defaults to {@link PhysicalCoreExecutor#pool()},
     * matching {@link GraphIndexBuilder}'s convenience constructors.
     */
    public HnswIndexBuilder withSimdExecutor(ForkJoinPool simdExecutor) {
        this.simdExecutor = simdExecutor;
        return this;
    }

    /**
     * ForkJoinPool instance for parallel stream operations. Defaults to
     * {@link ForkJoinPool#commonPool()}, matching {@link GraphIndexBuilder}'s convenience
     * constructors.
     */
    public HnswIndexBuilder withParallelExecutor(ForkJoinPool parallelExecutor) {
        this.parallelExecutor = parallelExecutor;
        return this;
    }

    /**
     * Continue building on top of an already-loaded {@link MutableGraphIndex} instead of creating
     * a new one. Mutually exclusive with {@link #withMaxDegree}/{@link #withMaxDegrees} and
     * {@link #withAddHierarchy}, which are ignored (and not required) when this is set, since the
     * existing graph already carries that information.
     */
    public HnswIndexBuilder withExistingGraph(MutableGraphIndex existingGraph) {
        this.existingGraph = existingGraph;
        return this;
    }

    /**
     * Validates that all required configuration has been supplied, and constructs the
     * corresponding {@link GraphIndexBuilder}.
     *
     * @throws IllegalStateException if a mutually-exclusive pair was over-specified, or if a
     * required value is missing; the message names every missing/conflicting value at once.
     */
    public GraphIndexBuilder build() {
        if (scoreProvider != null && (vectorValues != null || similarityFunction != null)) {
            throw new IllegalStateException(
                    "Set either withScoreProvider() or withVectorValues()+withSimilarityFunction(), not both");
        }

        List<String> missing = new ArrayList<>();
        BuildScoreProvider resolvedScoreProvider = scoreProvider;
        Integer resolvedDimension = dimension;

        if (resolvedScoreProvider == null) {
            if (vectorValues == null) {
                missing.add("vectorValues (or scoreProvider)");
            }
            if (similarityFunction == null) {
                missing.add("similarityFunction (or scoreProvider)");
            }
            if (vectorValues != null && similarityFunction != null) {
                resolvedScoreProvider = BuildScoreProvider.randomAccessScoreProvider(vectorValues, similarityFunction);
                int derivedDimension = vectorValues.dimension();
                if (resolvedDimension != null && resolvedDimension != derivedDimension) {
                    throw new IllegalStateException(String.format(
                            "dimension(%d) does not match vectorValues.dimension()=%d; " +
                            "omit withDimension() when using withVectorValues()",
                            resolvedDimension, derivedDimension));
                }
                resolvedDimension = derivedDimension;
            }
        } else if (resolvedDimension == null) {
            missing.add("dimension (required when using scoreProvider() directly)");
        }

        if (existingGraph == null) {
            if (maxDegrees == null) {
                missing.add("maxDegree/maxDegrees (or existingGraph)");
            }
            if (addHierarchy == null) {
                missing.add("addHierarchy (or existingGraph)");
            }
        }
        if (beamWidth == null) {
            missing.add("beamWidth");
        }
        if (neighborOverflow == null) {
            missing.add("neighborOverflow");
        }
        if (alpha == null) {
            missing.add("alpha");
        }

        if (!missing.isEmpty()) {
            throw new IllegalStateException(
                    "Cannot build GraphIndexBuilder, missing required value(s): " + String.join(", ", missing));
        }

        if (existingGraph != null) {
            return new GraphIndexBuilder(resolvedScoreProvider,
                    resolvedDimension,
                    existingGraph,
                    beamWidth,
                    neighborOverflow,
                    alpha,
                    refineFinalGraph,
                    simdExecutor,
                    parallelExecutor);
        }

        return new GraphIndexBuilder(resolvedScoreProvider,
                resolvedDimension,
                maxDegrees,
                beamWidth,
                neighborOverflow,
                alpha,
                addHierarchy,
                refineFinalGraph,
                simdExecutor,
                parallelExecutor);
    }
}

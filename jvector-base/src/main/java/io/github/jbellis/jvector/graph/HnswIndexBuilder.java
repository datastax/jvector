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
import io.github.jbellis.jvector.index.IndexBuilderValidation;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Fluent, validating builder for {@link GraphIndexBuilder}.
 * <p>
 * {@link GraphIndexBuilder} itself keeps its telescoping constructors and {@code final} fields
 * unchanged; this class only collects configuration via chainable {@code withXxx} methods,
 * applies the same default values that {@link GraphIndexBuilder}'s convenience constructors do,
 * and validates that everything required is present before delegating to the appropriate
 * {@link GraphIndexBuilder} constructor in {@link #build()}.
 * <p>
 * {@link #withVectorValues} is always required: it is what drives the actual build loop (it
 * supplies both the number of nodes to insert and the raw vector for each one), regardless of
 * which of the two mutually exclusive ways of supplying scoring is used, mirroring
 * {@link GraphIndexBuilder}'s own constructor overloads:
 * <ul>
 *     <li>{@link #withVectorValues} + {@link #withSimilarityFunction}, in which case a score
 *     provider performing exact comparisons is derived automatically (dimension is also derived
 *     automatically), or</li>
 *     <li>{@link #withVectorValues} + {@link #withScoreProvider}, to score with something other
 *     than exact comparison against the raw vectors (e.g. a PQ/BQ-compressed provider); dimension
 *     is still derived from {@link #withVectorValues} unless overridden with {@link #withDimension}
 *     for cross-validation.</li>
 * </ul>
 * Similarly, two mutually exclusive ways to supply the graph shape are accepted:
 * <ul>
 *     <li>{@link #withMaxDegree}/{@link #withMaxDegrees} + {@link #withAddHierarchy}, to build a
 *     new graph from scratch, or</li>
 *     <li>{@link #withExistingGraph}, to continue building on top of an already-loaded
 *     {@link MutableGraphIndex} (see {@link GraphIndexBuilder}'s {@code @Experimental} constructor
 *     of the same shape). In this case {@link #withVectorValues} must be a superset containing an
 *     entry for every ordinal already present in the existing graph (at the same ordinals) plus the
 *     new vectors to append; new nodes are inserted starting at the existing graph's
 *     {@link GraphIndex#getIdUpperBound()}.</li>
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
     * Supplies the score provider directly, for scoring that is not a plain exact comparison of
     * the raw vectors (e.g. a PQ/BQ-compressed provider). Mutually exclusive with
     * {@link #withSimilarityFunction}. {@link #withVectorValues} is still required alongside this:
     * it is what is actually iterated to drive node insertion, independent of how those nodes are
     * scored.
     */
    public HnswIndexBuilder withScoreProvider(BuildScoreProvider scoreProvider) {
        this.scoreProvider = scoreProvider;
        return this;
    }

    /**
     * Supplies the vectors to build the graph from. Always required: this is what is iterated to
     * drive node insertion (both the node count and the vector for each node), regardless of which
     * scoring option is used. Dimension is derived from this automatically.
     * <p>
     * Pair with {@link #withSimilarityFunction} for a score provider performing exact comparisons
     * against these vectors, or with {@link #withScoreProvider} to score some other way (in which
     * case these vectors are only used to drive insertion, not to compute scores).
     */
    public HnswIndexBuilder withVectorValues(RandomAccessVectorValues vectorValues) {
        this.vectorValues = vectorValues;
        return this;
    }

    /**
     * The similarity metric to use during construction, used to derive a score provider that
     * performs exact comparisons against {@link #withVectorValues}. Mutually exclusive with
     * {@link #withScoreProvider}.
     */
    public HnswIndexBuilder withSimilarityFunction(VectorSimilarityFunction similarityFunction) {
        this.similarityFunction = similarityFunction;
        return this;
    }

    /**
     * The vector dimension. Optional: it is always derived from {@link #withVectorValues}. Setting
     * this is only useful as a cross-check — {@link #build()} throws if it disagrees with
     * {@code withVectorValues().dimension()}.
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
     * <p>
     * The nodes already in {@code existingGraph} are <b>not</b> re-inserted. Instead,
     * {@link #withVectorValues} must be a superset RAVV: ordinals {@code [0, existingGraph.getIdUpperBound())}
     * must line up with the vectors already in the graph, and the remaining ordinals
     * {@code [existingGraph.getIdUpperBound(), vectorValues.size())} are the new vectors that get
     * appended.
     */
    public HnswIndexBuilder withExistingGraph(MutableGraphIndex existingGraph) {
        this.existingGraph = existingGraph;
        return this;
    }

    /**
     * Pre-sets this builder's fixed fields to the given recipe's recommended values, leaving the
     * recipe's free parameters (e.g. {@code dimensions}) for the caller to still supply.
     * <p>
     * Scaffolding only: the recipes' actual fixed-value formulas haven't been decided yet, so
     * every {@link HnswRecipe} currently refuses here rather than guess at numbers.
     *
     * @throws UnsupportedOperationException always, until a recipe's values are defined
     */
    public HnswIndexBuilder applyRecipe(HnswRecipe recipe) {
        throw new UnsupportedOperationException(
                "HnswRecipe." + recipe + " has no defined values yet");
    }

    /**
     * Validates that all required configuration has been supplied, builds the corresponding
     * {@link GraphIndexBuilder}, and drives it to completion.
     *
     * @throws IllegalStateException if a mutually-exclusive pair was over-specified, or if a
     * required value is missing; the message names every missing/conflicting value at once.
     */
    public GraphIndex build() {
        if (scoreProvider != null && similarityFunction != null) {
            throw new IllegalStateException(
                    "Set either withScoreProvider() or withSimilarityFunction(), not both");
        }

        new IndexBuilderValidation()
                .require("vectorValues", vectorValues)
                .requireCondition("similarityFunction (or scoreProvider)",
                        scoreProvider != null || similarityFunction != null)
                .requireCondition("maxDegree/maxDegrees (or existingGraph)",
                        existingGraph != null || maxDegrees != null)
                .requireCondition("addHierarchy (or existingGraph)",
                        existingGraph != null || addHierarchy != null)
                .require("beamWidth", beamWidth)
                .require("neighborOverflow", neighborOverflow)
                .require("alpha", alpha)
                .throwIfAny("Cannot build GraphIndexBuilder");

        int resolvedDimension = vectorValues.dimension();
        if (dimension != null && dimension != resolvedDimension) {
            throw new IllegalStateException(String.format(
                    "dimension(%d) does not match vectorValues.dimension()=%d; " +
                    "omit withDimension(), it is derived automatically from vectorValues",
                    dimension, resolvedDimension));
        }

        BuildScoreProvider resolvedScoreProvider = scoreProvider != null
                ? scoreProvider
                : BuildScoreProvider.randomAccessScoreProvider(vectorValues, similarityFunction);

        if (existingGraph != null) {
            int startingNodeOffset = existingGraph.getIdUpperBound();
            int size = vectorValues.size();
            if (size < startingNodeOffset) {
                throw new IllegalStateException(String.format(
                        "vectorValues.size()=%d is smaller than existingGraph.getIdUpperBound()=%d; " +
                        "when using withExistingGraph(), vectorValues must be a superset containing an " +
                        "entry for every node ordinal already in the existing graph, in addition to the " +
                        "new vectors being appended",
                        size, startingNodeOffset));
            }

            GraphIndexBuilder builder = new GraphIndexBuilder(resolvedScoreProvider,
                    resolvedDimension,
                    existingGraph,
                    beamWidth,
                    neighborOverflow,
                    alpha,
                    refineFinalGraph,
                    simdExecutor,
                    parallelExecutor);

            var vv = vectorValues.threadLocalSupplier();
            simdExecutor.submit(() -> IntStream.range(startingNodeOffset, size).parallel().forEach(node -> {
                builder.addGraphNode(node, vv.get().getVector(node));
            })).join();
            builder.cleanup();
            return builder.getGraph();
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
                parallelExecutor).build(vectorValues);
    }
}

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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.index.HnswRecipe;
import io.github.jbellis.jvector.index.IndexBuilderValidation;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * Fluent, validating builder for graph/HNSW indexes, with two ways to finish:
 * <ul>
 *     <li>{@link #build()} inserts every vector in {@link #withVectorValues} in one parallel batch
 *     and returns the finished {@link GraphIndex}.</li>
 *     <li>{@link #buildMutable()} returns an empty (or {@link #withExistingGraph existing}) graph
 *     wrapped in a {@link MutableHnswIndex}, which the caller adds nodes to one at a time,
 *     concurrently, while searching it &mdash; and can mark nodes deleted and rescore mid-build.</li>
 * </ul>
 * {@link GraphIndexBuilder} itself keeps its telescoping constructors and {@code final} fields
 * unchanged; this class only collects configuration via chainable {@code withXxx} methods,
 * applies the same default values that {@link GraphIndexBuilder}'s convenience constructors do,
 * and validates that everything required is present before delegating to the appropriate
 * {@link GraphIndexBuilder} constructor.
 * <p>
 * Scoring is supplied in one of two mutually exclusive ways, mirroring {@link GraphIndexBuilder}'s
 * own constructor overloads:
 * <ul>
 *     <li>{@link #withVectorValues} + {@link #withSimilarityFunction}, in which case a score
 *     provider performing exact comparisons is derived automatically (dimension is also derived
 *     automatically), or</li>
 *     <li>{@link #withScoreProvider}, to score with something other than exact comparison against
 *     the raw vectors (e.g. a PQ/BQ-compressed provider). Dimension is derived from
 *     {@link #withVectorValues} if set, otherwise it must be given with {@link #withDimension}.</li>
 * </ul>
 * {@link #build()} always requires {@link #withVectorValues}, because it is what drives the batch
 * insert loop (both the number of nodes and the vector for each one). {@link #buildMutable()} only
 * needs it for {@link #withSimilarityFunction}; with {@link #withScoreProvider} the caller supplies
 * each vector to {@link MutableHnswIndex#addNode} instead.
 * Similarly, two mutually exclusive ways to supply the graph shape are accepted:
 * <ul>
 *     <li>{@link #withMaxDegree}/{@link #withMaxDegrees} + {@link #withAddHierarchy}, to build a
 *     new graph from scratch, or</li>
 *     <li>{@link #withExistingGraph}, to continue building on top of an existing
 *     {@link OnHeapGraphIndex}, typically one reloaded with {@code OnHeapGraphIndex.load} (see
 *     {@link GraphIndexBuilder}'s {@code @Experimental} constructor of the same shape). In this case {@link #withVectorValues} must be a superset containing an
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
     * {@link #withSimilarityFunction}. {@link #build()} still requires {@link #withVectorValues}
     * alongside this, since it is what is iterated to drive node insertion, independent of how those
     * nodes are scored. {@link #buildMutable()} does not: the caller passes each vector to
     * {@link MutableHnswIndex#addNode}, and {@link #withDimension} supplies the dimension.
     */
    public HnswIndexBuilder withScoreProvider(BuildScoreProvider scoreProvider) {
        this.scoreProvider = scoreProvider;
        return this;
    }

    /**
     * Supplies the vectors to build the graph from. Required by {@link #build()}, which iterates it to
     * drive node insertion (both the node count and the vector for each node), regardless of which
     * scoring option is used. Required by {@link #buildMutable()} only together with
     * {@link #withSimilarityFunction}. Dimension is derived from this automatically.
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
     * The vector dimension. Required only by {@link #buildMutable()} with {@link #withScoreProvider}
     * and no {@link #withVectorValues}; otherwise it is derived from {@link #withVectorValues}, and
     * setting it is only a cross-check &mdash; building throws if it disagrees with
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
     * Whether to add an HNSW-style hierarchy on top of the Vamana index. Required when building a
     * new graph; must not be set together with {@link #withExistingGraph}, whose hierarchy is
     * already fixed.
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
     * Continue building on top of an existing {@link OnHeapGraphIndex} instead of creating
     * a new one. Mutually exclusive with {@link #withMaxDegree}/{@link #withMaxDegrees} and
     * {@link #withAddHierarchy}: the existing graph already carries that information, so setting
     * them as well is reported as a conflict.
     * <p>
     * The nodes already in {@code existingGraph} are <b>not</b> re-inserted, and the score provider
     * must cover their ordinals as well as the new ones. With {@link #build()},
     * {@link #withVectorValues} must be a superset RAVV: ordinals {@code [0, existingGraph.getIdUpperBound())}
     * must line up with the vectors already in the graph, and the remaining ordinals
     * {@code [existingGraph.getIdUpperBound(), vectorValues.size())} are the new vectors that get
     * appended. With {@link #buildMutable()}, the caller adds the new nodes itself, typically starting
     * at {@code existingGraph.getIdUpperBound()}.
     * <p>
     * The existing graph keeps the {@code DiversityProvider} it was created with, which prunes each new
     * node's neighbors: for a graph loaded with {@code OnHeapGraphIndex.load}, the one passed to
     * {@code load}; for a graph built in this process, one derived from the score provider it was built
     * with. Either way, that provider must be able to score the new ordinals too, not only this
     * builder's score provider.
     */
    public HnswIndexBuilder withExistingGraph(OnHeapGraphIndex existingGraph) {
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
    @Experimental
    public HnswIndexBuilder applyRecipe(HnswRecipe recipe) {
        throw new UnsupportedOperationException(
                "HnswRecipe." + recipe + " has no defined values yet");
    }

    /**
     * Validates that all required configuration has been supplied, builds the corresponding
     * {@link GraphIndexBuilder}, and drives it to completion: every vector in
     * {@link #withVectorValues} is inserted in parallel (on the {@link #withSimdExecutor SIMD
     * executor}), then the graph is cleaned up. With {@link #withExistingGraph}, only ordinals from
     * the existing graph's {@link GraphIndex#getIdUpperBound()} onwards are inserted.
     * <p>
     * Returns a {@link PersistableGraphIndex}, so the result can be written to disk with its
     * {@code getWriterBuilder}/{@code getParallelWriterBuilder} accessors without a cast. It is still a
     * {@link GraphIndex}, and assigning it to one is fine when persistence isn't needed.
     *
     * @throws IllegalStateException if any value is missing, out of range, or in conflict with
     * another setting; the message names every problem at once.
     */
    public PersistableGraphIndex build() {
        validate(true);
        int from = existingGraph == null ? 0 : existingGraph.getIdUpperBound();
        try (MutableHnswIndex index = newMutableIndex()) {
            index.addAllAndCleanup(vectorValues, from, simdExecutor);
            return index.graph();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Validates that all required configuration has been supplied and returns a
     * {@link MutableHnswIndex} for building the graph incrementally: the graph starts empty (or as
     * {@link #withExistingGraph the existing graph}), and the caller inserts nodes with
     * {@link MutableHnswIndex#addNode}, then calls {@link MutableHnswIndex#cleanup()} before writing it.
     * Nothing from {@link #withVectorValues} is inserted automatically.
     * <p>
     * Requires either {@link #withVectorValues} + {@link #withSimilarityFunction}, or
     * {@link #withScoreProvider} plus a dimension (from {@link #withVectorValues} or
     * {@link #withDimension}). Either way, as with {@link GraphIndexBuilder}, the score provider
     * must be able to score each ordinal by the time it is added; see
     * {@link MutableHnswIndex#addNode}.
     *
     * @throws IllegalStateException if any value is missing, out of range, or in conflict with
     * another setting; the message names every problem at once.
     */
    public MutableHnswIndex buildMutable() {
        validate(false);
        return newMutableIndex();
    }

    /**
     * Checks the whole configuration and reports every problem in one {@link IllegalStateException}:
     * missing values, conflicting settings, and out-of-range values (the same ranges
     * {@link GraphIndexBuilder}'s constructors enforce, so they never fail on a validated
     * configuration).
     */
    private void validate(boolean batch) {
        new IndexBuilderValidation()
                .requireCondition(batch ? "vectorValues" : "vectorValues (required with similarityFunction)",
                        vectorValues != null || (!batch && similarityFunction == null))
                .requireCondition("similarityFunction (or scoreProvider)",
                        scoreProvider != null || similarityFunction != null)
                .requireCondition("dimension (or vectorValues)",
                        batch || vectorValues != null || dimension != null || scoreProvider == null)
                .requireCondition("maxDegree/maxDegrees (or existingGraph)",
                        existingGraph != null || maxDegrees != null)
                .requireCondition("addHierarchy (or existingGraph)",
                        existingGraph != null || addHierarchy != null)
                .require("beamWidth", beamWidth)
                .require("neighborOverflow", neighborOverflow)
                .require("alpha", alpha)

                // conflicting settings
                .check(scoreProvider == null || similarityFunction == null,
                        "Set either withScoreProvider() or withSimilarityFunction(), not both")
                .check(existingGraph == null || maxDegrees == null,
                        "withExistingGraph() takes its max degrees from the existing graph; "
                                + "don't also set withMaxDegree()/withMaxDegrees()")
                .check(existingGraph == null || addHierarchy == null,
                        "withExistingGraph() takes its hierarchy from the existing graph; "
                                + "don't also set withAddHierarchy()")

                // out-of-range values (NaN fails the float checks too)
                .check(beamWidth == null || beamWidth > 0,
                        "beamWidth must be positive (was " + beamWidth + ")")
                .check(neighborOverflow == null || neighborOverflow >= 1.0f,
                        "neighborOverflow must be >= 1.0 (was " + neighborOverflow + ")")
                .check(alpha == null || alpha > 0,
                        "alpha must be positive (was " + alpha + ")")
                .check(maxDegrees == null || (!maxDegrees.isEmpty() && maxDegrees.stream().allMatch(d -> d != null && d > 0)),
                        "maxDegrees must be non-empty and positive (was " + maxDegrees + ")")
                .check(maxDegrees == null || maxDegrees.size() <= 1 || !Boolean.FALSE.equals(addHierarchy),
                        "multiple maxDegrees (one per layer) require withAddHierarchy(true)")
                .check(dimension == null || dimension > 0,
                        "dimension must be positive (was " + dimension + ")")
                .check(vectorValues == null || dimension == null || dimension == vectorValues.dimension(),
                        String.format("dimension(%s) does not match vectorValues.dimension()=%s; "
                                        + "omit withDimension(), it is derived automatically from vectorValues",
                                dimension, vectorValues == null ? null : vectorValues.dimension()))
                .check(!batch || existingGraph == null || vectorValues == null
                                || vectorValues.size() >= existingGraph.getIdUpperBound(),
                        String.format("vectorValues.size()=%s is smaller than existingGraph.getIdUpperBound()=%s; "
                                        + "when using withExistingGraph(), vectorValues must be a superset containing an "
                                        + "entry for every node ordinal already in the existing graph, in addition to the "
                                        + "new vectors being appended",
                                vectorValues == null ? null : vectorValues.size(),
                                existingGraph == null ? null : existingGraph.getIdUpperBound()))
                .throwIfAny(batch ? "Cannot build GraphIndex" : "Cannot build MutableHnswIndex");
    }

    /** Constructs the {@link GraphIndexBuilder} for already-validated configuration. */
    @SuppressWarnings("deprecation") // the constructors taking addHierarchy/refineFinalGraph explicitly
    private MutableHnswIndex newMutableIndex() {
        int resolvedDimension = vectorValues != null ? vectorValues.dimension() : dimension;
        BuildScoreProvider resolvedScoreProvider = scoreProvider != null
                ? scoreProvider
                : BuildScoreProvider.randomAccessScoreProvider(vectorValues, similarityFunction);

        GraphIndexBuilder builder;
        if (existingGraph != null) {
            builder = new GraphIndexBuilder(resolvedScoreProvider,
                    resolvedDimension,
                    existingGraph,
                    beamWidth,
                    neighborOverflow,
                    alpha,
                    refineFinalGraph,
                    simdExecutor,
                    parallelExecutor);
        } else {
            builder = new GraphIndexBuilder(resolvedScoreProvider,
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
        return new MutableHnswIndex(builder);
    }
}

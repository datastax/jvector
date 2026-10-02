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
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.index.HnswRecipe;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Fluent builder for graph/HNSW indexes. Obtain one from {@code Indexes.hnswBuilder(...)}, which picks
 * the concrete subclass from how scoring is supplied:
 * <ul>
 *     <li>{@code Indexes.hnswBuilder(RandomAccessVectorValues, VectorSimilarityFunction)} returns a
 *     {@link RavvHnswBuilder}, which derives the build score provider from the vectors, optionally
 *     compressing them first (see {@link #withCompressionType}).</li>
 *     <li>{@code Indexes.hnswBuilder(BuildScoreProvider, int)} returns a {@link ScoreProviderHnswBuilder},
 *     which uses the given score provider as is.</li>
 * </ul>
 * Every parameter starts at the same default {@link GraphIndexBuilder}'s convenience constructors use
 * (and that {@link HnswRecipe#DEFAULT} restates), so only the ones that differ need to be set.
 * <p>
 * The simplest use builds and populates a complete graph in one call; every setting has a default:
 * <pre>{@code
 * PersistableGraphIndex graph = Indexes.hnswBuilder(ravv, VectorSimilarityFunction.COSINE)
 *         .withCompressionType(CompressionType.PQ)   // optional: build with PQ-compressed scores
 *         .buildAndPopulate();
 * }</pre>
 * For more control, {@link #build()} creates the underlying {@link GraphIndexBuilder} on its first
 * call and returns its graph, which is still empty (or, with {@link #withExistingGraph}, holds only the
 * existing nodes); later calls return the same graph. Populate it either all at once with
 * {@link #populateGraph}, or incrementally with {@link #addGraphNode}, which is thread-safe and may run
 * concurrently with searches; call {@link #cleanup()} once incremental inserts and deletes are
 * finished. The {@code withXxx} settings must be made before the underlying builder is created;
 * changing them afterwards has no effect on it.
 * <p>
 * The methods that delegate to the underlying {@link GraphIndexBuilder} ({@link #populateGraph},
 * {@link #addGraphNode}, {@link #markNodeDeleted}, {@link #cleanup}, {@link #removeDeletedNodes} and
 * {@link #load}) call {@link #build()} first if it has not been called yet. The two queries,
 * {@link #insertsInProgress} and {@link #ramBytesUsed}, don't: before the graph is built they return 0.
 * <p>
 * Subclasses supply the underlying builder by implementing {@link #createGraphBuilder()}.
 * <p>
 * Close the builder when finished with it, to release the per-thread scratch space the underlying
 * builder caches for each thread that inserts; see {@link #close()}.
 */
public abstract class HnswIndexBuilder implements Closeable {
    // Graph shape and tuning settings shared by every subclass. How vectors are scored (and the
    // dimension) is the subclasses' business. Package-private, not private, for the subclasses,
    // rescore() and the tests in this package.
    List<Integer> maxDegrees = List.of(32);
    int beamWidth = 100;
    float neighborOverflow = 1.2f;
    float alpha = 1.2f;
    boolean addHierarchy = true;
    boolean refineFinalGraph = true;
    ForkJoinPool simdExecutor = PhysicalCoreExecutor.pool();
    ForkJoinPool parallelExecutor = ForkJoinPool.commonPool();
    MutableGraphIndex existingGraph;
    // Whether withMaxDegree(s)/withAddHierarchy were called, as opposed to holding their defaults or a
    // recipe's values; setting them explicitly conflicts with withExistingGraph.
    private boolean maxDegreesSet;
    private boolean addHierarchySet;
    /** Created once, by the first {@link #build()} (or delegating method) call; see {@code graphBuilder()}. */
    volatile GraphIndexBuilder graphBuilder;

    /**
     * Package-private so that {@link RavvHnswBuilder} and {@link ScoreProviderHnswBuilder} are the only
     * subclasses. Obtain a builder from {@code Indexes.hnswBuilder(...)}.
     */
    HnswIndexBuilder() {
    }

    /**
     * The compression to apply to the vectors before building, so the graph is built with
     * compressed (approximate) scores. Defaults to {@link CompressionType#NONE}. Only
     * {@link RavvHnswBuilder} supports it; {@link ScoreProviderHnswBuilder} logs a warning and
     * ignores it, since its score provider is fixed.
     */
    public abstract HnswIndexBuilder withCompressionType(CompressionType compressionType);

    /** The compression this builder will apply; {@link CompressionType#NONE} unless the subclass supports it. */
    CompressionType compressionType() {
        return CompressionType.NONE;
    }

    /** Sets a single max degree for all layers. Equivalent to {@code withMaxDegrees(List.of(maxDegree))}. */
    public HnswIndexBuilder withMaxDegree(int maxDegree) {
        return withMaxDegrees(List.of(maxDegree));
    }

    /**
     * The maximum number of connections a node can have in each layer; if fewer entries are
     * specified than the number of layers, the last entry is used for all remaining layers.
     * Defaults to {@code [32]}. Must not be set together with {@link #withExistingGraph}.
     */
    public HnswIndexBuilder withMaxDegrees(List<Integer> maxDegrees) {
        this.maxDegrees = maxDegrees;
        this.maxDegreesSet = true;
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
     * Whether to add an HNSW-style hierarchy on top of the Vamana index. Defaults to {@code true}.
     * Must not be set together with {@link #withExistingGraph}, whose hierarchy is already fixed.
     */
    public HnswIndexBuilder withAddHierarchy(boolean addHierarchy) {
        this.addHierarchy = addHierarchy;
        this.addHierarchySet = true;
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
     * Continue building on top of an existing graph, typically one reloaded with
     * {@code OnHeapGraphIndex.load}, instead of starting from an empty one. The existing graph is
     * mutated in place. Its max degrees and hierarchy are kept, so {@link #withMaxDegree},
     * {@link #withMaxDegrees} and {@link #withAddHierarchy} must not also be called: if they are,
     * building throws {@link IllegalStateException}. (Values set by {@link #applyRecipe} do not
     * conflict; the existing graph's take precedence.) New nodes should be added at ordinals from
     * {@link GraphIndex#getIdUpperBound()} onward.
     */
    public HnswIndexBuilder withExistingGraph(OnHeapGraphIndex existingGraph) {
        this.existingGraph = existingGraph;
        return this;
    }

    /**
     * Pre-sets this builder's fields to the given recipe's recommended values, looking each one up
     * by its {@link HnswRecipe.Param} name. Parameters the recipe doesn't set are left unchanged, and
     * any {@code withXxx} call made after this overrides the recipe's value.
     * <p>
     * Only {@link HnswRecipe#DEFAULT} is defined so far; it restates the defaults this builder
     * already starts with.
     *
     * @throws UnsupportedOperationException if the recipe has no defined values yet
     */
    @Experimental
    public HnswIndexBuilder applyRecipe(HnswRecipe recipe) {
        if (!recipe.isDefined()) {
            throw new UnsupportedOperationException(
                    "HnswRecipe." + recipe + " has no defined values yet");
        }
        if (recipe.has(HnswRecipe.Param.COMPRESSION_TYPE)) {
            CompressionType type = CompressionType.valueOf(recipe.get(HnswRecipe.Param.COMPRESSION_TYPE));
            // only go through the subclass when it changes something, so e.g. ScoreProviderHnswBuilder
            // doesn't warn about a recipe that just restates NONE
            if (type != compressionType()) {
                withCompressionType(type);
            }
        }
        if (recipe.has(HnswRecipe.Param.MAX_DEGREES)) {
            maxDegrees = recipe.get(HnswRecipe.Param.MAX_DEGREES);
        }
        if (recipe.has(HnswRecipe.Param.BEAM_WIDTH)) {
            beamWidth = recipe.get(HnswRecipe.Param.BEAM_WIDTH);
        }
        if (recipe.has(HnswRecipe.Param.NEIGHBOR_OVERFLOW)) {
            neighborOverflow = recipe.get(HnswRecipe.Param.NEIGHBOR_OVERFLOW);
        }
        if (recipe.has(HnswRecipe.Param.ALPHA)) {
            alpha = recipe.get(HnswRecipe.Param.ALPHA);
        }
        if (recipe.has(HnswRecipe.Param.ADD_HIERARCHY)) {
            addHierarchy = recipe.get(HnswRecipe.Param.ADD_HIERARCHY);
        }
        if (recipe.has(HnswRecipe.Param.REFINE_FINAL_GRAPH)) {
            refineFinalGraph = recipe.get(HnswRecipe.Param.REFINE_FINAL_GRAPH);
        }
        return this;
    }

    /**
     * Creates the underlying {@link GraphIndexBuilder} from this builder's current settings. Called
     * at most once per builder, by {@link #graphBuilder()}, while holding this builder's lock.
     */
    protected abstract GraphIndexBuilder createGraphBuilder();

    /**
     * Creates a {@link GraphIndexBuilder} from this builder's settings that scores with
     * {@code buildScoreProvider}: one continuing {@link #withExistingGraph the existing graph} if set,
     * otherwise one for a new graph. For subclasses' {@link #createGraphBuilder()}.
     */
    GraphIndexBuilder newGraphBuilder(BuildScoreProvider buildScoreProvider, int dimension) {
        if (existingGraph != null) {
            return new GraphIndexBuilder(buildScoreProvider, dimension, existingGraph, beamWidth,
                    neighborOverflow, alpha, refineFinalGraph, simdExecutor, parallelExecutor, null);
        }
        return new GraphIndexBuilder(buildScoreProvider, dimension, maxDegrees, beamWidth,
                neighborOverflow, alpha, addHierarchy, refineFinalGraph, simdExecutor, parallelExecutor, null);
    }

    /**
     * Returns the graph this builder builds, creating the underlying {@link GraphIndexBuilder} from
     * this builder's settings on the first call. The graph is empty unless {@link #withExistingGraph}
     * was set; add nodes to it with {@link #populateGraph} or {@link #addGraphNode}.
     * <p>
     * Idempotent and thread-safe: every call, from any thread, returns the same graph. Settings
     * changed with {@code withXxx} after the first call have no effect on it.
     *
     * @throws IllegalStateException if {@link #withExistingGraph} was combined with
     *         {@link #withMaxDegrees} or {@link #withAddHierarchy}
     */
    public final PersistableGraphIndex build() {
        return graphBuilder().getGraph();
    }

    /** Returns the graph being built. Equivalent to {@link #build()}. */
    public PersistableGraphIndex getGraph() {
        return build();
    }

    /**
     * Returns a builder holding a copy of {@code other}'s graph, with every edge re-scored by
     * {@code newProvider}, e.g. after the PQ codebook has been refined. The copy keeps
     * {@code other}'s nodes marked deleted and entry node. See {@link GraphIndexBuilder#rescore}.
     * <p>
     * The returned builder is a {@link ScoreProviderHnswBuilder} scoring with {@code newProvider}, and
     * is already built: {@link #build()} and {@link #getGraph()} return the copy, and nodes added to it
     * are scored with {@code newProvider}. Its settings are {@code other}'s, except that the max
     * degrees and hierarchy are read from {@code other}'s graph, so they are correct even when
     * {@code other} was built {@link #withExistingGraph on an existing graph}.
     * <p>
     * Builds {@code other} first if it has not been built yet. Must not run concurrently with
     * modifications to {@code other}'s graph.
     */
    public static HnswIndexBuilder rescore(HnswIndexBuilder other, BuildScoreProvider newProvider) {
        GraphIndexBuilder source = other.graphBuilder();
        MutableGraphIndex sourceGraph = source.graph;
        HnswIndexBuilder rescored = new ScoreProviderHnswBuilder(newProvider, sourceGraph.getDimension());
        rescored.maxDegrees = sourceGraph.maxDegrees();
        rescored.addHierarchy = sourceGraph.isHierarchical();
        rescored.beamWidth = other.beamWidth;
        rescored.neighborOverflow = other.neighborOverflow;
        rescored.alpha = other.alpha;
        rescored.refineFinalGraph = other.refineFinalGraph;
        rescored.simdExecutor = other.simdExecutor;
        rescored.parallelExecutor = other.parallelExecutor;
        rescored.graphBuilder = GraphIndexBuilder.rescore(source, newProvider);
        return rescored;
    }

    /**
     * Builds the graph and populates it from the vectors this builder was created with, in one call:
     * equivalent to {@code populateGraph(vectorValues)}. This is the simplest way to build a complete
     * index:
     * <pre>{@code
     * PersistableGraphIndex graph = Indexes.hnswBuilder(ravv, similarityFunction).buildAndPopulate();
     * }</pre>
     * Only a builder created from vectors ({@code Indexes.hnswBuilder(RandomAccessVectorValues,
     * VectorSimilarityFunction)}) has vectors to populate from; one created from a
     * {@link BuildScoreProvider} throws, and is populated with {@link #populateGraph} or
     * {@link #addGraphNode} instead.
     *
     * @return the populated graph
     * @throws UnsupportedOperationException if this builder was not created from vectors
     */
    public PersistableGraphIndex buildAndPopulate() {
        throw new UnsupportedOperationException(
                "This builder was created from a BuildScoreProvider and has no vectors of its own; "
                        + "use populateGraph(RandomAccessVectorValues) or addGraphNode instead");
    }

    /**
     * Adds the vectors in {@code ravv} to the graph in parallel, then calls {@link #cleanup()}.
     * Vector {@code i} is added at ordinal {@code i}, for every ordinal from the graph's
     * {@link GraphIndex#getIdUpperBound() id upper bound} up to {@code ravv.size() - 1}: for a new,
     * empty graph that is all of them, and when continuing a graph (see {@link #withExistingGraph}),
     * {@code ravv} is a superset whose leading entries are the nodes already in the graph, and only
     * the new ones are added. Builds the graph first if it has not been built yet.
     *
     * @return the populated graph
     */
    public PersistableGraphIndex populateGraph(RandomAccessVectorValues ravv) {
        GraphIndexBuilder gib = graphBuilder();
        int from = gib.graph.getIdUpperBound();
        if (from == 0) {
            return gib.build(ravv);
        }
        var vectors = ravv.threadLocalSupplier();
        simdExecutor.submit(() -> IntStream.range(from, ravv.size()).parallel()
                .forEach(node -> gib.addGraphNode(node, vectors.get().getVector(node)))).join();
        gib.cleanup();
        return gib.graph;
    }

    /**
     * Completes removal of deleted nodes, trims neighbor lists to the configured degree, refines
     * the graph if {@link #withRefineFinalGraph} is set, and marks it complete. Must be called
     * before writing the graph to disk, and not during concurrent modifications. See
     * {@link GraphIndexBuilder#cleanup()}.
     */
    public void cleanup() {
        graphBuilder().cleanup();
    }

    /**
     * The number of {@link #addGraphNode} calls in progress, or 0 if the graph hasn't been built yet.
     * See {@link GraphIndexBuilder#insertsInProgress()}.
     */
    public int insertsInProgress() {
        GraphIndexBuilder gib = this.graphBuilder;
        return gib == null ? 0 : gib.insertsInProgress();
    }

    /**
     * Rejects shape settings that conflict with {@link #withExistingGraph}, naming every conflict in
     * one exception.
     */
    private void validateShapeSettings() {
        if (existingGraph == null || (!maxDegreesSet && !addHierarchySet)) {
            return;
        }
        List<String> problems = new ArrayList<>();
        if (maxDegreesSet) {
            problems.add("withExistingGraph() takes its max degrees from the existing graph; "
                    + "don't also set withMaxDegree()/withMaxDegrees()");
        }
        if (addHierarchySet) {
            problems.add("withExistingGraph() takes its hierarchy from the existing graph; "
                    + "don't also set withAddHierarchy()");
        }
        throw new IllegalStateException("Cannot build HNSW index: " + String.join("; ", problems));
    }

    /**
     * Returns the underlying {@link GraphIndexBuilder}, creating it with {@link #createGraphBuilder()}
     * on the first call. Double-checked locking on the {@code volatile} {@link #graphBuilder} field,
     * so creation happens exactly once even when several threads call in concurrently.
     */
    private GraphIndexBuilder graphBuilder() {
        GraphIndexBuilder gib = this.graphBuilder;
        if (gib != null) {
            return gib;
        }
        synchronized (this) {
            if (this.graphBuilder == null) {
                validateShapeSettings();
                this.graphBuilder = createGraphBuilder();
            }
            return this.graphBuilder;
        }
    }

    /**
     * Inserts {@code node} with the given vector. Thread-safe: may be called concurrently with other
     * inserts and with searches. See {@link GraphIndexBuilder#addGraphNode(int, VectorFloat)}.
     *
     * @return an estimate of the number of extra bytes the graph uses after adding the node
     */
    public long addGraphNode(int node, VectorFloat<?> vector) {
        return graphBuilder().addGraphNode(node, vector);
    }

    /**
     * Inserts {@code node}, scoring it with {@code searchScoreProvider}, which must be compatible with
     * this builder's build score provider. See
     * {@link GraphIndexBuilder#addGraphNode(int, SearchScoreProvider)}.
     *
     * @return an estimate of the number of extra bytes the graph uses after adding the node
     */
    public long addGraphNode(int node, SearchScoreProvider searchScoreProvider) {
        return graphBuilder().addGraphNode(node, searchScoreProvider);
    }

    /**
     * Marks {@code node} deleted. It is hidden from searches immediately, but stays in the graph, so
     * its edges keep the graph connected, until {@link #removeDeletedNodes()} or {@link #cleanup()}
     * removes it. Thread-safe. See {@link GraphIndexBuilder#markNodeDeleted(int)}.
     */
    public void markNodeDeleted(int node) {
        graphBuilder().markNodeDeleted(node);
    }

    /**
     * Removes the nodes marked deleted (with {@link #markNodeDeleted}) from the graph and
     * repairs their neighbors' connections. Not thread-safe with respect to other modifications. See
     * {@link GraphIndexBuilder#removeDeletedNodes()}.
     *
     * @return an estimate of the memory no longer used
     */
    public long removeDeletedNodes() {
        return graphBuilder().removeDeletedNodes();
    }

    /**
     * The memory used by the underlying {@link GraphIndexBuilder}, including its graph, or 0 if the
     * graph hasn't been built yet.
     */
    public long ramBytesUsed() {
        GraphIndexBuilder gib = this.graphBuilder;
        return gib == null ? 0 : gib.ramBytesUsed();
    }

    /**
     * Loads a graph saved with {@code OnHeapGraphIndex.save} into this builder's graph, which must be
     * empty. Delegates to the deprecated {@link GraphIndexBuilder#load}; prefer
     * {@code OnHeapGraphIndex.load} with {@link #withExistingGraph}.
     */
    public void load(RandomAccessReader in) throws IOException {
        graphBuilder().load(in);
    }

    /**
     * Releases the per-thread scratch space (including a searcher per thread) that the underlying
     * {@link GraphIndexBuilder} caches for each thread that has inserted. Does nothing if the builder
     * was never built. The graph is unaffected and stays usable; if more nodes are added afterwards,
     * the scratch space is recreated on demand, and the builder should be closed again when done.
     * <p>
     * Not thread-safe: must not be called while inserts or other operations on this builder are in
     * progress.
     */
    @Override
    public void close() throws IOException {
        GraphIndexBuilder gib = this.graphBuilder;
        if (gib != null) {
            gib.close();
        }
    }

}

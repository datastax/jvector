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

package io.github.jbellis.jvector.ivf;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.index.IndexBuilderValidation;
import io.github.jbellis.jvector.index.IvfRecipe;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.concurrent.ForkJoinPool;

/**
 * Fluent builder for an {@link IvfIndex}, mirroring {@code HnswIndexBuilder}'s shape.
 * <p>
 * Only the construction inputs every backing needs are wired up so far &mdash; the vectors to
 * build from and how to score them, matching {@code HnswIndexBuilder} exactly. IVF's own
 * construction parameters (e.g. {@code nlist}) are still being defined by the IVF design and are
 * deliberately not guessed at here; {@link #build()} refuses until they exist.
 */
public class IvfIndexBuilder {
    private BuildScoreProvider scoreProvider;
    private RandomAccessVectorValues vectorValues;
    private VectorSimilarityFunction similarityFunction;

    private ForkJoinPool simdExecutor = PhysicalCoreExecutor.pool();
    private ForkJoinPool parallelExecutor = ForkJoinPool.commonPool();

    public IvfIndexBuilder() {
    }

    /**
     * Supplies the score provider directly, for scoring that is not a plain exact comparison of
     * the raw vectors (e.g. a PQ/BQ-compressed provider). Mutually exclusive with
     * {@link #withSimilarityFunction}.
     */
    public IvfIndexBuilder withScoreProvider(BuildScoreProvider scoreProvider) {
        this.scoreProvider = scoreProvider;
        return this;
    }

    /**
     * Supplies the vectors to build the index from. Always required.
     */
    public IvfIndexBuilder withVectorValues(RandomAccessVectorValues vectorValues) {
        this.vectorValues = vectorValues;
        return this;
    }

    /**
     * The similarity metric to use during construction, used to derive a score provider that
     * performs exact comparisons against {@link #withVectorValues}. Mutually exclusive with
     * {@link #withScoreProvider}.
     */
    public IvfIndexBuilder withSimilarityFunction(VectorSimilarityFunction similarityFunction) {
        this.similarityFunction = similarityFunction;
        return this;
    }

    /**
     * ForkJoinPool instance for SIMD operations. Defaults to {@link PhysicalCoreExecutor#pool()}.
     */
    public IvfIndexBuilder withSimdExecutor(ForkJoinPool simdExecutor) {
        this.simdExecutor = simdExecutor;
        return this;
    }

    /**
     * ForkJoinPool instance for parallel stream operations. Defaults to
     * {@link ForkJoinPool#commonPool()}.
     */
    public IvfIndexBuilder withParallelExecutor(ForkJoinPool parallelExecutor) {
        this.parallelExecutor = parallelExecutor;
        return this;
    }

    /**
     * Pre-sets this builder's fixed fields to the given recipe's recommended values, leaving the
     * recipe's free parameters for the caller to still supply.
     * <p>
     * Scaffolding only: the recipes' actual fixed-value formulas haven't been decided yet, so
     * every {@link IvfRecipe} currently refuses here rather than guess at numbers.
     *
     * @throws UnsupportedOperationException always, until a recipe's values are defined
     */
    public IvfIndexBuilder applyRecipe(IvfRecipe recipe) {
        throw new UnsupportedOperationException(
                "IvfRecipe." + recipe + " has no defined values yet");
    }

    /**
     * Validates the construction inputs common to every backing, then refuses: IVF's own
     * construction parameters (beyond the vectors and scoring already validated here) haven't
     * been defined yet, so there is nothing to actually build.
     *
     * @throws IllegalStateException if a mutually-exclusive pair was over-specified, or a
     * required common value is missing
     * @throws UnsupportedOperationException always, until IVF's construction parameters and
     * backing implementation exist
     */
    public IvfIndex build() {
        if (scoreProvider != null && similarityFunction != null) {
            throw new IllegalStateException(
                    "Set either withScoreProvider() or withSimilarityFunction(), not both");
        }

        new IndexBuilderValidation()
                .require("vectorValues", vectorValues)
                .requireCondition("similarityFunction (or scoreProvider)",
                        scoreProvider != null || similarityFunction != null)
                .throwIfAny("Cannot build IvfIndex");

        throw new UnsupportedOperationException(
                "IVF construction is not yet implemented: its construction parameters and backing "
                        + "implementation are still being defined");
    }
}

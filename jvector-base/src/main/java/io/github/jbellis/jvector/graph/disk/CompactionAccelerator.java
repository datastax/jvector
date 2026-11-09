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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/**
 * Runs parts of {@link OnDiskGraphIndexCompactor}'s full-precision merge somewhere else, for example on a GPU, without
 * changing what the merge computes. Found with {@link CompactionAccelerators#find()}; without one, or when
 * {@link #walk} returns false, the compactor does the work itself.
 */
@Experimental
public interface CompactionAccelerator {

    /** The name {@link CompactionAccelerators#PROPERTY} selects it by. */
    String name();

    /** Whether this accelerator can walk cells of {@code dimension}-dimensional vectors under {@code similarity}. */
    boolean supports(VectorSimilarityFunction similarity, int dimension);

    /**
     * Scores every (cell, query) pair of one target source's cell walk and keeps each query's top
     * {@link CellWalk#topK()} candidates through {@link CellWalk#setTopK}. Scores must be JVector's exact scores of the
     * raw vectors ({@link CellWalk#score}).
     *
     * @return true if it did the walk; false to leave it to the compactor, which then walks as usual
     */
    boolean walk(CellWalk walk);

    /**
     * Ranks the cells each query of a range probes ({@link CellProbe#setProbes}), in place of the compactor's beam search.
     *
     * @return true if it ranked them; false (the default) to leave it to the compactor
     */
    default boolean probe(CellProbe probe) {
        return false;
    }

    /**
     * Trains the merge's wide code, as {@code ProductQuantization.compute(sample, subspaces, clusters, globallyCenter)}
     * would, in place of the compactor's own training. The code is used only for the finalize's diversity comparisons.
     *
     * @return the trained quantization, or null (the default) to leave it to the compactor
     */
    default ProductQuantization trainWideCode(RandomAccessVectorValues sample, int subspaces, int clusters, boolean globallyCenter) {
        return null;
    }
}

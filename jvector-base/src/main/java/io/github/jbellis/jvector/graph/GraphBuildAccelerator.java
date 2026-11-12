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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/**
 * A pluggable way to construct a graph index outside {@link GraphIndexBuilder}'s incremental insertion, for example
 * on a GPU. Implementations are discovered with {@link java.util.ServiceLoader} and selected with
 * {@link GraphBuildAccelerators#find()}.
 * <p>
 * An accelerator fills an empty {@link GraphIndexBuilder}: for every node and layer it computes a diversity-pruned
 * neighbor list (at most {@link GraphIndexBuilder#getMaxDegree()} neighbors, pruned with the builder's alpha) and
 * adds it with {@link GraphIndexBuilder#addGraphNodeWithNeighbors}, then calls
 * {@link GraphIndexBuilder#completeExternalBuild}. With a hierarchy, node levels come from
 * {@link GraphIndexBuilder#sampleGraphLevels}. The caller then runs {@link GraphIndexBuilder#cleanup()} and writes the
 * index as usual, so search and the on-disk format are unchanged.
 */
@Experimental
public interface GraphBuildAccelerator {

    /** @return the name used to select this accelerator ({@code -Djvector.graph.accelerator=<name>}) */
    String name();

    /**
     * @param builder    the builder that would be filled; its max degree and alpha are the build parameters
     * @param vectors    the vectors
     * @param similarity the similarity function
     * @return whether this accelerator can build this graph on this machine (device present, supported similarity,
     * dimension and degree, enough memory); if not, the caller builds it with {@link GraphIndexBuilder} as usual
     */
    boolean supports(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity);

    /**
     * Builds the graph of all {@code vectors} into {@code builder}, which must be empty.
     *
     * @param builder    the builder to fill; its max degree, alpha and hierarchy setting are the build parameters
     * @param vectors    the vectors, ordinal i being node i
     * @param similarity the similarity function
     */
    void build(GraphIndexBuilder builder, RandomAccessVectorValues vectors, VectorSimilarityFunction similarity);
}

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

package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.BQVectors;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Encapsulates comparing node distances for GraphIndexBuilder.
 */
public interface BuildScoreProvider {
    /** Vector type support for creating and manipulating vectors. */
    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * Returns true if the primary score functions used for construction are exact.
     * This is modestly redundant, but it saves having to allocate new Search/Diversity provider
     * objects in some hot construction loops.
     * @return true if the primary score functions are exact, false if they are approximate
     */
    boolean isExact();

    /**
     * Returns the approximate centroid of the known nodes. The closest node to this centroid
     * is used as the graph entry point, so this is called when the entry point is deleted
     * or every time the graph size doubles.
     * <p>
     * This is not called on a path that blocks searches or modifications, so it is okay for it to be O(N).
     * @return the approximate centroid of the known nodes
     */
    VectorFloat<?> approximateCentroid();

    /**
     * Creates a search score provider to use internally during construction.
     * <p>
     * "Internally" means that this may differ from a typical SSP in that it may use
     * approximate scores without reranking. (In this case, reranking will be done
     * separately by the ConcurrentNeighborSet diversity code.)
     * @param vector the query vector to provide similarity scores against
     * @return a SearchScoreProvider for the given query vector
     */
    SearchScoreProvider searchProviderFor(VectorFloat<?> vector);

    /**
     * Creates a search score provider to use internally during construction.
     * <p>
     * "Internally" means that this may differ from a typical SSP in that it may use
     * approximate scores without reranking. (In this case, reranking will be done
     * separately by the ConcurrentNeighborSet diversity code.)
     * @param node1 the graph node to provide similarity scores against
     * @return a SearchScoreProvider for the given node
     */
    SearchScoreProvider searchProviderFor(int node1);

    /**
     * Creates a score provider to use internally during construction.
     * <p>
     * The difference between the diversity provider and the search provider is
     * that the diversity provider is only expected to be used a few dozen times per node,
     * which influences the implementation choices.
     * <p>
     * When scoring is approximate, the scores from the search and diversity provider
     * must be consistent, i.e. mixing different types of CompressedVectors will cause problems.
     * @param node1 the graph node to provide diversity scores against
     * @return a SearchScoreProvider for diversity computation
     */
    SearchScoreProvider diversityProviderFor(int node1);

    /**
     * Returns a BSP that performs exact score comparisons using the given RandomAccessVectorValues and VectorSimilarityFunction.
     * @param ravv the RandomAccessVectorValues providing access to vectors
     * @param similarityFunction the similarity function to use for scoring
     * @return a BuildScoreProvider that performs exact score comparisons
     */
    static BuildScoreProvider randomAccessScoreProvider(RandomAccessVectorValues ravv, VectorSimilarityFunction similarityFunction) {
        // We need two sources of vectors in order to perform diversity check comparisons without
        // colliding.  ThreadLocalSupplier makes this a no-op if the RAVV is actually un-shared.
        var vectors = ravv.threadLocalSupplier();
        var vectorsCopy = ravv.threadLocalSupplier();

        return new BuildScoreProvider() {
            @Override
            public boolean isExact() {
                return true;
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                var vv = vectors.get();
                var centroid = vts.createFloatVector(vv.dimension());
                for (int i = 0; i < vv.size(); i++) {
                    var v = vv.getVector(i);
                    if (v != null) { // MapRandomAccessVectorValues is not necessarily dense
                        VectorUtil.addInPlace(centroid, v);
                    }
                }
                VectorUtil.scale(centroid, 1.0f / vv.size());
                return centroid;
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                var vc = vectorsCopy.get();
                return DefaultSearchScoreProvider.exact(vector, similarityFunction, vc);
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                RandomAccessVectorValues randomAccessVectorValues = vectors.get();
                var v = randomAccessVectorValues.getVector(node1);
                return searchProviderFor(v);
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                RandomAccessVectorValues randomAccessVectorValues = vectors.get();
                var v = randomAccessVectorValues.getVector(node1);
                var vc = vectorsCopy.get();
                return DefaultSearchScoreProvider.exact(v, similarityFunction, vc);
            }
        };
    }

    /**
     * Returns a BSP that performs approximate score comparisons using the given PQVectors,
     * with reranking performed using RandomAccessVectorValues (which is intended to be
     * InlineVectorValues for building incrementally, but should technically
     * work with any RAVV implementation).
     * <p>
     * This implementation is not thread safe; results should never be published to another thread.
     * @param vsf the vector similarity function to use
     * @param pqv the product quantized vectors
     * @return a BuildScoreProvider that uses product quantization for approximate scoring
     */
    static BuildScoreProvider pqBuildScoreProvider(VectorSimilarityFunction vsf, PQVectors pqv) {
        int dimension = pqv.getOriginalSize() / Float.BYTES;
        final ThreadLocal<VectorFloat<?>> reusableVector = ThreadLocal.withInitial(() -> vts.createFloatVector(dimension));;

        return new BuildScoreProvider() {
            @Override
            public boolean isExact() {
                return false;
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                // like searchProviderFor, this skips reranking; unlike sPF, it uses pqv.scoreFunctionFor
                // instead of precomputedScoreFunctionFor; since we only perform a few dozen comparisons
                // during diversity computation, this is cheaper than precomputing a lookup table
                var asf = pqv.diversityFunctionFor(node1, vsf); // not precomputed!
                return new DefaultSearchScoreProvider(asf);
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                VectorFloat<?> decoded = reusableVector.get();
                pqv.getCompressor().decode(pqv.get(node1), decoded);
                return searchProviderFor(decoded);
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                // deliberately skips reranking even though we are using an approximate score function
                return new DefaultSearchScoreProvider(pqv.precomputedScoreFunctionFor(vector, vsf));
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                return pqv.getCompressor().getOrComputeCentroid();
            }
        };
    }

    /**
     * Returns a BSP that performs approximate score comparisons using binary quantization.
     * Binary quantization compresses vectors to single bits and uses Hamming distance for
     * similarity computation, providing a fast approximation suitable for cosine similarity.
     * @param bqv the binary quantized vectors
     * @return a BuildScoreProvider that uses binary quantization for approximate scoring
     */
    static BuildScoreProvider bqBuildScoreProvider(BQVectors bqv) {
        return new BuildScoreProvider() {
            @Override
            public boolean isExact() {
                return false;
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                // centroid = zeros is actually a decent approximation
                return vts.createFloatVector(bqv.getCompressor().getOriginalDimension());
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                return new DefaultSearchScoreProvider(bqv.precomputedScoreFunctionFor(vector, null));
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                var encoded1 = bqv.get(node1);
                return new DefaultSearchScoreProvider(new ScoreFunction() {
                    @Override
                    public boolean isExact() {
                        return false;
                    }

                    @Override
                    public float similarityTo(int node2) {
                        return bqv.similarityBetween(encoded1, bqv.get(node2));
                    }
                });
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                return searchProviderFor(node1);
            }
        };
    }
}

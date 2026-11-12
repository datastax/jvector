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

import io.github.jbellis.jvector.graph.FlatVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RemappedRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.ASHSymmetricScorer;
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
    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * @return true if the primary score functions used for construction are exact.  This
     * is modestly redundant, but it saves having to allocate new Search/Diversity provider
     * objects in some hot construction loops.
     */
    boolean isExact();

    /**
     * @return the approximate centroid of the known nodes.  We use the closest node
     * to this centroid as the graph entry point, so this is called when the entry point is deleted
     * or every time the graph size doubles.
     * <p>
     * This is not called on a path that blocks searches or modifications, so it is okay for it to be O(N).
     */
    VectorFloat<?> approximateCentroid();

    /**
     * Create a search score provider to use *internally* during construction.
     * <p>
     * "Internally" means that this may differ from a typical SSP in that it may use
     * approximate scores *without* reranking.  (In this case, reranking will be done
     * separately by the ConcurrentNeighborSet diversity code.)
     * <p>
     * @param vector the query vector to provide similarity scores against
     */
    SearchScoreProvider searchProviderFor(VectorFloat<?> vector);

    /**
     * Create a search score provider to use *internally* during construction.
     * <p>
     * "Internally" means that this may differ from a typical SSP in that it may use
     * approximate scores *without* reranking.  (In this case, reranking will be done
     * separately by the ConcurrentNeighborSet diversity code.)
     * <p>
     * @param node1 the graph node to provide similarity scores against
     */
    SearchScoreProvider searchProviderFor(int node1);

    /**
     * Create a score provider to use internally during construction.
     * <p>
     * The difference between the diversity provider and the search provider is
     * that the diversity provider is only expected to be used a few dozen times per node,
     * which influences the implementation choices.
     * <p>
     * When scoring is approximate, the scores from the search and diversity provider
     * must be consistent, i.e. mixing different types of CompressedVectors will cause problems.
     */
    SearchScoreProvider diversityProviderFor(int node1);

    /**
     * Create the diversity provider's score function. See {@link #diversityProviderFor(int)} for documentation
     * on the use of the ScoreFunction.
     */
    default ScoreFunction diversityScoreFunctionFor(int node1) {
        return diversityProviderFor(node1).scoreFunction();
    }

    /**
     * Like {@link #diversityScoreFunctionFor(int)}, but may re-aim {@code previous} at {@code node1} and return it
     * instead of creating a new score function. {@code previous} is null or a score function this method returned
     * earlier that the caller no longer needs for its earlier node.
     */
    default ScoreFunction diversityScoreFunctionFor(int node1, ScoreFunction previous) {
        return diversityScoreFunctionFor(node1);
    }

    /**
     * Returns a BSP that performs exact score comparisons using the given RandomAccessVectorValues and VectorSimilarityFunction.
     *
     * Helper method for the special case that mapping between graph node IDs and ravv ordinals is the identity function.
     */
    static BuildScoreProvider randomAccessScoreProvider(RandomAccessVectorValues ravv, int[] graphToRavvOrdMap, VectorSimilarityFunction similarityFunction) {
        return randomAccessScoreProvider(new RemappedRandomAccessVectorValues(ravv, graphToRavvOrdMap), similarityFunction);
    }

    /**
     * Returns a BSP that performs exact score comparisons using the given RandomAccessVectorValues and VectorSimilarityFunction.
     * graphToRavvOrdMap maps graph node IDs to ravv ordinals.
     */
    static BuildScoreProvider randomAccessScoreProvider(RandomAccessVectorValues ravv, VectorSimilarityFunction similarityFunction) {
        if (ravv instanceof FlatVectorValues) {
            return flatScoreProvider((FlatVectorValues) ravv, similarityFunction);
        }
        // We need two sources of vectors in order to perform diversity check comparisons without
        // colliding.  ThreadLocalSupplier makes this a no-op if the RAVV is actually un-shared.
        var vectors = ravv.threadLocalSupplier();
        var vectorsCopy = ravv.threadLocalSupplier();

        return new BuildScoreProvider() {
            /// Exact scores against `node1`.
            ///
            /// For a value-shared source the anchor is read on the first comparison and kept as
            /// a private copy, so creating a score function reads nothing, later reads through the
            /// same views can't overwrite the anchor, and each comparison reads only `node2`.
            private ScoreFunction.ExactScoreFunction scoreFunctionFor(int node1) {
                if (ravv.isValueShared()) {
                    return new ScoreFunction.ExactScoreFunction() {
                        private VectorFloat<?> anchor;

                        @Override
                        public float similarityTo(int node2) {
                            var a = anchor;
                            if (a == null) anchor = a = vectors.get().getVector(node1).copy();
                            return similarityFunction.compare(a, vectorsCopy.get().getVector(node2));
                        }
                    };
                }
                var vector = vectors.get().getVector(node1);
                var vc = vectorsCopy.get();
                return node2 -> similarityFunction.compare(vector, vc.getVector(node2));
            }

            @Override
            public boolean isExact() {
                return true;
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                var vv = vectors.get();
                var centroid = vts.createFloatVector(vv.dimension());
                int size = vv.size();
                for (int i = 0; i < size; i++) {
                    var v = vv.getVector(i);
                    if (v != null) VectorUtil.addInPlace(centroid, v);
                }
                VectorUtil.scale(centroid, 1.0f / size);
                return centroid;
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                // Construction keeps the standard stopping rule for full-precision scores.
                var vc = vectorsCopy.get();
                return new DefaultSearchScoreProvider(DefaultSearchScoreProvider.exactScoreFunction(vector, similarityFunction, vc));
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                return new DefaultSearchScoreProvider(scoreFunctionFor(node1));
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                return new DefaultSearchScoreProvider(scoreFunctionFor(node1));
            }

            @Override
            public ScoreFunction diversityScoreFunctionFor(int node1) {
                return scoreFunctionFor(node1);
            }
        };
    }

    /** Exact scores read in place from a flat store; diversity checks re-aim one score function per caller. */
    private static BuildScoreProvider flatScoreProvider(FlatVectorValues flat, VectorSimilarityFunction similarityFunction) {
        return new BuildScoreProvider() {
            @Override
            public boolean isExact() {
                return true;
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                var centroid = vts.createFloatVector(flat.dimension());
                var row = vts.createFloatVector(flat.dimension());
                for (int i = 0; i < flat.size(); i++) {
                    flat.getVectorInto(i, row, 0);
                    VectorUtil.addInPlace(centroid, row);
                }
                VectorUtil.scale(centroid, 1.0f / flat.size());
                return centroid;
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                return new DefaultSearchScoreProvider(flat.scoreFunctionFor(vector, similarityFunction));
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                return new DefaultSearchScoreProvider(flat.scoreFunctionFor(node1, similarityFunction));
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                return new DefaultSearchScoreProvider(flat.scoreFunctionFor(node1, similarityFunction));
            }

            @Override
            public ScoreFunction diversityScoreFunctionFor(int node1) {
                return flat.scoreFunctionFor(node1, similarityFunction);
            }

            @Override
            public ScoreFunction diversityScoreFunctionFor(int node1, ScoreFunction previous) {
                if (previous instanceof FlatVectorValues.AnchoredScoreFunction) {
                    var anchored = (FlatVectorValues.AnchoredScoreFunction) previous;
                    if (anchored.scores(flat, similarityFunction)) return anchored.anchorAt(node1);
                }
                return flat.scoreFunctionFor(node1, similarityFunction);
            }
        };
    }

    /**
     * Returns a BSP that performs approximate score comparisons using the given PQVectors,
     * with reranking performed using RandomAccessVectorValues (which is intended to be
     * InlineVectorValues for building incrementally, but should technically
     * work with any RAVV implementation).
     * This class is not thread safe, we should never publish its results to another thread.
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
            public ScoreFunction diversityScoreFunctionFor(int node1) {
                // like searchProviderFor, this skips reranking; unlike sPF, it uses pqv.scoreFunctionFor
                // instead of precomputedScoreFunctionFor; since we only perform a few dozen comparisons
                // during diversity computation, this is cheaper than precomputing a lookup table
                return pqv.diversityFunctionFor(node1, vsf); // not precomputed!
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                var asf = diversityScoreFunctionFor(node1);
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
                return new DefaultSearchScoreProvider(diversityScoreFunctionFor(node1));
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                return searchProviderFor(node1);
            }

            @Override
            public ScoreFunction diversityScoreFunctionFor(int node1) {
                var encoded1 = bqv.get(node1);
                return new ScoreFunction() {
                    @Override
                    public boolean isExact() {
                        return false;
                    }

                    @Override
                    public float similarityTo(int node2) {
                        return bqv.similarityBetween(encoded1, bqv.get(node2));
                    }
                };
            }
        };
    }

    /**
     * Returns a BSP that performs approximate score comparisons using the given ASHVectors.
     *
     * <p><b>Construction (node-node) scoring:</b> Uses symmetric ASH–ASH dot-product approximation
     * computed from the complete encoded payload + per-vector header fields (scale, offset).
     *
     * <p><b>Query (vector-node) scoring:</b> Delegates to ASHVectors' query-time score function
     * (asymmetric float query → ASH vector).
     *
     * <p><b>Constraint:</b> This provider requires {@code C=1} (single landmark) and
     * {@link VectorSimilarityFunction#DOT_PRODUCT}.
     */
    static BuildScoreProvider ashBuildScoreProvider(VectorSimilarityFunction vsf, ASHVectors ashv) {
        if (vsf != VectorSimilarityFunction.DOT_PRODUCT) {
            throw new UnsupportedOperationException("ASH build scoring supports DOT_PRODUCT only");
        }

        final AsymmetricHashing ash = ashv.getCompressor();
        if (ash.landmarkCount != 1) {
            throw new IllegalArgumentException("ASH build scoring requires landmarkCount==1, got " + ash.landmarkCount);
        }

        final var symmetric = new ASHSymmetricScorer(ashv);

        return new BuildScoreProvider() {
            @Override
            public boolean isExact() {
                return false;
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                // In the C=1 case, ASH uses the dataset mean as its single landmark.
                // Return a copy so callers can't mutate ASH's internal landmark.
                return ash.landmarks[0].copy();
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                // Query-time asymmetric scoring is already implemented by ASHVectors.
                return new DefaultSearchScoreProvider(ashv.precomputedScoreFunctionFor(vector, vsf), null, VectorSimilarityFunction.DOT_PRODUCT);
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                // A stored node's length is not kept; the largest encoded length bounds it.
                float m = ash.maxEncodedNorm();
                return new DefaultSearchScoreProvider(symmetric.scoreFunctionFor(node1), null,
                        VectorSimilarityFunction.DOT_PRODUCT, m * m);
            }

            @Override
            public SearchScoreProvider diversityProviderFor(int node1) {
                // Keep search/diversity consistent for approximate scoring.
                return searchProviderFor(node1);
            }
        };
    }
}

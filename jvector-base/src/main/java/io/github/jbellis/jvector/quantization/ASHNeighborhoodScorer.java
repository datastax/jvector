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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Objects;

/**
 * Prepared query state for scoring serialized {@link FusedASHLayout} neighborhoods.
 *
 * <p>Construction computes the query projection, landmark terms, and nibble LUT once. Each
 * {@link #score} call then performs lookup, binary16 header decoding, landmark correction, and
 * conversion to the graph's nonnegative DOT_PRODUCT similarity. The graph decoder and isolated
 * benchmark share this implementation.</p>
 *
 * <p>Prepared state is immutable. Instances may be shared across threads with independent output
 * arrays and immutable packed inputs. Scalar/Panama execution supports all model landmarks;
 * a capable native provider handles C=1 with one call per entire neighborhood.</p>
 */
public final class ASHNeighborhoodScorer {
    private final int maxDegree;
    private final int blockSize;
    private final int blocks;
    private final int dimensions;
    private final int bits;
    private final int groups;
    private final int featureSize;
    private final float[] lut;
    private final float[] dotQMu;
    private final VectorUtilSupport neighborhoodBackend;
    private final ASHLutKernel kernel = new ASHLutKernel();

    /**
     * Prepares a query for neighborhoods with a fixed serialized shape.
     *
     * @param ash compressor describing the projection, bit width, and landmarks
     * @param query original-dimensional query; callers apply any desired normalization
     * @param maxDegree positive scored capacity, including padded graph lanes
     * @param blockSize serialized block width: 8, 16, or 32
     * @throws IllegalArgumentException for a dimension mismatch or invalid shape
     * @throws ArithmeticException if the serialized shape cannot fit a Java array
     */
    public ASHNeighborhoodScorer(AsymmetricHashing ash, VectorFloat<?> query, int maxDegree, int blockSize) {
        this(ash, prepareQuery(ash, query, maxDegree, blockSize), maxDegree, blockSize);
    }

    private static ASHScorer.QueryPrecompute prepareQuery(
            AsymmetricHashing ash, VectorFloat<?> query, int maxDegree, int blockSize) {
        Objects.requireNonNull(ash, "ash");
        Objects.requireNonNull(query, "query");
        if (query.length() != ash.originalDimension) {
            throw new IllegalArgumentException("ASH query dimension must be " + ash.originalDimension);
        }
        FusedASHLayout.featureSize(maxDegree, ash.quantizedDim, ash.bitsPerDimension, blockSize);
        return new ASHScorer(ash).precomputeQuery(query);
    }

    /** Reuses preparation already shared with upper-layer scoring in {@link FusedASHDecoder}. */
    ASHNeighborhoodScorer(AsymmetricHashing ash, ASHScorer.QueryPrecompute query, int maxDegree, int blockSize) {
        FusedASHLayout.validateBitsPerDimension(ash.bitsPerDimension);
        FusedASHLayout.validateBlockSize(blockSize);
        this.maxDegree = maxDegree;
        this.blockSize = blockSize;
        this.dimensions = ash.quantizedDim;
        this.bits = ash.bitsPerDimension;
        this.groups = FusedASHLayout.codeGroups(dimensions, bits);
        this.blocks = FusedASHLayout.blocksPerNode(maxDegree, blockSize);
        this.featureSize = FusedASHLayout.featureSize(maxDegree, dimensions, bits, blockSize);
        this.dotQMu = query.dotQMuByLandmark;
        this.lut = new float[Math.multiplyExact(groups, 16)];
        FusedASHLayout.buildQueryLut(query.qProj, dimensions, bits, lut);
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        this.neighborhoodBackend =
                kernel.isSimd() && dotQMu.length == 1 && backend.supportsAshNeighborhoodScoring() ? backend : null;
    }

    /**
     * Scores the configured capacity, including padded neighbors, in serialized lane order.
     *
     * <p>On failure, output contents are unspecified. The graph decoder invalidates its cached
     * origin before this call, so partially written scores cannot be reused.</p>
     *
     * @param packed complete neighborhood bytes in the constructor's layout; not modified
     * @param out receives one similarity per configured lane; later elements remain unchanged
     * @throws IndexOutOfBoundsException if either array is too small
     * @throws IllegalStateException if a scored lane refers to an invalid landmark
     */
    public void score(byte[] packed, float[] out) {
        Objects.checkFromIndexSize(0, featureSize, packed.length);
        Objects.checkFromIndexSize(0, maxDegree, out.length);
        if (neighborhoodBackend != null) {
            neighborhoodBackend.ashScoreNeighborhood(packed, groups, blockSize, maxDegree, lut, dotQMu[0], out);
            return;
        }
        for (int block = 0; block < blocks; block++) {
            int offset = FusedASHLayout.blockOffset(block, dimensions, bits, blockSize);
            int base = block * blockSize;
            int count = Math.min(blockSize, maxDegree - base);
            kernel.score(packed, offset, groups, blockSize, 0, count, lut, out, base);
            for (int lane = 0; lane < count; lane++) {
                int landmark = FusedASHLayout.readLandmark(packed, offset, lane, dimensions, bits, blockSize);
                if (landmark >= dotQMu.length) {
                    throw new IllegalStateException("Invalid FusedASH landmark: " + landmark);
                }
                out[base + lane] = ASHScorer.toSimilarity(
                        FusedASHLayout.readScale(packed, offset, lane, dimensions, bits, blockSize) * out[base + lane]
                        + dotQMu[landmark]
                        + FusedASHLayout.readOffset(packed, offset, lane, dimensions, bits, blockSize));
            }
        }
    }

    @Override
    public String toString() {
        Object implementation = neighborhoodBackend == null ? kernel : "native Highway whole-neighborhood FFM";
        return "ASHNeighborhoodScorer[" + implementation + ", blockSize=" + blockSize + ", capacity=" + maxDegree + "]";
    }
}

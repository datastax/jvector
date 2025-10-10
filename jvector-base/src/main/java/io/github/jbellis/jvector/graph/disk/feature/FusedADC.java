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

package io.github.jbellis.jvector.graph.disk.feature;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.FusedADCPQDecoder;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Implements Quick ADC-style scoring by fusing PQ-encoded neighbors into an OnDiskGraphIndex.
 */
public class FusedADC implements Feature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ProductQuantization pq;
    private final int maxDegree;
    private final ThreadLocal<VectorFloat<?>> reusableResults;
    private final ExplicitThreadLocal<ByteSequence<?>> reusableNeighbors;
    private ByteSequence<?> compressedNeighbors = null;

    /**
     * Constructs a FusedADC feature with the given parameters.
     *
     * @param maxDegree the maximum degree of the graph (must be 32)
     * @param pq the product quantization scheme to use for encoding neighbors
     */
    public FusedADC(int maxDegree, ProductQuantization pq) {
        if (maxDegree != 32) {
            throw new IllegalArgumentException("maxDegree must be 32 for FusedADC. This limitation may be removed in future releases");
        }
        if (pq.getClusterCount() != 256) {
            throw new IllegalArgumentException("FusedADC requires a 256-cluster PQ. This limitation may be removed in future releases");
        }
        this.maxDegree = maxDegree;
        this.pq = pq;
        this.reusableResults = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(maxDegree));
        this.reusableNeighbors = ExplicitThreadLocal.withInitial(() -> vectorTypeSupport.createByteSequence(pq.compressedVectorSize() * maxDegree));
    }

    @Override
    public FeatureId id() {
        return FeatureId.FUSED_ADC;
    }

    @Override
    public int headerSize() {
        return pq.compressorSize();
    }

    @Override
    public int featureSize() {
        return pq.compressedVectorSize() * maxDegree;
    }

    static FusedADC load(CommonHeader header, RandomAccessReader reader) {
        // TODO doesn't work with different degrees
        try {
            return new FusedADC(header.layerInfo.get(0).degree, ProductQuantization.load(reader));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Creates an approximate score function for the given query vector using fused PQ decoding.
     *
     * @param queryVector the query vector to compute similarities against
     * @param vsf the vector similarity function to use (DOT_PRODUCT, EUCLIDEAN, or COSINE)
     * @param view the graph index view providing access to packed neighbors
     * @param esf exact score function for fallback computations
     * @return an approximate score function that efficiently scores quantized neighbors
     */
    public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf, OnDiskGraphIndex.View view, ScoreFunction.ExactScoreFunction esf) {
        var neighbors = new PackedNeighbors(view);
        return FusedADCPQDecoder.newDecoder(neighbors, pq, queryVector, reusableResults.get(), vsf, esf);
    }

    @Override
    public void writeHeader(DataOutput out) throws IOException {
        pq.write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    // this is an awkward fit for the Feature.State design since we need to
    // generate the fused set based on the neighbors of the node, not just the node itself
    @Override
    public void writeInline(DataOutput out, Feature.State state_) throws IOException {
        if (compressedNeighbors == null) {
            compressedNeighbors = vectorTypeSupport.createByteSequence(pq.compressedVectorSize() * maxDegree);
        }
        var state = (FusedADC.State) state_;
        var pqv = state.pqVectors;

        var neighbors = state.view.getNeighborsIterator(0, state.nodeId);
        int n = 0;
        compressedNeighbors.zero();
        while (neighbors.hasNext()) {
            var compressed = pqv.get(neighbors.nextInt());
            for (int j = 0; j < pqv.getCompressedSize(); j++) {
                compressedNeighbors.set(j * maxDegree + n, compressed.get(j));
            }
            n++;
        }

        vectorTypeSupport.writeByteSequence(out, compressedNeighbors);
    }

    /**
     * State required for writing fused ADC data for a single node.
     */
    public static class State implements Feature.State {
        /** The graph view providing access to node neighbors */
        public final ImmutableGraphIndex.View view;
        /** The PQ-encoded vectors for all nodes in the graph */
        public final PQVectors pqVectors;
        /** The ordinal ID of the node being written */
        public final int nodeId;

        /**
         * Creates state for writing FusedADC data for the specified node.
         *
         * @param view the graph view providing access to node neighbors
         * @param pqVectors the PQ-encoded vectors for all nodes
         * @param nodeId the ordinal ID of the node being written
         */
        public State(ImmutableGraphIndex.View view, PQVectors pqVectors, int nodeId) {
            this.view = view;
            this.pqVectors = pqVectors;
            this.nodeId = nodeId;
        }
    }

    /**
     * Provides access to PQ-encoded neighbor vectors stored in transposed format on disk.
     * This enables efficient batch similarity computations using SIMD instructions.
     */
    public class PackedNeighbors {
        private final OnDiskGraphIndex.View view;

        /**
         * Creates a PackedNeighbors accessor for the given graph view.
         *
         * @param view the on-disk graph index view
         */
        public PackedNeighbors(OnDiskGraphIndex.View view) {
            this.view = view;
        }

        /**
         * Retrieves the transposed PQ-encoded neighbors for the specified node from disk.
         *
         * @param node the node ordinal ID
         * @return a byte sequence containing the transposed PQ codes of all neighbors
         */
        public ByteSequence<?> getPackedNeighbors(int node) {
            try {
                var reader = view.featureReaderForNode(node, FeatureId.FUSED_ADC);
                var tlNeighbors = reusableNeighbors.get();
                vectorTypeSupport.readByteSequence(reader, tlNeighbors);
                return tlNeighbors;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        /**
         * Returns the maximum degree configured for this FusedADC instance.
         *
         * @return the maximum number of neighbors per node
         */
        public int maxDegree() {
            return maxDegree;
        }
    }
}

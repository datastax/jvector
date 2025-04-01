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
import io.github.jbellis.jvector.graph.GraphIndex;
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

        var neighbors = state.view.getNeighborsIterator(0, state.nodeId); // TODO
        int n = 0;
        var neighborSize = neighbors.size();
        compressedNeighbors.zero();
        for (; n < neighborSize; n++) {
            var compressed = pqv.get(neighbors.nextInt());
            for (int j = 0; j < pqv.getCompressedSize(); j++) {
                compressedNeighbors.set(j * maxDegree + n, compressed.get(j));
            }
        }

        vectorTypeSupport.writeByteSequence(out, compressedNeighbors);
    }

    public static class State implements Feature.State {
        public final GraphIndex.View view;
        public final PQVectors pqVectors;
        public final int nodeId;

        public State(GraphIndex.View view, PQVectors pqVectors, int nodeId) {
            this.view = view;
            this.pqVectors = pqVectors;
            this.nodeId = nodeId;
        }
    }

    public class PackedNeighbors {
        private final OnDiskGraphIndex.View view;

        public PackedNeighbors(OnDiskGraphIndex.View view) {
            this.view = view;
        }

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

        public int maxDegree() {
            return maxDegree;
        }
    }
}

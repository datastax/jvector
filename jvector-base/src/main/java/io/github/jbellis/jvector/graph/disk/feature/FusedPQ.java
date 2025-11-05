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
import io.github.jbellis.jvector.quantization.FusedPQDecoder;
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
import java.util.function.IntFunction;

/**
 * Implements scoring by fusing PQ-encoded neighbors into an OnDiskGraphIndex.
 */
public class FusedPQ extends AbstractFeature implements FusedFeature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ProductQuantization pq;
    private final int maxDegree;
    private final ThreadLocal<VectorFloat<?>> reusableResults;
    private final ExplicitThreadLocal<ByteSequence<?>> reusableNeighborCodes;
    private final ExplicitThreadLocal<ByteSequence<?>> pqCodeScratch;

    public FusedPQ(int maxDegree, ProductQuantization pq) {
        if (pq.getClusterCount() != 256) {
            throw new IllegalArgumentException("FusedPQ requires a 256-cluster PQ. This limitation may be removed in future releases");
        }
        this.maxDegree = maxDegree;
        this.pq = pq;
        this.reusableResults = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(maxDegree));
        this.reusableNeighborCodes = ExplicitThreadLocal.withInitial(() -> vectorTypeSupport.createByteSequence(pq.compressedVectorSize() * maxDegree));
        this.pqCodeScratch = ExplicitThreadLocal.withInitial(() -> vectorTypeSupport.createByteSequence(pq.compressedVectorSize()));
    }

    @Override
    public FeatureId id() {
        return FeatureId.FUSED_PQ;
    }

    public ProductQuantization getPQ() {
        return pq;
    }

    @Override
    public int headerSize() {
        return pq.compressorSize();
    }

    @Override
    public int featureSize() {
        return pq.compressedVectorSize() * maxDegree;
    }

    static FusedPQ load(CommonHeader header, RandomAccessReader reader) {
        try {
            return new FusedPQ(header.layerInfo.get(0).degree, ProductQuantization.load(reader));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * @param view The view needs to be the one used by the searcher
     * @param esf
     * @return
     */
    public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf, OnDiskGraphIndex.View view, ScoreFunction.ExactScoreFunction esf) {
        var neighbors = new PackedNeighbors(view);
        var hierarchyCachedFeatures = view.getInlineSourceFeatures();
        return FusedPQDecoder.newDecoder(neighbors, pq, hierarchyCachedFeatures, queryVector, reusableNeighborCodes.get(), reusableResults.get(), vsf, esf);
    }

    @Override
    public void writeHeader(DataOutput out) throws IOException {
        pq.write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    // this is an awkward fit for the Feature.State design since we need to
    // generate the fused set based on the neighbors of the node, not just the node itself
    @Override
    public void writeInline(DataOutput out, Feature.State state_) throws IOException {
        var state = (FusedPQ.State) state_;

        var neighbors = state.view.getNeighborsIterator(0, state.nodeId);
        int count = 0;
        while (neighbors.hasNext()) {
            int node = neighbors.nextInt();
            var compressed = state.compressedVectorFunction.apply(node);
            vectorTypeSupport.writeByteSequence(out, compressed.copy());
            count++;
        }
        pqCodeScratch.get().zero();
        for (; count < maxDegree; count++) {
            vectorTypeSupport.writeByteSequence(out, pqCodeScratch.get());
        }
    }

    public static class State implements Feature.State {
        public final ImmutableGraphIndex.View view;
        public final IntFunction<ByteSequence<?>> compressedVectorFunction;
        public final int nodeId;

        public State(ImmutableGraphIndex.View view, PQVectors pqVectors, int nodeId) {
            this(view, pqVectors::get, nodeId);
        }

        public State(ImmutableGraphIndex.View view, IntFunction<ByteSequence<?>> compressedVectorFunction, int nodeId) {
            this.view = view;
            this.compressedVectorFunction = compressedVectorFunction;
            this.nodeId = nodeId;
        }
    }

    @Override
    public void writeSourceFeature(DataOutput out, Feature.State state_) throws IOException {
        var state = (FusedPQ.State) state_;
        var compressed = state.compressedVectorFunction.apply(state.nodeId);
        var temp = pqCodeScratch.get();
        for (int i = 0; i < compressed.length(); i++) {
            temp.set(i, compressed.get(i));
        }
        vectorTypeSupport.writeByteSequence(out, temp);
    }

    public static class FusedPQInlineSource implements InlineSource {
        private ByteSequence<?> code;

        public FusedPQInlineSource(ByteSequence<?> code) {
            this.code = code;
        }

        @Override
        public long ramBytesUsed() {
            return code.length();
        }

        public ByteSequence<?> getCode() {
            return code;
        }
    }

    @Override
    public InlineSource loadSourceFeature(RandomAccessReader in) throws IOException {
        int length = pq.getSubspaceCount();
        var code = vectorTypeSupport.createByteSequence(length);
        vectorTypeSupport.readByteSequence(in, code);
        return new FusedPQInlineSource(code);
    }

    public class PackedNeighbors {
        private final OnDiskGraphIndex.View view;

        public PackedNeighbors(OnDiskGraphIndex.View view) {
            this.view = view;
        }

        public void readInto(int node, ByteSequence<?> neighborCodes) {
            try {
                view.getPackedNeighbors(node, FeatureId.FUSED_PQ,
                        reader -> {
                            try {
                                vectorTypeSupport.readByteSequence(reader, neighborCodes);
                            } catch (IOException e) {
                                throw new RuntimeException(e);
                            }
                        }
                );
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        public int maxDegree() {
            return maxDegree;
        }
    }
}

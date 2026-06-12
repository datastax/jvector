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

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.FusedASHDecoder;
import io.github.jbellis.jvector.quantization.FusedASHLayout;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.function.IntFunction;

/**
 * Fuses ASH-encoded neighbors into an {@link OnDiskGraphIndex} for block-oriented
 * graph-neighborhood scoring.
 *
 * <p>Unlike regular {@link ASHVectors}, this feature stores each node's layer-0
 * neighbor codes in a neighborhood-local block layout. The decoder reads and
 * scores the whole neighborhood in {@code enableSimilarityToNeighbors(origin)},
 * then {@code similarityToNeighbor(origin, i)} is a cheap score lookup.</p>
 *
 * <p>The fused layout supports only {@code bitsPerDimension in {1,2,4}} because
 * it is based on 4-bit projection groups and a 16-entry query LUT per group.
 * Core standalone ASH may support additional bit widths, but those are not part
 * of the fused graph hot path.</p>
 */
public class FusedASH extends AbstractFeature implements FusedFeature {
    private final int maxDegree;
    private final int blockSize;
    private final AsymmetricHashing ash;
    private final int featureSize;

    private final ThreadLocal<byte[]> reusableNeighborhoodBytes;
    private final ThreadLocal<float[]> reusableScores;
    private final ThreadLocal<byte[]> writeScratch;

    public FusedASH(int maxDegree, AsymmetricHashing ash) {
        this(maxDegree, ash, FusedASHLayout.chooseBlockSize(maxDegree));
    }

    public FusedASH(int maxDegree, AsymmetricHashing ash, int blockSize) {
        if (maxDegree <= 0) {
            throw new IllegalArgumentException("maxDegree must be > 0");
        }
        if (ash == null) {
            throw new NullPointerException("ash");
        }
        FusedASHLayout.validateBitsPerDimension(ash.bitsPerDimension);
        FusedASHLayout.validateBlockSize(blockSize);

        this.maxDegree = maxDegree;
        this.ash = ash;
        this.blockSize = blockSize;
        this.featureSize = FusedASHLayout.featureSize(
                maxDegree,
                ash.quantizedDim,
                ash.bitsPerDimension,
                blockSize);

        this.reusableNeighborhoodBytes = ThreadLocal.withInitial(() -> new byte[featureSize]);
        this.reusableScores = ThreadLocal.withInitial(() -> new float[maxDegree]);
        this.writeScratch = ThreadLocal.withInitial(() -> new byte[featureSize]);
    }

    @Override
    public FeatureId id() {
        return FeatureId.FUSED_ASH;
    }

    public AsymmetricHashing getASH() {
        return ash;
    }

    public int blockSize() {
        return blockSize;
    }

    @Override
    public int headerSize() {
        return ash.compressorSize() + Integer.BYTES; // blockSize
    }

    @Override
    public int featureSize() {
        return featureSize;
    }

    static FusedASH load(CommonHeader header, RandomAccessReader reader) {
        try {
            AsymmetricHashing ash = AsymmetricHashing.load(reader);
            int blockSize = reader.readInt();
            return new FusedASH(header.layerInfo.get(0).degree, ash, blockSize);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void writeHeader(IndexWriter out) throws IOException {
        ash.write(out, OnDiskGraphIndex.CURRENT_VERSION);
        out.writeInt(blockSize);
    }

    /**
     * Writes a fused, block-packed ASH payload for {@code state.nodeId}'s layer-0
     * neighbors. Neighbor lanes are stored in graph neighbor iterator order and
     * padded with zeros up to {@code maxDegree}.
     */
    @Override
    public void writeInline(IndexWriter out, Feature.State state_) throws IOException {
        var state = (FusedASH.State) state_;
        byte[] scratch = writeScratch.get();
        Arrays.fill(scratch, (byte) 0);

        var neighbors = state.view.getNeighborsIterator(0, state.nodeId);
        int count = 0;
        while (neighbors.hasNext()) {
            int node = neighbors.nextInt();
            if (count >= maxDegree) {
                throw new IOException("Node " + state.nodeId + " has more than maxDegree=" + maxDegree + " neighbors");
            }

            var compressed = state.compressedVectorFunction.apply(node);
            packForFused(scratch, count, compressed);
            count++;
        }

        writeBytes(out, scratch, featureSize);
    }

    private void packForFused(byte[] dest, int neighborIndex, AsymmetricHashing.QuantizedVector vector) {
        int blockIndex = neighborIndex / blockSize;
        int lane = neighborIndex % blockSize;
        int blockOffset = FusedASHLayout.blockOffset(
                blockIndex,
                ash.quantizedDim,
                ash.bitsPerDimension,
                blockSize);

        if (ash.bitsPerDimension == 1) {
            packOneBitForFused(dest, blockOffset, lane, vector);
        } else {
            // For 2-bit and 4-bit ASH, the canonical vector already stores the
            // C++ projection-mode adjusted offset used by FusedASH scoring.
            FusedASHLayout.packQuantizedVector(
                    dest,
                    blockOffset,
                    lane,
                    vector,
                    ash.quantizedDim,
                    ash.bitsPerDimension,
                    blockSize);
        }
    }

    /**
     * The regular 1-bit ASH vector keeps the legacy unadjusted offset, because
     * its standalone scorer uses A(q - μ). FusedASH scores via Aq for every
     * supported bit width, so we apply the projection-offset adjustment here.
     */
    private void packOneBitForFused(byte[] dest, int blockOffset, int lane, AsymmetricHashing.QuantizedVector vector) {
        int groups = FusedASHLayout.codeGroups(ash.quantizedDim, ash.bitsPerDimension);
        for (int group = 0; group < groups; group++) {
            int nibble = FusedASHLayout.signNibbleFromWords(vector.binaryVector, group, ash.quantizedDim);
            FusedASHLayout.setPackedNibble(dest, blockOffset, lane, group, blockSize, nibble);
        }

        int c = vector.landmark & 0xFF;
        if (c >= ash.landmarkCount) {
            throw new IllegalArgumentException("Invalid ASH landmark " + c + " for landmarkCount=" + ash.landmarkCount);
        }

        float landmarkDot = dotOneBitProjectionCode(ash.landmarkProj[c], vector.binaryVector, ash.quantizedDim);
        float adjustedOffset = vector.offset - vector.scale * landmarkDot;

        FusedASHLayout.writeLaneHeader(
                dest,
                blockOffset,
                lane,
                ash.quantizedDim,
                ash.bitsPerDimension,
                blockSize,
                vector.scale,
                adjustedOffset,
                vector.landmark);
    }

    private static float dotOneBitProjectionCode(float[] projected, long[] signWords, int quantizedDim) {
        float sum = 0f;
        for (int j = 0; j < quantizedDim; j++) {
            boolean positive = ((signWords[j >>> 6] >>> (j & 63)) & 1L) != 0L;
            sum += positive ? projected[j] : -projected[j];
        }
        return sum;
    }

    private static void writeBytes(IndexWriter out, byte[] bytes, int length) throws IOException {
        for (int i = 0; i < length; i++) {
            out.writeByte(bytes[i]);
        }
    }

    public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(
            VectorFloat<?> queryVector,
            VectorSimilarityFunction vsf,
            OnDiskGraphIndex.View view,
            ScoreFunction.ExactScoreFunction esf) {

        var neighborhoods = new PackedNeighborhoods(view);
        var hierarchyCachedFeatures = view.getInlineSourceFeatures();

        return FusedASHDecoder.newDecoder(
                neighborhoods,
                ash,
                hierarchyCachedFeatures,
                source -> ((FusedASHInlineSource) source).getVector(),
                queryVector,
                reusableNeighborhoodBytes.get(),
                reusableScores.get(),
                blockSize,
                vsf);
    }

    @Override
    public void writeSourceFeature(IndexWriter out, Feature.State state_) throws IOException {
        var state = (FusedASH.State) state_;
        var compressed = state.compressedVectorFunction.apply(state.nodeId);
        compressed.write(out, ash.quantizedDim, ash.bitsPerDimension);
    }

    @Override
    public InlineSource loadSourceFeature(RandomAccessReader in) throws IOException {
        var vector = AsymmetricHashing.QuantizedVector.load(in, ash.quantizedDim, ash.bitsPerDimension);
        return new FusedASHInlineSource(vector);
    }

    public class FusedASHInlineSource implements InlineSource {
        private final AsymmetricHashing.QuantizedVector vector;

        FusedASHInlineSource(AsymmetricHashing.QuantizedVector vector) {
            this.vector = vector;
        }

        public AsymmetricHashing.QuantizedVector getVector() {
            return vector;
        }

        @Override
        public long ramBytesUsed() {
            return ash.compressedVectorSize();
        }
    }

    public class PackedNeighborhoods implements FusedASHDecoder.PackedNeighborhoods {
        private final OnDiskGraphIndex.View view;

        PackedNeighborhoods(OnDiskGraphIndex.View view) {
            this.view = view;
        }

        @Override
        public void readInto(int origin, byte[] dest) {
            if (dest.length < featureSize) {
                throw new IllegalArgumentException("Destination buffer is smaller than FusedASH feature size");
            }

            try {
                view.getPackedNeighbors(origin, FeatureId.FUSED_ASH, reader -> {
                    try {
                        reader.readFully(dest);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public int maxDegree() {
            return maxDegree;
        }

        @Override
        public int featureSize() {
            return featureSize;
        }
    }

    public static class State implements Feature.State {
        public final ImmutableGraphIndex.View view;
        public final IntFunction<AsymmetricHashing.QuantizedVector> compressedVectorFunction;
        public final int nodeId;

        public State(ImmutableGraphIndex.View view, ASHVectors ashVectors, int nodeId) {
            this(view, ashVectors::get, nodeId);
        }

        public State(
                ImmutableGraphIndex.View view,
                IntFunction<AsymmetricHashing.QuantizedVector> compressedVectorFunction,
                int nodeId) {
            this.view = view;
            this.compressedVectorFunction = compressedVectorFunction;
            this.nodeId = nodeId;
        }
    }
}

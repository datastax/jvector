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
import io.github.jbellis.jvector.graph.disk.CommonHeader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.quantization.NVQScorer;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.NVQuantization.QuantizedVector;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Implements the storage of NuVeQ vectors in an on-disk graph index.  These can be used for reranking.
 */
public class NVQ implements Feature {
    private final NVQuantization nvq;
    private final NVQScorer scorer;
    private final ThreadLocal<QuantizedVector> reusableQuantizedVector;

    /**
     * Constructs an NVQ feature with the given NVQuantization compressor.
     *
     * @param nvq the NVQuantization instance to use for encoding/decoding
     */
    public NVQ(NVQuantization nvq) {
        this.nvq = nvq;
        scorer = new NVQScorer(this.nvq);
        reusableQuantizedVector = ThreadLocal.withInitial(() -> NVQuantization.QuantizedVector.createEmpty(nvq.subvectorSizesAndOffsets, nvq.bitsPerDimension));
    }

    @Override
    public FeatureId id() {
        return FeatureId.NVQ_VECTORS;
    }

    @Override
    public int headerSize() {
        return nvq.compressorSize();
    }

    /**
     * Returns the size in bytes of a single NVQ-quantized vector.
     *
     * @return the feature size in bytes
     */
    @Override
    public int featureSize() { return nvq.compressedVectorSize();}

    /**
     * Returns the dimensionality of the original uncompressed vectors.
     *
     * @return the vector dimension
     */
    public int dimension() {
        return nvq.globalMean.length();
    }

    /**
     * Loads an NVQ feature from a reader.
     *
     * @param header the common header (unused but required by signature)
     * @param reader the reader to load from
     * @return the loaded NVQ feature
     */
    static NVQ load(CommonHeader header, RandomAccessReader reader) {
        try {
            return new NVQ(NVQuantization.load(reader));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void writeHeader(DataOutput out) throws IOException {
        nvq.write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    @Override
    public void writeInline(DataOutput out, Feature.State state_) throws IOException {
        var state = (NVQ.State) state_;
        state.vector.write(out);
    }

    /**
     * Represents the state of an NVQ-quantized vector for a single node.
     */
    public static class State implements Feature.State {
        /**
         * The quantized vector.
         */
        public final QuantizedVector vector;

        /**
         * Constructs a State with the given quantized vector.
         *
         * @param vector the quantized vector
         */
        public State(QuantizedVector vector) {
            this.vector = vector;
        }
    }

    /**
     * Creates a reranking score function that loads NVQ vectors from disk and computes exact scores.
     *
     * @param queryVector the query vector
     * @param vsf the vector similarity function to use
     * @param source the source to read NVQ vectors from
     * @return an exact score function for reranking
     */
    public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector,
                                                        VectorSimilarityFunction vsf,
                                                        FeatureSource source) {
        var function = scorer.scoreFunctionFor(queryVector, vsf);

        return node2 -> {
            try {
                var reader = source.featureReaderForNode(node2, FeatureId.NVQ_VECTORS);
                QuantizedVector.loadInto(reader, reusableQuantizedVector.get());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            return function.similarityTo(reusableQuantizedVector.get());
        };
    }
}

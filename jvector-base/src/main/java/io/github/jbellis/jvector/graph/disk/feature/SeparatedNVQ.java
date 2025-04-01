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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;

public class SeparatedNVQ implements SeparatedFeature {
    private final NVQuantization nvq;
    private final NVQScorer scorer;
    private final ThreadLocal<NVQuantization.QuantizedVector> reusableQuantizedVector;
    private long offset;

    public SeparatedNVQ(NVQuantization nvq, long offset) {
        this.nvq = nvq;
        this.offset = offset;
        scorer = new NVQScorer(this.nvq);
        reusableQuantizedVector = ThreadLocal.withInitial(() -> NVQuantization.QuantizedVector.createEmpty(nvq.subvectorSizesAndOffsets, nvq.bitsPerDimension));
    }

    @Override
    public void setOffset(long offset) {
        this.offset = offset;
    }

    @Override
    public long getOffset() {
        return offset;
    }

    @Override
    public FeatureId id() {
        return FeatureId.SEPARATED_NVQ;
    }

    @Override
    public int headerSize() {
        return nvq.compressorSize() + Long.BYTES;
    }

    @Override
    public int featureSize() {
        return nvq.compressedVectorSize();
    }

    @Override
    public void writeHeader(DataOutput out) throws IOException {
        nvq.write(out, OnDiskGraphIndex.CURRENT_VERSION);
        out.writeLong(offset);
    }

    @Override
    public void writeSeparately(DataOutput out, State state_) throws IOException {
        var state = (NVQ.State) state_;
        if (state.vector != null) {
            state.vector.write(out);
        } else {
            // Write zeros for missing vector
            NVQuantization.QuantizedVector.createEmpty(nvq.subvectorSizesAndOffsets, nvq.bitsPerDimension).write(out);
        }
    }

    // Using NVQ.State

    static SeparatedNVQ load(CommonHeader header, RandomAccessReader reader) {
        try {
            var nvq = NVQuantization.load(reader);
            long offset = reader.readLong();
            return new SeparatedNVQ(nvq, offset);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public int dimension() {
        return nvq.globalMean.length();
    }

    ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector,
                                                VectorSimilarityFunction vsf,
                                                FeatureSource source) {
        var function = scorer.scoreFunctionFor(queryVector, vsf);

        return node2 -> {
            try {
                var reader = source.featureReaderForNode(node2, FeatureId.SEPARATED_NVQ);
                NVQuantization.QuantizedVector.loadInto(reader, reusableQuantizedVector.get());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            return function.similarityTo(reusableQuantizedVector.get());
        };
    }
}

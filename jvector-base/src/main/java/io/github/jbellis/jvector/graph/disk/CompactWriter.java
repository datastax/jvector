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

package io.github.jbellis.jvector.graph.disk;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import static io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor.SelectedVecCache;
import static io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor.WriteResult;

final class CompactWriter implements AutoCloseable {

    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static final int FOOTER_MAGIC = 0x4a564244;
    private static final int FOOTER_OFFSET_SIZE = Long.BYTES;
    private static final int FOOTER_MAGIC_SIZE = Integer.BYTES;
    private static final int FOOTER_SIZE = FOOTER_MAGIC_SIZE + FOOTER_OFFSET_SIZE;

    private final RandomAccessWriter writer;
    private final int recordSize;
    private final long startOffset;
    private final int headerSize;
    private final Header header;
    private final int version;
    private final FusedPQ fusedPQFeature;
    private final ProductQuantization pq;
    private final int baseDegree;
    private final int maxOrdinal;
    private final ThreadLocal<ByteBuffer> bufferPerThread;
    private final ThreadLocal<ByteSequence<?>> zeroPQ;
    private final boolean fusedPQEnabled;
    private final Path outputPath;
    private final List<CommonHeader.LayerInfo> configuredLayerInfo;
    private final List<Integer> configuredLayerDegrees;
    private final List<UpperLayerFeatureRecord> level1FeatureRecords;

    CompactWriter(Path outputPath,
                  int maxOrdinal,
                  int numBaseLayerNodes,
                  long startOffset,
                  List<CommonHeader.LayerInfo> layerInfo,
                  int entryNode,
                  int dimension,
                  List<Integer> layerDegrees,
                  ProductQuantization pq,
                  int pqLength,
                  boolean fusedPQEnabled)
            throws IOException {
        this.fusedPQEnabled = fusedPQEnabled;
        this.version = OnDiskGraphIndex.CURRENT_VERSION;
        this.outputPath = outputPath;
        this.writer = new BufferedRandomAccessWriter(outputPath);
        this.startOffset = startOffset;
        this.configuredLayerInfo = new ArrayList<>(layerInfo);
        this.configuredLayerDegrees = new ArrayList<>(layerDegrees);
        this.baseDegree = layerDegrees.get(0);
        this.pq = pq;
        this.maxOrdinal = maxOrdinal;
        this.level1FeatureRecords = new ArrayList<>();

        Map<FeatureId, Feature> featureMap = new LinkedHashMap<>();
        InlineVectors inlineVectorFeature = new InlineVectors(dimension);
        featureMap.put(FeatureId.INLINE_VECTORS, inlineVectorFeature);
        if (fusedPQEnabled) {
            this.fusedPQFeature = new FusedPQ(Collections.max(layerDegrees), pq);
            featureMap.put(FeatureId.FUSED_PQ, this.fusedPQFeature);
        } else {
            this.fusedPQFeature = null;
        }

        int rsize = Integer.BYTES + inlineVectorFeature.featureSize() + Integer.BYTES + baseDegree * Integer.BYTES;
        if (fusedPQEnabled) {
            rsize += fusedPQFeature.featureSize();
        }
        this.recordSize = rsize;

        this.configuredLayerInfo.set(0, new CommonHeader.LayerInfo(numBaseLayerNodes, baseDegree));
        var commonHeader = new CommonHeader(this.version, dimension, entryNode, this.configuredLayerInfo, this.maxOrdinal + 1);
        this.header = new Header(commonHeader, featureMap);
        this.headerSize = header.size();

        this.bufferPerThread = ThreadLocal.withInitial(() -> {
            ByteBuffer buffer = ByteBuffer.allocate(recordSize);
            buffer.order(ByteOrder.BIG_ENDIAN);
            return buffer;
        });
        this.zeroPQ = ThreadLocal.withInitial(() -> {
            var vec = vectorTypeSupport.createByteSequence(pqLength > 0 ? pqLength : 1);
            vec.zero();
            return vec;
        });
    }

    public void writeHeader() throws IOException {
        writer.seek(startOffset);
        header.write(writer);
        assert writer.position() == startOffset + headerSize : String.format("%d != %d", writer.position(), startOffset + headerSize);
        writer.flush();
    }

    void writeFooter() throws IOException {
        if (fusedPQEnabled && version == 6 && !level1FeatureRecords.isEmpty()) {
            for (UpperLayerFeatureRecord record : level1FeatureRecords) {
                writer.writeInt(record.ordinal);
                vectorTypeSupport.writeByteSequence(writer, record.pqCode);
            }
        }
        long headerOffset = writer.position();
        header.write(writer);
        writer.writeLong(headerOffset);
        writer.writeInt(FOOTER_MAGIC);
        final long expectedPosition = headerOffset + headerSize + FOOTER_SIZE;
        assert writer.position() == expectedPosition : String.format("%d != %d", writer.position(), expectedPosition);
    }

    public void offsetAfterInline() throws IOException {
        long offset = startOffset + headerSize + (long) (maxOrdinal + 1) * recordSize;
        writer.seek(offset);
    }

    public Path getOutputPath() {
        return outputPath;
    }

    public void writeUpperLayerNode(int level, int ordinal, int[] neighbors, ByteSequence<?> level1PqCode) throws IOException {
        writer.writeInt(ordinal);
        writer.writeInt(neighbors.length);
        int degree = configuredLayerDegrees.get(level);
        int n = 0;
        for (; n < neighbors.length; n++) {
            writer.writeInt(neighbors[n]);
        }
        for (; n < degree; n++) {
            writer.writeInt(-1);
        }
        if (fusedPQEnabled && version == 6 && level == 1 && level1PqCode != null) {
            level1FeatureRecords.add(new UpperLayerFeatureRecord(ordinal, level1PqCode.copy()));
        }
    }

    public void close() throws IOException {
        final var endOfGraphPosition = writer.position();
        writer.seek(endOfGraphPosition);
        writer.flush();
    }

    public WriteResult writeInlineNodeRecord(int ordinal, VectorFloat<?> vec, SelectedVecCache selectedCache, ByteSequence<?> pqCode) throws IOException
    {
        var bwriter = new ByteBufferIndexWriter(bufferPerThread.get());

        long fileOffset = startOffset + headerSize + (long) ordinal * recordSize;
        bwriter.reset();
        bwriter.writeInt(ordinal);

        for(int i = 0; i < vec.length(); ++i) {
            bwriter.writeFloat(vec.get(i));
        }

        // write fused PQ
        // since we build a graph in a streaming way,
        // we cannot use fusedPQfeature.writeInline
        if (fusedPQEnabled) {
            int k = 0;
            for (; k < selectedCache.size; k++) {
                pqCode.zero();
                pq.encodeTo(selectedCache.vecs[k], pqCode);
                vectorTypeSupport.writeByteSequence(bwriter, pqCode);
            }
            for (; k < baseDegree; k++) {
                vectorTypeSupport.writeByteSequence(bwriter, zeroPQ.get());
            }
        }

        // write neighbors list
        bwriter.writeInt(selectedCache.size);
        int n = 0;
        for (; n < selectedCache.size; n++) {
            bwriter.writeInt(selectedCache.nodes[n]);
        }

        // pad out to base layer degree
        for (; n < baseDegree; n++) {
            bwriter.writeInt(-1);
        }

        if (bwriter.bytesWritten() != recordSize) {
            throw new IllegalStateException(
                    String.format("Record size mismatch for ordinal %d: expected %d bytes, wrote %d bytes, base degree: %d",
                            ordinal, recordSize, bwriter.bytesWritten(), baseDegree));
        }

        ByteBuffer dataCopy = bwriter.cloneBuffer();

        return new WriteResult(ordinal, fileOffset, dataCopy);
    }

    static final class UpperLayerFeatureRecord {
        final int ordinal;
        final ByteSequence<?> pqCode;

        UpperLayerFeatureRecord(int ordinal, ByteSequence<?> pqCode) {
            this.ordinal = ordinal;
            this.pqCode = pqCode;
        }
    }
}

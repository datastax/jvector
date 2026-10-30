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
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

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
    private final FusedFeature fusedFeature;
    private final int baseDegree;
    private final int maxOrdinal;
    private final int entryNode;
    // Vector components stored per record: the full dimension, or 0 for a topology-only output.
    private final int storedDimension;
    private final ThreadLocal<ByteBuffer> bufferPerThread;
    private final ThreadLocal<ByteSequence<?>> zeroPQ;
    private final boolean fusedPQEnabled;
    // The output file, or null when the graph is streamed to a caller's IndexWriter.
    private final Path outputPath;
    // Where the strategies map their transient sections: past the end of the output file, or at
    // the start of a separate scratch file when the graph is streamed.
    private final Path scratchPath;
    private final List<CommonHeader.LayerInfo> configuredLayerInfo;
    private final List<Integer> configuredLayerDegrees;
    private final List<UpperLayerFeatureRecord> level1FeatureRecords;
    // The output ordinal of every merged ordinal, when the caller chose them; null when the output is numbered by the
    // merged ordinals. Every ordinal handed to this writer is a merged one and is translated here, on its way out.
    private final int[] outputOrdinals;
    // PQ code for the entry node, required when hierarchy is disabled (no level 1).
    // Mirrors what AbstractGraphIndexWriter.writeSparseLevels writes in the no-hierarchy branch.
    private ByteSequence<?> entryNodePqCode;

    // Optional pre-computed PQ codes by new ordinal. When set, writeInlineNodeRecord copies
    // codes from the cache instead of calling pq.encodeTo per neighbor. The cache handles
    // per-thread view duplication internally, so positions don't race.
    private PreEncodedCodeCache pqCodeCache;
    private int pqCodeSize;
    private ThreadLocal<byte[]> pqCodeBufPerThread;

    /**
     * Writes the graph to {@code outputPath}, which also holds the strategies' transient sections past
     * its projected end until the compactor truncates them.
     *
     * @param storedDimension the vector components each record stores, which is also the dimension
     *                        the header declares: the vectors' dimension, or 0 when the sources were
     *                        written topology-only and the output must stay so
     * @param outputOrdinals  the output ordinal of every merged ordinal, or null to number the output by
     *                        the merged ordinals
     */
    CompactWriter(Path outputPath,
                  int maxOrdinal,
                  int numBaseLayerNodes,
                  long startOffset,
                  List<CommonHeader.LayerInfo> layerInfo,
                  int entryNode,
                  int storedDimension,
                  List<Integer> layerDegrees,
                  FusedFeature fusedFeature,
                  int[] outputOrdinals)
            throws IOException {
        this(new BufferedRandomAccessWriter(outputPath), outputPath, outputPath, startOffset,
             maxOrdinal, numBaseLayerNodes, layerInfo, entryNode, storedDimension, layerDegrees, fusedFeature, outputOrdinals);
    }

    /**
     * Writes the graph strictly sequentially to {@code output}, starting at its current position, and
     * leaves it open. The strategies' transient sections go to {@code scratchPath} instead, which the
     * compactor deletes when it is done.
     */
    CompactWriter(IndexWriter output,
                  Path scratchPath,
                  int maxOrdinal,
                  int numBaseLayerNodes,
                  List<CommonHeader.LayerInfo> layerInfo,
                  int entryNode,
                  int storedDimension,
                  List<Integer> layerDegrees,
                  FusedFeature fusedFeature,
                  int[] outputOrdinals)
            throws IOException {
        this(new SequentialWriter(output), null, scratchPath, output.position(),
             maxOrdinal, numBaseLayerNodes, layerInfo, entryNode, storedDimension, layerDegrees, fusedFeature, outputOrdinals);
    }

    private CompactWriter(RandomAccessWriter writer,
                          Path outputPath,
                          Path scratchPath,
                          long startOffset,
                          int maxOrdinal,
                          int numBaseLayerNodes,
                          List<CommonHeader.LayerInfo> layerInfo,
                          int entryNode,
                          int storedDimension,
                          List<Integer> layerDegrees,
                          FusedFeature fusedFeature,
                          int[] outputOrdinals) {
        this.outputOrdinals = outputOrdinals;
        entryNode = outputOrdinal(entryNode);
        this.fusedFeature = fusedFeature;
        this.fusedPQEnabled = fusedFeature != null;
        // This writer only ever targets the current on-disk format -- unlike AbstractGraphIndexWriter,
        // there is no builder/version parameter that lets a caller pick an older version. Because of
        // that, `fusedPQEnabled` alone is a safe stand-in for "the format supports fused PQ" everywhere
        // below: there's no other version this class could be writing. We still check the format here,
        // once, instead of just assuming it, so that if this class is ever changed to support writing
        // older versions, or a future version drops fused PQ support, this fails loudly at construction
        // instead of silently omitting the fused-PQ footer/packed-neighbor data later on.
        int version = OnDiskGraphIndex.CURRENT_VERSION;
        if (fusedPQEnabled && !GraphIndexFormatFactory.forVersion(version).supportsFeature(FeatureId.FUSED_PQ)) {
            throw new IllegalStateException(
                    "Fused PQ requires a format version that supports it; version " + version + " does not");
        }
        this.outputPath = outputPath;
        this.scratchPath = scratchPath;
        this.writer = writer;
        this.startOffset = startOffset;
        this.configuredLayerInfo = new ArrayList<>(layerInfo);
        this.configuredLayerDegrees = new ArrayList<>(layerDegrees);
        this.baseDegree = layerDegrees.get(0);
        this.maxOrdinal = maxOrdinal;
        this.entryNode = entryNode;
        this.storedDimension = storedDimension;
        this.level1FeatureRecords = new ArrayList<>();
        this.entryNodePqCode = null;

        Map<FeatureId, Feature> featureMap = new LinkedHashMap<>();
        InlineVectors inlineVectorFeature = new InlineVectors(storedDimension);
        featureMap.put(FeatureId.INLINE_VECTORS, inlineVectorFeature);
        if (fusedFeature != null) {
            featureMap.put(fusedFeature.id(), fusedFeature);
        }

        int rsize = Integer.BYTES + inlineVectorFeature.featureSize() + Integer.BYTES + baseDegree * Integer.BYTES;
        if (fusedFeature != null) {
            rsize += fusedFeature.featureSize();
        }
        this.recordSize = rsize;

        this.configuredLayerInfo.set(0, new CommonHeader.LayerInfo(numBaseLayerNodes, baseDegree));
        var commonHeader = new CommonHeader(version, storedDimension, entryNode, this.configuredLayerInfo, this.maxOrdinal + 1);
        this.header = new Header(commonHeader, featureMap);
        this.headerSize = header.size();

        final int codeSize = fusedFeature != null ? fusedFeature.codeSize() : 1;
        this.bufferPerThread = ThreadLocal.withInitial(() -> {
            ByteBuffer buffer = ByteBuffer.allocate(recordSize);
            buffer.order(ByteOrder.BIG_ENDIAN);
            return buffer;
        });
        this.zeroPQ = ThreadLocal.withInitial(() -> {
            var vec = vectorTypeSupport.createByteSequence(codeSize);
            vec.zero();
            return vec;
        });
    }

    /**
     * Enables the pre-computed PQ code cache. Must be called before any writeInlineNodeRecord
     * call. Once enabled, neighbor PQ codes are copied from {@code cache} instead of being
     * re-encoded per write.
     *
     * @param cache       a cache holding one code per new ordinal, for ordinals
     *                    {@code 0..maxOrdinal}
     * @param pqCodeSize  bytes per code (== FusedFeature.codeSize() of the source's feature)
     */
    public void enablePqCodeCache(PreEncodedCodeCache cache, int pqCodeSize) {
        this.pqCodeCache = cache;
        this.pqCodeSize = pqCodeSize;
        this.pqCodeBufPerThread = ThreadLocal.withInitial(() -> new byte[pqCodeSize]);
    }

    /** The pre-encoded code cache, or null when not enabled. Codes are keyed by new ordinal. */
    PreEncodedCodeCache pqCodeCache() {
        return pqCodeCache;
    }

    /**
     * Whether {@link #writeInlineNodeRecord} reads {@code selectedCache.vecs} — true only for
     * fused output without the pre-encoded code cache, where neighbor codes are produced by
     * encoding the neighbor vectors. Callers that can supply neighbor ids without vectors
     * (the retained-only fast path) use this to skip the vector reads.
     */
    boolean needsNeighborVectors() {
        return fusedPQEnabled && pqCodeCache == null;
    }

    public void writeHeader() throws IOException {
        writer.seek(startOffset);
        header.write(writer);
        assert writer.position() == startOffset + headerSize : String.format("%d != %d", writer.position(), startOffset + headerSize);
        writer.flush();
    }

    void writeFooter() throws IOException {
        // No "&& version == 6" here: the constructor already verified that the format this writer
        // targets supports fused PQ whenever fusedPQEnabled is true, so that's the only condition
        // this needs to check. See the constructor for why that's a safe simplification.
        if (fusedPQEnabled) {
            if (!level1FeatureRecords.isEmpty()) {
                // Hierarchy is enabled: write PQ source feature for every level-1 node.
                // Mirrors AbstractGraphIndexWriter.writeSparseLevels (getMaxLevel >= 1 branch).
                for (UpperLayerFeatureRecord record : level1FeatureRecords) {
                    writer.writeInt(record.ordinal);
                    vectorTypeSupport.writeByteSequence(writer, record.pqCode);
                }
            } else if (entryNodePqCode != null) {
                // No hierarchy: write the entry node's own PQ code so that
                // OnDiskGraphIndex.loadInMemoryFeatures can populate hierarchyCachedFeatures
                // and GraphSearcher.initializeInternal can score the entry point.
                // Mirrors AbstractGraphIndexWriter.writeSparseLevels (getMaxLevel == 0 branch).
                writer.writeInt(entryNode);
                vectorTypeSupport.writeByteSequence(writer, entryNodePqCode);
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

    /**
     * The file the offsets {@link #reserveScratch} returns refer to: the output file itself, or a
     * separate scratch file when the graph is streamed to an {@link IndexWriter}.
     */
    public Path scratchPath() {
        return scratchPath;
    }

    public void setEntryNodePqCode(ByteSequence<?> code) {
        this.entryNodePqCode = code;
    }

    private long scratchWatermark = -1;

    /**
     * Reserves {@code bytes} of transient space in {@link #scratchPath()} and returns its offset:
     * past the projected end of an output file, or from the start of a separate scratch file.
     * Strategies claim their pre-encode caches through this rather than starting at
     * {@link #projectedOutputSize()} directly, so that two of them - an inline fused strategy and a
     * sidecar scratch - can coexist instead of mapping on top of each other. An output file is still
     * truncated back to {@link #projectedOutputSize()} when compaction completes, since every
     * reserved region lies past it.
     */
    public synchronized long reserveScratch(long bytes) {
        if (scratchWatermark < 0) {
            scratchWatermark = outputPath != null ? projectedOutputSize() : 0;
        }
        long at = scratchWatermark;
        scratchWatermark += Math.max(0L, bytes);
        return at;
    }

    /**
     * Returns the projected end-of-output offset, computed from layer counts and degrees.
     * Used to place a transient PQ-code mmap section past the end of the output during
     * compaction; the file is truncated back to this size when compaction completes.
     */
    public long projectedOutputSize() {
        long total = startOffset + headerSize;
        // Level-0 records: one per ordinal slot.
        total += (long) (maxOrdinal + 1) * recordSize;
        // Upper-layer records: written by writeUpperLayerNode as
        // [ord int][count int][degree neighbor ints].
        for (int level = 1; level < configuredLayerDegrees.size(); level++) {
            int degree = configuredLayerDegrees.get(level);
            int count = configuredLayerInfo.get(level).size;
            total += (long) count * (Integer.BYTES * 2L + (long) degree * Integer.BYTES);
        }
        // PQ feature records written at the start of writeFooter() when fused PQ is enabled:
        //   - hierarchy enabled: one [ord, code] record per level-1 node;
        //   - no hierarchy:      one [entryOrd, code] record for the entry node only.
        // No "&& version == 6" here either -- see the constructor for why fusedPQEnabled alone
        // already implies the target format supports this layout.
        if (fusedPQEnabled) {
            int pqSize = fusedFeature.codeSize();
            if (configuredLayerInfo.size() > 1) {
                int level1Count = configuredLayerInfo.get(1).size;
                total += (long) level1Count * (Integer.BYTES + pqSize);
            } else {
                total += Integer.BYTES + pqSize;
            }
        }
        // Footer trailer: header copy + headerOffset (long) + magic (int).
        total += headerSize + FOOTER_SIZE;
        return total;
    }

    /** The ordinal {@code mergedOrdinal} is written as. */
    private int outputOrdinal(int mergedOrdinal) {
        return outputOrdinals == null ? mergedOrdinal : outputOrdinals[mergedOrdinal];
    }

    public void writeUpperLayerNode(int level, int ordinal, int[] neighbors, ByteSequence<?> level1PqCode) throws IOException {
        ordinal = outputOrdinal(ordinal);
        writer.writeInt(ordinal);
        writer.writeInt(neighbors.length);
        int degree = configuredLayerDegrees.get(level);
        int n = 0;
        for (; n < neighbors.length; n++) {
            writer.writeInt(outputOrdinal(neighbors[n]));
        }
        for (; n < degree; n++) {
            writer.writeInt(-1);
        }
        // No "&& version == 6" here either -- see the constructor for why fusedPQEnabled alone
        // already implies the target format supports this layout.
        if (fusedPQEnabled && level == 1 && level1PqCode != null) {
            level1FeatureRecords.add(new UpperLayerFeatureRecord(ordinal, level1PqCode.copy()));
        }
    }

    public void close() throws IOException {
        final var endOfGraphPosition = writer.position();
        writer.seek(endOfGraphPosition);
        writer.flush();
        writer.close();
    }

    public WriteResult writeInlineNodeRecord(int ordinal, VectorFloat<?> vec, SelectedVecCache selectedCache, ByteSequence<?> pqCode) throws IOException
    {
        var bwriter = new ByteBufferIndexWriter(bufferPerThread.get());

        // The neighbours' codes are looked up by merged ordinal below, before their ids are translated
        ordinal = outputOrdinal(ordinal);
        long fileOffset = startOffset + headerSize + (long) ordinal * recordSize;
        bwriter.reset();
        bwriter.writeInt(ordinal);

        for(int i = 0; i < storedDimension; ++i) {
            bwriter.writeFloat(vec.get(i));
        }

        // write fused PQ
        // since we build a graph in a streaming way,
        // we cannot use fusedPQfeature.writeInline
        if (fusedPQEnabled) {
            int k = 0;
            if (pqCodeCache != null) {
                // Look up neighbors' codes from the pre-encoded mmap'd cache instead of re-encoding.
                byte[] codeBuf = pqCodeBufPerThread.get();
                for (; k < selectedCache.size; k++) {
                    int newOrd = selectedCache.nodes[k]; // already remapped before this call
                    pqCodeCache.get(newOrd, codeBuf);
                    bwriter.write(codeBuf, 0, pqCodeSize);
                }
            } else {
                var compressor = fusedFeature.getCompressor();
                for (; k < selectedCache.size; k++) {
                    pqCode.zero();
                    compressor.encodeTo(selectedCache.vecs[k], pqCode);
                    vectorTypeSupport.writeByteSequence(bwriter, pqCode);
                }
            }
            for (; k < baseDegree; k++) {
                vectorTypeSupport.writeByteSequence(bwriter, zeroPQ.get());
            }
        }

        // write neighbors list
        bwriter.writeInt(selectedCache.size);
        int n = 0;
        for (; n < selectedCache.size; n++) {
            bwriter.writeInt(outputOrdinal(selectedCache.nodes[n]));
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

    /**
     * Opens the destination of the level-0 records, which {@link #writeInlineNodeRecord} only
     * encodes: positional writes into the output file, or sequential writes to a streamed output.
     */
    RecordSink openRecordSink() throws IOException {
        if (outputOrdinals != null) {
            return new MappedRecordSink();
        }
        return outputPath != null ? new PositionalRecordSink(outputPath) : new SequentialRecordSink();
    }

    /** Receives the level-0 records batch by batch, from the single thread that drains the batches. */
    interface RecordSink extends AutoCloseable {
        /** One batch's records, consecutive and in ordinal order; batches may arrive in any order. */
        void write(List<WriteResult> results) throws IOException;

        /** Called once every batch has been handed over; fails if a record never arrived. */
        void finish() throws IOException;

        @Override
        void close() throws IOException;
    }

    private static final class PositionalRecordSink implements RecordSink {
        private final FileChannel fc;
        // A batch's records are consecutive ordinals, hence adjacent on disk: gather each
        // contiguous run into one positional write instead of one syscall per record.
        private final ByteBuffer run = ByteBuffer.allocateDirect(4 << 20);

        PositionalRecordSink(Path outputPath) throws IOException {
            this.fc = FileChannel.open(outputPath, StandardOpenOption.WRITE, StandardOpenOption.READ);
        }

        @Override
        public void write(List<WriteResult> results) throws IOException {
            long runStart = -1, runEnd = -1;
            for (WriteResult r : results) {
                int len = r.data.remaining();
                if (runStart >= 0 && (r.fileOffset != runEnd || run.remaining() < len)) {
                    run.flip();
                    writeFully(fc, run, runStart);
                    run.clear();
                    runStart = -1;
                }
                if (len > run.capacity()) {
                    writeFully(fc, r.data, r.fileOffset);
                    continue;
                }
                if (runStart < 0) {
                    runStart = runEnd = r.fileOffset;
                }
                run.put(r.data);
                runEnd += len;
            }
            if (runStart >= 0) {
                run.flip();
                writeFully(fc, run, runStart);
                run.clear();
            }
        }

        @Override
        public void finish() {
            // every record has its own position in the file
        }

        @Override
        public void close() throws IOException {
            fc.close();
        }

        private static void writeFully(FileChannel fc, ByteBuffer b, long pos) throws IOException {
            while (b.hasRemaining()) {
                pos += fc.write(b, pos);
            }
        }
    }

    /**
     * Writes the records in ordinal order. The output ordinals follow the order level 0 is processed
     * in, so a batch that finishes early only waits for the few still in flight before it: what is
     * held here is bounded by the compactor's window of batches in flight.
     */
    private final class SequentialRecordSink implements RecordSink {
        private final TreeMap<Long, List<WriteResult>> pending = new TreeMap<>();
        private long next = startOffset + headerSize;
        private byte[] copy = new byte[0];

        @Override
        public void write(List<WriteResult> results) throws IOException {
            if (results.isEmpty()) {
                return;
            }
            pending.put(results.get(0).fileOffset, results);
            Map.Entry<Long, List<WriteResult>> head;
            while ((head = pending.firstEntry()) != null && head.getKey() == next) {
                pending.pollFirstEntry();
                for (WriteResult r : head.getValue()) {
                    if (r.fileOffset != next) {
                        throw new IllegalStateException("level-0 record of ordinal " + r.newOrdinal + " at offset " + r.fileOffset
                                                        + " does not follow the record written before it, which ends at " + next);
                    }
                    ByteBuffer data = r.data;
                    int len = data.remaining();
                    if (data.hasArray()) {
                        writer.write(data.array(), data.arrayOffset() + data.position(), len);
                    } else {
                        if (copy.length < len) {
                            copy = new byte[len];
                        }
                        data.get(copy, 0, len);
                        writer.write(copy, 0, len);
                    }
                    next += len;
                }
            }
        }

        @Override
        public void finish() {
            long end = startOffset + headerSize + (long) (maxOrdinal + 1) * recordSize;
            if (next != end || !pending.isEmpty()) {
                throw new IllegalStateException("level-0 records written up to offset " + next + " of " + end
                                                + ", with " + pending.size() + " batches still waiting for a missing one");
            }
        }

        @Override
        public void close() {
            pending.clear();
        }
    }

    /**
     * Writes the records when the caller chose the output ordinals. They arrive in the merge's own order, which is
     * not the output's, so a batch's records are scattered over the whole level-0 section: a positional write per
     * record would cost a system call each, and a reorder buffer would end up holding most of the graph. The section
     * is mapped instead and every record is copied to its place: in the output file itself, or, for a streamed output,
     * in the scratch file, from which the complete section is copied to the output in order.
     */
    private final class MappedRecordSink implements RecordSink {
        private final FileChannel fc;
        private final PreEncodedCodeCache records;
        private final int count = maxOrdinal + 1;
        private final boolean streamed = outputPath == null;
        private byte[] copy = new byte[0];
        private int written;

        MappedRecordSink() throws IOException {
            long sectionBytes = PreEncodedCodeCache.sectionBytes(count, recordSize);
            long at = streamed ? reserveScratch(sectionBytes) : startOffset + headerSize;
            this.fc = FileChannel.open(streamed ? scratchPath : outputPath, StandardOpenOption.READ, StandardOpenOption.WRITE);
            try {
                this.records = PreEncodedCodeCache.map(fc, at, count, recordSize);
            } catch (IOException | RuntimeException e) {
                fc.close();
                throw e;
            }
        }

        @Override
        public void write(List<WriteResult> results) {
            for (WriteResult r : results) {
                ByteBuffer data = r.data;
                if (data.hasArray()) {
                    records.put(r.newOrdinal, data.array(), data.arrayOffset() + data.position());
                } else {
                    if (copy.length < recordSize) {
                        copy = new byte[recordSize];
                    }
                    data.get(copy, 0, recordSize);
                    records.put(r.newOrdinal, copy, 0);
                }
            }
            written += results.size();
        }

        @Override
        public void finish() throws IOException {
            if (written != count) {
                throw new IllegalStateException(written + " level-0 records written of " + count);
            }
            if (streamed) {
                byte[] record = new byte[recordSize];
                for (int ordinal = 0; ordinal < count; ordinal++) {
                    records.get(ordinal, record);
                    writer.write(record, 0, recordSize);
                }
            }
        }

        @Override
        public void close() throws IOException {
            records.close();
            fc.close();
        }
    }

    /**
     * The caller's output, driven as the writer of this class: it never seeks anywhere but where it
     * already is, so a seek elsewhere means a write out of order and fails. The caller owns the
     * output, so closing this leaves it open.
     */
    private static final class SequentialWriter implements RandomAccessWriter {
        private final IndexWriter out;

        SequentialWriter(IndexWriter out) {
            this.out = out;
        }

        @Override
        public void seek(long position) throws IOException {
            long at = out.position();
            if (position != at) {
                throw new IllegalStateException("a sequential output cannot move from offset " + at + " to " + position);
            }
        }

        @Override
        public long position() throws IOException {
            return out.position();
        }

        @Override
        public void flush() {
            // the caller owns the output and decides when it is flushed
        }

        @Override
        public long checksum(long startOffset, long endOffset) {
            throw new UnsupportedOperationException("a sequential output cannot be read back");
        }

        @Override
        public void close() {
            // the caller owns the output
        }

        @Override public void writeFloats(float[] floats, int offset, int count) throws IOException { out.writeFloats(floats, offset, count); }
        @Override public void write(int b) throws IOException { out.write(b); }
        @Override public void write(byte[] b) throws IOException { out.write(b); }
        @Override public void write(byte[] b, int off, int len) throws IOException { out.write(b, off, len); }
        @Override public void writeBoolean(boolean v) throws IOException { out.writeBoolean(v); }
        @Override public void writeByte(int v) throws IOException { out.writeByte(v); }
        @Override public void writeShort(int v) throws IOException { out.writeShort(v); }
        @Override public void writeChar(int v) throws IOException { out.writeChar(v); }
        @Override public void writeInt(int v) throws IOException { out.writeInt(v); }
        @Override public void writeLong(long v) throws IOException { out.writeLong(v); }
        @Override public void writeFloat(float v) throws IOException { out.writeFloat(v); }
        @Override public void writeDouble(double v) throws IOException { out.writeDouble(v); }
        @Override public void writeBytes(String s) throws IOException { out.writeBytes(s); }
        @Override public void writeChars(String s) throws IOException { out.writeChars(s); }
        @Override public void writeUTF(String s) throws IOException { out.writeUTF(s); }
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

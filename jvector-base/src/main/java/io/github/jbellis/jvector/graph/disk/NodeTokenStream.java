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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * The node token stream: an index's base-layer structure as a sequence of byte-prefixed ordinal
 * tokens, emitted in ordinal (address) order, stored as an additive section of the index file.
 *
 * <p>Grammar, one node after another for every ordinal {@code 0..nodeCount-1}:
 * <pre>
 *   NODE  prefix 1LLLLLLL-ish: bit 7 set, bit 6 = live, bits 0..5 = the node's highest level
 *         then the ordinal, as an unsigned varint delta from the previous NODE's ordinal
 *   NB    prefix 0x01, then a neighbour ordinal — one token per base-layer edge, in adjacency order
 *         delta encoding: zigzag varint of (neighbour - node); raw encoding: a big-endian int
 * </pre>
 * A dead ordinal (a hole in the mapping, or a compaction output slot with no record) is a NODE
 * token with the live bit clear and no NB tokens. Upper-level adjacency is not in the stream: it is
 * already stored sequentially and loaded whole at open; the NODE level bits give membership.
 *
 * <p>Layout in the file: {@code [graph body][section][trailer][footer]}. The section starts with
 * {@link #SECTION_MAGIC}, version, encoding, node count, base degree and max level, then the
 * tokens. The trailer — the section's length and {@link #TRAILER_MAGIC} — sits immediately before
 * the footer's header copy, so a reader that has the footer's {@code headerOffset} finds the
 * section by looking {@link #TRAILER_SIZE} bytes before it; a reader that does not look never
 * touches it, and every other byte of the index is where it was. Delta encoding is what makes the
 * stream small: under similarity-ordered ordinals a node's neighbours sit at nearby ordinals and
 * the deltas are short, which is the ramp {@code vector_merge_splat_design.md} §6 describes.
 *
 * <p>Writers: {@link AbstractGraphIndexWriter} emits it from the in-memory graph before the
 * footer; {@link OnDiskGraphIndexCompactor} emits it at the end of a merge from the written
 * output, after refinement has settled the adjacency. Properties: {@value #ENABLED_PROPERTY}
 * (default true) and {@value #ENCODING_PROPERTY} ({@code delta} or {@code raw}, default delta —
 * raw exists to measure the delta form against).
 */
@Experimental
public final class NodeTokenStream {
    public static final int SECTION_MAGIC = 0x544B5354; // "TKST"
    public static final int TRAILER_MAGIC = 0x544B5345; // "TKSE"
    public static final int VERSION = 1;
    public static final int TRAILER_SIZE = Long.BYTES + Integer.BYTES;
    /** magic, version, encoding, nodeCount, degree, maxLevel */
    public static final int HEADER_SIZE = Integer.BYTES + Integer.BYTES + 1 + Integer.BYTES + Integer.BYTES + 1;
    public static final byte ENCODING_RAW = 0;
    public static final byte ENCODING_DELTA = 1;
    public static final String ENABLED_PROPERTY = "jvector.tokenStream";
    public static final String ENCODING_PROPERTY = "jvector.tokenStream.encoding";

    static final int NODE_TOKEN = 0x80;
    static final int NODE_LIVE = 0x40;
    static final int NODE_LEVEL_MASK = 0x3F;
    static final int NB_TOKEN = 0x01;
    /** Highest level the NODE prefix can carry. */
    public static final int MAX_LEVEL = NODE_LEVEL_MASK;

    private NodeTokenStream() {
    }

    public static boolean enabledByDefault() {
        return !"false".equalsIgnoreCase(System.getProperty(ENABLED_PROPERTY, "true"));
    }

    public static byte encodingByDefault() {
        return "raw".equalsIgnoreCase(System.getProperty(ENCODING_PROPERTY, "delta")) ? ENCODING_RAW : ENCODING_DELTA;
    }

    /** Bytes the same structure costs as one prefix byte plus a 4-byte ordinal per token. */
    public static long rawEquivalentBytes(long nodes, long edges) {
        return 5L * (nodes + edges);
    }

    public static String encodingName(byte encoding) {
        return encoding == ENCODING_RAW ? "raw" : "delta";
    }

    /** Where a section lies in the file: {@code [offset, offset + length)}, header included, trailer excluded. */
    public static final class Section {
        public final long offset;
        public final long length;

        Section(long offset, long length) {
            this.offset = offset;
            this.length = length;
        }

        @Override
        public String toString() {
            return "NodeTokenStream.Section[offset=" + offset + ", length=" + length + "]";
        }
    }

    /**
     * Finds the section that precedes the footer whose header copy begins at {@code headerOffset},
     * or {@code null} if the file carries none. {@code lowerBound} is the lowest offset the section
     * could start at (the end of the index header), so a stray magic in a file without a section
     * cannot be mistaken for one.
     */
    public static Section locate(RandomAccessReader in, long headerOffset, long lowerBound) throws IOException {
        long trailerOffset = headerOffset - TRAILER_SIZE;
        if (trailerOffset < lowerBound) {
            return null;
        }
        in.seek(trailerOffset);
        long length = in.readLong();
        int magic = in.readInt();
        if (magic != TRAILER_MAGIC || length < HEADER_SIZE) {
            return null;
        }
        long start = trailerOffset - length;
        if (start < lowerBound) {
            return null;
        }
        in.seek(start);
        if (in.readInt() != SECTION_MAGIC) {
            return null;
        }
        return new Section(start, length);
    }

    /**
     * Writes one section: header on construction, tokens through {@link #node} and
     * {@link #neighbor}, then {@link #finish} flushes and writes the trailer. Ordinals must arrive
     * ascending and every ordinal below {@code nodeCount} must be named exactly once.
     */
    public static final class Encoder {
        private final IndexWriter out;
        private final byte encoding;
        private final int nodeCount;
        private final byte[] buf = new byte[1 << 16];
        private int pos;
        private long bytes;
        private long nodes;
        private long edges;
        private long liveNodes;
        private int prevNode = -1;
        private int currentNode = -1;

        public Encoder(IndexWriter out, byte encoding, int nodeCount, int degree, int maxLevel) throws IOException {
            if (encoding != ENCODING_RAW && encoding != ENCODING_DELTA) {
                throw new IllegalArgumentException("unknown encoding " + encoding);
            }
            if (maxLevel < 0 || maxLevel > MAX_LEVEL) {
                throw new IllegalArgumentException("maxLevel out of range: " + maxLevel);
            }
            this.out = out;
            this.encoding = encoding;
            this.nodeCount = nodeCount;
            out.writeInt(SECTION_MAGIC);
            out.writeInt(VERSION);
            out.writeByte(encoding);
            out.writeInt(nodeCount);
            out.writeInt(degree);
            out.writeByte(maxLevel);
            bytes = HEADER_SIZE;
        }

        public void node(int ordinal, boolean live, int level) throws IOException {
            if (ordinal != prevNode + 1) {
                throw new IllegalStateException("ordinals must be contiguous and ascending: got " + ordinal + " after " + prevNode);
            }
            if (ordinal >= nodeCount) {
                throw new IllegalStateException("ordinal " + ordinal + " beyond nodeCount " + nodeCount);
            }
            if (level < 0 || level > MAX_LEVEL) {
                throw new IllegalArgumentException("level out of range: " + level);
            }
            ensure(6);
            buf[pos++] = (byte) (NODE_TOKEN | (live ? NODE_LIVE : 0) | (level & NODE_LEVEL_MASK));
            if (encoding == ENCODING_RAW) {
                putInt(ordinal);
            } else {
                putUnsignedVarint((long) ordinal - prevNode);
            }
            prevNode = ordinal;
            currentNode = ordinal;
            nodes++;
            if (live) {
                liveNodes++;
            }
        }

        public void neighbor(int ordinal) throws IOException {
            if (currentNode < 0) {
                throw new IllegalStateException("neighbor before any node");
            }
            ensure(6);
            buf[pos++] = NB_TOKEN;
            if (encoding == ENCODING_RAW) {
                putInt(ordinal);
            } else {
                putZigzagVarint(ordinal - currentNode);
            }
            edges++;
        }

        /** Flushes the tokens and writes the trailer. Returns the section length (trailer excluded). */
        public long finish() throws IOException {
            if (nodes != nodeCount) {
                throw new IllegalStateException("wrote " + nodes + " nodes of " + nodeCount);
            }
            flush();
            out.writeLong(bytes);
            out.writeInt(TRAILER_MAGIC);
            return bytes;
        }

        public long bytes() {
            return bytes;
        }

        public long nodes() {
            return nodes;
        }

        public long liveNodes() {
            return liveNodes;
        }

        public long edges() {
            return edges;
        }

        public byte encoding() {
            return encoding;
        }

        private void ensure(int n) throws IOException {
            if (pos + n > buf.length) {
                flush();
            }
        }

        private void flush() throws IOException {
            if (pos > 0) {
                out.write(buf, 0, pos);
                bytes += pos;
                pos = 0;
            }
        }

        private void putInt(int v) {
            buf[pos++] = (byte) (v >>> 24);
            buf[pos++] = (byte) (v >>> 16);
            buf[pos++] = (byte) (v >>> 8);
            buf[pos++] = (byte) v;
        }

        private void putUnsignedVarint(long v) {
            while ((v & ~0x7FL) != 0) {
                buf[pos++] = (byte) ((v & 0x7F) | 0x80);
                v >>>= 7;
            }
            buf[pos++] = (byte) v;
        }

        private void putZigzagVarint(int v) {
            putUnsignedVarint(((long) v << 1) ^ (v >> 31));
        }
    }

    /** Sequential decoder over one section: {@link #next()} advances to the next node. */
    public static final class Reader implements AutoCloseable {
        private final RandomAccessReader in;
        private final long end;
        private long filePos;
        private final byte[] buf = new byte[1 << 20];
        private int bufPos;
        private int bufLen;

        public final byte encoding;
        public final int nodeCount;
        public final int degree;
        public final int maxLevel;

        private int prevNode = -1;
        private int ordinal = -1;
        private boolean live;
        private int level;
        private int[] neighbors;
        private int neighborCount;
        private long nodesRead;

        public Reader(RandomAccessReader in, Section section) throws IOException {
            this.in = in;
            in.seek(section.offset);
            int magic = in.readInt();
            if (magic != SECTION_MAGIC) {
                throw new IOException("not a token stream section at " + section.offset + ": magic " + Integer.toHexString(magic));
            }
            int version = in.readInt();
            if (version != VERSION) {
                throw new IOException("unsupported token stream version " + version);
            }
            byte[] b1 = new byte[1];
            in.readFully(b1);
            encoding = b1[0];
            if (encoding != ENCODING_RAW && encoding != ENCODING_DELTA) {
                throw new IOException("unknown token stream encoding " + encoding);
            }
            nodeCount = in.readInt();
            degree = in.readInt();
            in.readFully(b1);
            maxLevel = b1[0];
            neighbors = new int[Math.max(degree, 1)];
            filePos = section.offset + HEADER_SIZE;
            end = section.offset + section.length;
        }

        /** Advances to the next NODE token; false at the end of the section. */
        public boolean next() throws IOException {
            if (!hasByte()) {
                if (nodesRead != nodeCount) {
                    throw new IOException("token stream ended after " + nodesRead + " of " + nodeCount + " nodes");
                }
                return false;
            }
            int t = readByte();
            if ((t & NODE_TOKEN) == 0) {
                throw new IOException("expected NODE token, got 0x" + Integer.toHexString(t) + " at node " + nodesRead);
            }
            live = (t & NODE_LIVE) != 0;
            level = t & NODE_LEVEL_MASK;
            int o = encoding == ENCODING_RAW ? readInt() : (int) (prevNode + readUnsignedVarint());
            if (o != prevNode + 1) {
                throw new IOException("non-contiguous ordinal " + o + " after " + prevNode);
            }
            prevNode = o;
            ordinal = o;
            nodesRead++;
            neighborCount = 0;
            while (hasByte() && peekByte() == NB_TOKEN) {
                readByte();
                int nb = encoding == ENCODING_RAW ? readInt() : ordinal + readZigzagVarint();
                if (neighborCount == neighbors.length) {
                    neighbors = Arrays.copyOf(neighbors, neighbors.length * 2);
                }
                neighbors[neighborCount++] = nb;
            }
            return true;
        }

        public int ordinal() {
            return ordinal;
        }

        public boolean live() {
            return live;
        }

        public int level() {
            return level;
        }

        public int neighborCount() {
            return neighborCount;
        }

        /** The current node's neighbours; valid in {@code [0, neighborCount())} until the next {@link #next()}. */
        public int[] neighbors() {
            return neighbors;
        }

        private boolean hasByte() throws IOException {
            if (bufPos < bufLen) {
                return true;
            }
            long remaining = end - filePos;
            if (remaining <= 0) {
                return false;
            }
            int n = (int) Math.min(buf.length, remaining);
            in.seek(filePos);
            in.readFully(ByteBuffer.wrap(buf, 0, n));
            filePos += n;
            bufPos = 0;
            bufLen = n;
            return true;
        }

        private int peekByte() throws IOException {
            if (!hasByte()) {
                throw new IOException("unexpected end of token stream");
            }
            return buf[bufPos] & 0xFF;
        }

        private int readByte() throws IOException {
            int b = peekByte();
            bufPos++;
            return b;
        }

        private int readInt() throws IOException {
            return (readByte() << 24) | (readByte() << 16) | (readByte() << 8) | readByte();
        }

        private long readUnsignedVarint() throws IOException {
            long v = 0;
            int shift = 0;
            while (true) {
                int b = readByte();
                v |= (long) (b & 0x7F) << shift;
                if ((b & 0x80) == 0) {
                    return v;
                }
                shift += 7;
                if (shift > 63) {
                    throw new IOException("varint too long");
                }
            }
        }

        private int readZigzagVarint() throws IOException {
            long u = readUnsignedVarint();
            return (int) ((u >>> 1) ^ -(u & 1));
        }

        @Override
        public void close() throws IOException {
            in.close();
        }
    }
}

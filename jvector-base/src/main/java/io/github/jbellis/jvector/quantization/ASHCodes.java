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

import io.github.jbellis.jvector.disk.ByteReadable;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.RamUsageEstimator;

import java.io.IOException;
import java.util.Arrays;

/**
 * The in-memory store of an {@link ASHVectors} collection: every stored code, and nothing else.
 *
 * <p>Codes sit back to back, at a fixed stride and with no padding, in pages of primitive arrays. There is
 * no per-vector object or array, so the heap cost per vector is its code body plus the decoded header
 * ({@value #HEADER_RAM_BYTES} bytes in the parallel {@link #scales}, {@link #offsets} and {@link #landmarks}
 * arrays) and a negligible share of the page headers.</p>
 *
 * <ul>
 *   <li>Sign words, used by 1-bit codes and by the generic sign-plus-extra-bit widths (3, 5..9 bits), are in
 *   {@code long} pages, {@link #signWords} per vector.</li>
 *   <li>Byte bodies, which are the canonical projection codes at 2 and 4 bits and the extra bits of the
 *   generic widths, are in {@code byte} pages, {@link #bodyBytes} per vector.</li>
 * </ul>
 *
 * <p>Both kinds of page hold the same power-of-two number of vectors, so vector {@code i} is in page
 * {@code i >>> pageShift} at slot {@code i & pageMask}. Pages are kept under G1's smallest humongous
 * threshold (half of a 1 MiB region) so the collector never rounds them up to whole regions; the last
 * page is trimmed to the vectors it holds.</p>
 *
 * <p>Concurrency: a store is filled once, by the encoder (tasks own disjoint vectors and are joined
 * before the store is published) or by the loader, and is read-only afterward.</p>
 */
final class ASHCodes implements Accountable {
    /** Target page size in bytes; see the class comment. */
    static final int DEFAULT_PAGE_BYTES = 256 << 10;

    /** Decoded header bytes per vector: a float scale, a float offset and a byte landmark. */
    static final int HEADER_RAM_BYTES = Float.BYTES + Float.BYTES + Byte.BYTES;

    final int size;
    final int quantizedDim;
    final int bitsPerDimension;
    /** Sign words per vector; zero for the 2/4-bit projection codes. */
    final int signWords;
    /** Body bytes per vector; zero for 1-bit codes. */
    final int bodyBytes;
    final int pageShift;
    final int pageMask;
    /** Null when {@link #signWords} is zero. */
    final long[][] signPages;
    /** Null when {@link #bodyBytes} is zero. */
    final byte[][] bodyPages;
    /** Binary16-rounded values decoded to float, as serialized. */
    final float[] scales;
    final float[] offsets;
    final byte[] landmarks;
    private final long ramBytesUsed;

    private ASHCodes(int size, int quantizedDim, int bitsPerDimension, int pageBytes) {
        if (size < 0) throw new IllegalArgumentException("size must be nonnegative");
        if (pageBytes <= 0) throw new IllegalArgumentException("pageBytes must be positive");
        this.size = size;
        this.quantizedDim = quantizedDim;
        this.bitsPerDimension = bitsPerDimension;
        this.signWords = AsymmetricHashing.QuantizedVector.signWordsForDims(quantizedDim, bitsPerDimension);
        this.bodyBytes = AsymmetricHashing.QuantizedVector.bodyBytesForDims(quantizedDim, bitsPerDimension);
        int strideBytes = Math.max(1, Math.max(Math.multiplyExact(signWords, Long.BYTES), bodyBytes));
        // A power of two, so locating a vector is a shift and a mask.
        this.pageShift = 31 - Integer.numberOfLeadingZeros(Math.max(1, pageBytes / strideBytes));
        this.pageMask = (1 << pageShift) - 1;
        int pages = Math.toIntExact((size + (long) pageMask) >>> pageShift);
        this.signPages = signWords == 0 ? null : new long[pages][];
        this.bodyPages = bodyBytes == 0 ? null : new byte[pages][];
        long bytes = RamUsageEstimator.shallowSizeOfInstance(ASHCodes.class);
        for (int p = 0; p < pages; p++) {
            int vectors = (int) Math.min(pageMask + 1L, size - ((long) p << pageShift));
            if (signPages != null) {
                signPages[p] = new long[Math.multiplyExact(vectors, signWords)];
                bytes += RamUsageEstimator.sizeOf(signPages[p]);
            }
            if (bodyPages != null) {
                bodyPages[p] = new byte[Math.multiplyExact(vectors, bodyBytes)];
                bytes += RamUsageEstimator.sizeOf(bodyPages[p]);
            }
        }
        this.scales = new float[size];
        this.offsets = new float[size];
        this.landmarks = new byte[size];
        if (signPages != null) bytes += RamUsageEstimator.shallowSizeOf(signPages);
        if (bodyPages != null) bytes += RamUsageEstimator.shallowSizeOf(bodyPages);
        bytes += RamUsageEstimator.sizeOf(scales) + RamUsageEstimator.sizeOf(offsets)
                + RamUsageEstimator.sizeOf(landmarks);
        this.ramBytesUsed = bytes;
    }

    /** An all-zero store for {@code size} vectors, to be filled with {@link #set}. */
    static ASHCodes allocate(int size, int quantizedDim, int bitsPerDimension) {
        return allocate(size, quantizedDim, bitsPerDimension, DEFAULT_PAGE_BYTES);
    }

    /** As {@link #allocate(int, int, int)} with a chosen page size; tests use small pages. */
    static ASHCodes allocate(int size, int quantizedDim, int bitsPerDimension, int pageBytes) {
        return new ASHCodes(size, quantizedDim, bitsPerDimension, pageBytes);
    }

    /** Copies the given vectors, which must all have this model's shape. Vectors are not retained. */
    static ASHCodes copyOf(AsymmetricHashing.QuantizedVector[] vectors, int quantizedDim, int bitsPerDimension) {
        ASHCodes codes = allocate(vectors.length, quantizedDim, bitsPerDimension);
        for (int i = 0; i < vectors.length; i++) codes.set(i, vectors[i]);
        return codes;
    }

    long[] signPage(int node) { return signPages[node >>> pageShift]; }

    int signOffset(int node) { return (node & pageMask) * signWords; }

    byte[] bodyPage(int node) { return bodyPages[node >>> pageShift]; }

    int bodyOffset(int node) { return (node & pageMask) * bodyBytes; }

    /** Stores {@code vector} at {@code node}. For filling only; see the class comment. */
    void set(int node, AsymmetricHashing.QuantizedVector vector) {
        java.util.Objects.checkIndex(node, size);
        if (vector.binaryVector.length < signWords || vector.extraBits.length < bodyBytes)
            throw new IllegalArgumentException("ASH vector " + node + " does not match the model's code shape");
        if (signWords != 0) System.arraycopy(vector.binaryVector, 0, signPage(node), signOffset(node), signWords);
        if (bodyBytes != 0) System.arraycopy(vector.extraBits, 0, bodyPage(node), bodyOffset(node), bodyBytes);
        scales[node] = vector.scale;
        offsets[node] = vector.offset;
        landmarks[node] = vector.landmark;
    }

    /** Copies the stored vector into {@code dest}, replacing its arrays if their lengths differ. */
    void copyTo(int node, AsymmetricHashing.QuantizedVector dest) {
        java.util.Objects.checkIndex(node, size);
        if (dest.binaryVector == null || dest.binaryVector.length != signWords) dest.binaryVector = new long[signWords];
        if (dest.extraBits == null || dest.extraBits.length != bodyBytes) dest.extraBits = new byte[bodyBytes];
        if (signWords != 0) System.arraycopy(signPage(node), signOffset(node), dest.binaryVector, 0, signWords);
        if (bodyBytes != 0) System.arraycopy(bodyPage(node), bodyOffset(node), dest.extraBits, 0, bodyBytes);
        dest.scale = scales[node];
        dest.offset = offsets[node];
        dest.landmark = landmarks[node];
    }

    /** A new, independent copy of the stored vector. */
    AsymmetricHashing.QuantizedVector get(int node) {
        var vector = AsymmetricHashing.QuantizedVector.createEmpty(quantizedDim, bitsPerDimension);
        copyTo(node, vector);
        return vector;
    }

    /** A store holding vector {@code newToOld[i]} at {@code i}. */
    ASHCodes permute(int[] newToOld) {
        if (newToOld.length != size) throw new IllegalArgumentException("Permutation must cover every vector");
        ASHCodes result = new ASHCodes(size, quantizedDim, bitsPerDimension, (pageMask + 1) * strideBytes());
        for (int i = 0; i < size; i++) {
            int old = newToOld[i];
            if (signWords != 0)
                System.arraycopy(signPage(old), signOffset(old), result.signPage(i), result.signOffset(i), signWords);
            if (bodyBytes != 0)
                System.arraycopy(bodyPage(old), bodyOffset(old), result.bodyPage(i), result.bodyOffset(i), bodyBytes);
            result.scales[i] = scales[old];
            result.offsets[i] = offsets[old];
            result.landmarks[i] = landmarks[old];
        }
        return result;
    }

    private int strideBytes() {
        return Math.max(1, Math.max(signWords * Long.BYTES, bodyBytes));
    }

    /** Writes every vector in the same format as {@link AsymmetricHashing.QuantizedVector#write}. */
    void write(IndexWriter out) throws IOException {
        for (int i = 0; i < size; i++) {
            AsymmetricHashing.writeFloat16(out, scales[i]);
            AsymmetricHashing.writeFloat16(out, offsets[i]);
            out.writeByte(landmarks[i]);
            if (signWords != 0) {
                long[] page = signPage(i);
                for (int w = 0, at = signOffset(i); w < signWords; w++) out.writeLong(page[at + w]);
            }
            if (bodyBytes != 0) out.write(bodyPage(i), bodyOffset(i), bodyBytes);
        }
    }

    /** Reads {@code size} vectors written by {@link #write} or {@link AsymmetricHashing.QuantizedVector#write}. */
    static ASHCodes load(RandomAccessReader in, int size, int quantizedDim, int bitsPerDimension) throws IOException {
        if (!(in instanceof ByteReadable)) throw new IOException("ASH requires ByteReadable reader for landmark");
        ASHCodes codes = allocate(size, quantizedDim, bitsPerDimension);
        byte[] body = new byte[codes.bodyBytes];
        for (int i = 0; i < size; i++) {
            codes.scales[i] = AsymmetricHashing.readFloat16(in);
            codes.offsets[i] = AsymmetricHashing.readFloat16(in);
            codes.landmarks[i] = ((ByteReadable) in).readByte();
            if (codes.signWords != 0) {
                long[] page = codes.signPage(i);
                for (int w = 0, at = codes.signOffset(i); w < codes.signWords; w++) page[at + w] = in.readLong();
            }
            if (codes.bodyBytes != 0) {
                in.readFully(body);
                System.arraycopy(body, 0, codes.bodyPage(i), codes.bodyOffset(i), codes.bodyBytes);
            }
        }
        return codes;
    }

    /** Fixed at allocation: pages, header arrays and this object. */
    @Override
    public long ramBytesUsed() {
        return ramBytesUsed;
    }

    /** Same vectors in the same order, comparing headers by their float bits. */
    boolean contentEquals(ASHCodes that) {
        if (size != that.size || signWords != that.signWords || bodyBytes != that.bodyBytes) return false;
        if (!Arrays.equals(landmarks, that.landmarks)) return false;
        for (int i = 0; i < size; i++) {
            if (Float.floatToIntBits(scales[i]) != Float.floatToIntBits(that.scales[i])
                    || Float.floatToIntBits(offsets[i]) != Float.floatToIntBits(that.offsets[i])) return false;
            if (signWords != 0 && !Arrays.equals(signPage(i), signOffset(i), signOffset(i) + signWords,
                    that.signPage(i), that.signOffset(i), that.signOffset(i) + signWords)) return false;
            if (bodyBytes != 0 && !Arrays.equals(bodyPage(i), bodyOffset(i), bodyOffset(i) + bodyBytes,
                    that.bodyPage(i), that.bodyOffset(i), that.bodyOffset(i) + bodyBytes)) return false;
        }
        return true;
    }

    /** Consistent with {@link #contentEquals}, independent of page size. */
    int contentHash() {
        int h = size;
        for (int i = 0; i < size; i++) {
            h = 31 * h + Float.floatToIntBits(scales[i]);
            h = 31 * h + Float.floatToIntBits(offsets[i]);
            h = 31 * h + landmarks[i];
            if (signWords != 0) {
                long[] page = signPage(i);
                for (int w = 0, at = signOffset(i); w < signWords; w++) h = 31 * h + Long.hashCode(page[at + w]);
            }
            if (bodyBytes != 0) {
                byte[] page = bodyPage(i);
                for (int b = 0, at = bodyOffset(i); b < bodyBytes; b++) h = 31 * h + page[at + b];
            }
        }
        return h;
    }
}

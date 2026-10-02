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

import java.util.*;
import java.util.concurrent.*;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.FloatArray;
import static java.lang.Math.*;

/**
 * Per-thread decoder for the wide code: a candidate's code becomes a near-exact vector, scored
 * and compared like one. With two-dimensional subspaces (every dimension up to 384) each
 * centroid is one packed {@code long}, so a decode is {@code m} table reads plus one centroid
 * add and no per-subspace bookkeeping; other subspace sizes take a flat per-subspace copy.
 * Vectors that expose no heap array ({@link FloatArray}) take the generic decode.
 */
final class WideDecoder {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final PreEncodedCodeCache cache;
    private final boolean raw;   // the store holds the vectors themselves rather than codes
    private final int m;
    private final int clusterCount;
    private final ByteSequence<?> code;
    private final byte[] codeBytes;
    private final float[] codebooks;      // every subspace's codebook, subspace sub at codebookBase[sub]; null without heap arrays
    private final long[] pairs;           // two-dimensional subspaces only: centroid (sub, c) packed at [sub * clusterCount + c]
    private final int[] codebookBase, sizes, offsets;
    private final float[] center;

    private final ProductQuantization pqWide;   // null when the store holds raw vectors
    private final int dimension;

    WideDecoder(PreEncodedCodeCache cache, boolean raw, ProductQuantization pqWide, int dimension) {
        this.cache = cache;
        this.raw = raw;
        this.pqWide = pqWide;
        this.dimension = dimension;
        this.m = raw ? 4 * dimension : pqWide.getSubspaceCount();
        this.clusterCount = raw ? 0 : pqWide.getClusterCount();
        this.code = vectorTypeSupport.createByteSequence(m);
        this.codeBytes = new byte[m];
        if (raw) {
            sizes = offsets = codebookBase = null;
            codebooks = null;
            pairs = null;
            center = null;
            return;
        }
        sizes = new int[m];
        offsets = new int[m];
        codebookBase = new int[m];
        boolean flat = true, allPairs = true;
        int total = 0, offset = 0;
        for (int sub = 0; sub < m; sub++) {
            VectorFloat<?> codebook = pqWide.getCodebookVector(sub);
            flat &= codebook instanceof FloatArray;
            sizes[sub] = pqWide.getSubvectorSize(sub);
            allPairs &= sizes[sub] == 2;
            offsets[sub] = offset;
            offset += sizes[sub];
            codebookBase[sub] = total;
            total += codebook.length();
        }
        VectorFloat<?> globalCentroid = pqWide.getGlobalCentroid();
        flat &= globalCentroid == null || globalCentroid instanceof FloatArray;
        if (flat) {
            codebooks = new float[total];
            for (int sub = 0; sub < m; sub++) {
                VectorFloat<?> codebook = pqWide.getCodebookVector(sub);
                for (int i = 0; i < codebook.length(); i++) {
                    codebooks[codebookBase[sub] + i] = codebook.get(i);
                }
            }
            if (allPairs) {
                pairs = new long[m * clusterCount];
                for (int sub = 0; sub < m; sub++) {
                    for (int c = 0; c < clusterCount; c++) {
                        int at = codebookBase[sub] + 2 * c;
                        pairs[sub * clusterCount + c] = (Float.floatToRawIntBits(codebooks[at]) & 0xFFFFFFFFL)
                                | ((long) Float.floatToRawIntBits(codebooks[at + 1]) << 32);
                    }
                }
            } else {
                pairs = null;
            }
            if (globalCentroid == null) {
                center = null;
            } else {
                center = new float[dimension];
                for (int i = 0; i < dimension; i++) {
                    center[i] = globalCentroid.get(i);
                }
            }
        } else {
            codebooks = null;
            pairs = null;
            center = null;
        }
    }

    void decode(int newOrdinal, VectorFloat<?> dst) {
        if (raw) {
            if (dst instanceof FloatArray) {
                cache.getFloats(newOrdinal, ((FloatArray) dst).array(), dimension);
            } else {
                cache.get(newOrdinal, codeBytes);
                RawVectorCode.decodeInto(codeBytes, dst, dimension);
            }
            return;
        }
        cache.get(newOrdinal, codeBytes);
        if (codebooks != null && dst instanceof FloatArray) {
            float[] out = ((FloatArray) dst).array();
            if (pairs != null) {
                long[] table = pairs;
                int k = clusterCount;
                for (int sub = 0; sub < m; sub++) {
                    long pair = table[sub * k + (codeBytes[sub] & 0xFF)];
                    out[2 * sub] = Float.intBitsToFloat((int) pair);
                    out[2 * sub + 1] = Float.intBitsToFloat((int) (pair >>> 32));
                }
            } else {
                float[] cb = codebooks;
                for (int sub = 0; sub < m; sub++) {
                    int size = sizes[sub];
                    int from = codebookBase[sub] + (codeBytes[sub] & 0xFF) * size;
                    int to = offsets[sub];
                    for (int i = 0; i < size; i++) {
                        out[to + i] = cb[from + i];
                    }
                }
            }
            if (center != null) {
                float[] c = center;
                for (int i = 0; i < dimension; i++) {
                    out[i] += c[i];
                }
            }
            return;
        }
        for (int i = 0; i < m; i++) {
            code.set(i, codeBytes[i]);
        }
        pqWide.decode(code, dst);
    }
}

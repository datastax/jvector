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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

/**
 * A {@link CompactionAccelerator} in plain Java, through the public contracts of {@link CellProbe} and {@link CellWalk}
 * only: the wide code trained by {@code ProductQuantization.compute}, every query ranks every center exactly, and every (cell, query) pair's members are scored with
 * {@link CellWalk#score}, the top k kept per query. Tests select it as {@value #NAME}.
 */
public final class ReferenceCompactionAccelerator implements CompactionAccelerator {

    public static final String NAME = "reference";
    static final AtomicInteger WALKS = new AtomicInteger();
    static final AtomicInteger PROBES = new AtomicInteger();
    static final AtomicInteger WIDE_CODES = new AtomicInteger();

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public boolean supports(VectorSimilarityFunction similarity, int dimension) {
        return true;
    }

    @Override
    public boolean walk(CellWalk walk) {
        int n = walk.queryCount();
        int dim = walk.dimension();
        int k = walk.topK();
        float[][] queries = new float[n][dim];
        float[] queryNorms = new float[n];
        for (int i = 0; i < n; i++) {
            walk.query(i, queries[i]);
            queryNorms[i] = dot(queries[i], queries[i]);
        }
        long[][] best = new long[n][];
        int[] sizes = new int[n];
        for (int i = 0; i < n; i++) {
            best[i] = new long[k];
        }
        float[] row = new float[dim];
        ByteBuffer bytes = ByteBuffer.allocate(dim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int p = 0; p < walk.pairCount(); p++) {
            long pair = walk.pairs()[p];
            int cell = (int) (pair >>> 32);
            int i = (int) pair;
            for (int member = walk.cellStart(cell); member < walk.cellEnd(cell); member++) {
                bytes.clear();
                walk.copyRows(member, 1, bytes);
                bytes.flip();
                bytes.asFloatBuffer().get(row);
                float s = walk.score(dot(queries[i], row), queryNorms[i], dot(row, row));
                long key = ((long) Float.floatToRawIntBits(s) << 32) | (member & 0xFFFFFFFFL);
                insert(best[i], sizes, i, key);
            }
        }
        for (int i = 0; i < n; i++) {
            walk.setTopK(i, best[i], 0, sizes[i]);
        }
        WALKS.incrementAndGet();
        return true;
    }

    @Override
    public ProductQuantization trainWideCode(RandomAccessVectorValues sample, int subspaces, int clusters, boolean globallyCenter) {
        WIDE_CODES.incrementAndGet();
        return ProductQuantization.compute(sample, subspaces, clusters, globallyCenter);
    }

    @Override
    public boolean probe(CellProbe probe) {
        int dim = probe.dimension();
        int centers = probe.centerCount();
        float[] rows = new float[centers * dim];
        probe.copyCenters(0, centers, rows, 0);
        float[] q = new float[dim];
        float[] center = new float[dim];
        int want = Math.min(probe.maxProbes(), centers);
        for (int i = 0; i < probe.queryCount(); i++) {
            probe.query(i, q);
            Integer[] order = new Integer[centers];
            float[] rank = new float[centers];
            for (int c = 0; c < centers; c++) {
                System.arraycopy(rows, c * dim, center, 0, dim);
                float dot = dot(q, center);
                float norm2 = dot(center, center);
                switch (probe.similarity()) {
                    case EUCLIDEAN:
                        rank[c] = 2 * dot - norm2;
                        break;
                    case COSINE:
                        rank[c] = dot / (float) Math.sqrt(Math.max(1e-12, norm2));
                        break;
                    default:
                        rank[c] = dot;
                }
                order[c] = c;
            }
            java.util.Arrays.sort(order, (x, y) -> Float.compare(rank[y], rank[x]));
            int[] best = new int[want];
            for (int r = 0; r < want; r++) {
                best[r] = order[r];
            }
            probe.setProbes(i, best, 0, want);
        }
        PROBES.incrementAndGet();
        return true;
    }

    private static void insert(long[] top, int[] sizes, int i, long key) {
        int size = sizes[i];
        if (size < top.length) {
            top[size] = key;
            sizes[i] = size + 1;
            return;
        }
        int min = 0;
        for (int e = 1; e < size; e++) {
            if (top[e] < top[min]) {
                min = e;
            }
        }
        if (key > top[min]) {
            top[min] = key;
        }
    }

    private static float dot(float[] a, float[] b) {
        float s = 0;
        for (int d = 0; d < a.length; d++) {
            s += a[d] * b[d];
        }
        return s;
    }
}

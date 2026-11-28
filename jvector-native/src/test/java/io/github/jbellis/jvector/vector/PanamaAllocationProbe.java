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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;

/**
 * Bytes allocated per operation by Panama kernels, measured after warm-up from callers at every depth from 0 to
 * {@link #MAX_DEPTH} calls above the operation. Run one case per fresh JVM with
 * {@code -Djvector.vectorization_provider=panama}; the case is fixed when the class loads, so every call is
 * monomorphic and C2 inlines the chain into each root, the way it compiles a builder. A vector boxed anywhere on the
 * path costs tens of bytes or more per operation; the designed allocation (a decoder object, a centered query copy)
 * is a fixed, small amount.
 *
 * <p>Usage: {@code -Dprobe.case=<case> PanamaAllocationProbe}; prints
 * {@code RESULT <case> <worst bytes per op> <allowed> depth=<worst depth>} and the figure for each depth.
 */
public final class PanamaAllocationProbe {
    /** One operation; returns a value so the work is not dead. */
    interface Op {
        float run(int i);
    }

    /** An operation and the bytes per operation it allocates by design. */
    static final class Case {
        final Op op;
        final double allowedBytes;

        Case(Op op, double allowedBytes) {
            this.op = op;
            this.allowedBytes = allowedBytes;
        }
    }

    /** Case names, for the test that runs them. */
    static List<String> caseNames() {
        List<String> names = new ArrayList<>();
        for (String vsf : new String[]{"dot", "euclidean", "cosine"}) {
            for (int size : PQ_SUBSPACE_SIZES) names.add("pq-" + vsf + "-" + size);
            for (int size : PQ_SUBSPACE_SIZES) names.add("pqpair-" + vsf + "-" + size);
        }
        for (String f : new String[]{"dot", "euclidean", "cosine"}) {
            for (int dim : EXACT_DIMS) names.add("exact-" + f + "-" + dim);
        }
        names.add("exact4-dot-768");
        names.add("exact4-euclidean-768");
        names.add("exact4-cosine-768");
        names.add("nvq-dot-768");
        names.add("nvq-euclidean-768");
        names.add("nvq-cosine-768");
        names.add("ash-masked-768");
        names.add("ash-masked-700");
        names.add("ash-lut-96");
        names.add("ash-projection2-768");
        names.add("ash-projection4-768");
        names.add("ash-offsetbinary3-768");
        for (int dim : new int[]{100, 768}) {
            names.add("int8-dot-" + dim);
            names.add("int8-euclidean-" + dim);
            names.add("int8-cosine-" + dim);
        }
        names.add("hamming-768");
        names.add("ash-symmetric1-768");
        names.add("ash-symmetric2-768");
        names.add("ash-symmetric4-768");
        names.add("ash-int8units-768");
        names.add("ash-dotrow-768");
        names.add("ash-projectall-768");
        names.add("ash-rows2-768");
        names.add("ash-rows4-768");
        names.add("ash-rows5-768");
        names.add("ash-int8neighborhood-768");
        names.add("ash-lutneighborhood-768");
        names.add("ash-canonical-768");
        names.add("ash-projectbatch-768");
        names.add("ash-int8project-768");
        names.add("ash-fp16project-768");
        for (String op : new String[]{"sum", "scale", "addinplace", "subinplace", "mininplace", "max", "min", "sub",
                                      "nvqquantize", "nvqloss", "nvquniformloss"}) {
            names.add("util-" + op + "-768");
        }
        return names;
    }

    // PQ subspace sizes: 96 dimensions in 96, 48, 32, 24, 6, 4 or 3 subspaces; 8 floats as 768 dimensions in 96.
    static final int[] PQ_SUBSPACE_SIZES = {1, 2, 3, 4, 8, 16, 24, 32};
    static final int[] EXACT_DIMS = {2, 8, 24, 25, 40, 100, 768};

    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorUtilSupport IMPL = VectorizationProvider.getInstance().getVectorUtilSupport();
    private static final String NAME = System.getProperty("probe.case", "");
    // With -Dprobe.mixed=true, every other case runs first, so the JDK's shared vector methods have seen every
    // register size before the measured case compiles, as in a process that holds indexes of several shapes.
    private static final boolean MIXED = Boolean.getBoolean("probe.mixed");
    private static final Case CASE = select(NAME);
    private static final Op OP = CASE.op;

    public static void main(String[] args) {
        if (IMPL.getClass() != PanamaVectorUtilSupport.class) {
            throw new IllegalStateException("Panama provider required, got " + IMPL.getClass().getName());
        }
        var threads = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        long warmNanos = Long.getLong("probe.warmMillis", 4000) * 1_000_000L;
        float sink = 0;
        if (MIXED) {
            for (String other : caseNames()) {
                if (other.equals(NAME)) continue;
                Op op = select(other).op;
                for (int k = 0; k < 20_000; k++) sink += op.run(k);
            }
        }
        // Warm up every root long enough for C2 to compile each one with the chain below it.
        long start = System.nanoTime();
        int i = 0;
        while (System.nanoTime() - start < warmNanos) {
            for (int depth = 0; depth <= MAX_DEPTH; depth++) sink += root(depth, i++);
        }
        // Then measure each root in a few rounds; the smallest round is the figure (it leaves out one-off
        // allocation such as a late recompilation's). The worst depth is the result.
        double worst = 0;
        int worstDepth = 0;
        StringBuilder perDepth = new StringBuilder();
        for (int depth = 0; depth <= MAX_DEPTH; depth++) {
            double best = Double.MAX_VALUE;
            for (int round = 0; round < 3; round++) {
                long ops = 0;
                long before = threads.getCurrentThreadAllocatedBytes();
                long roundStart = System.nanoTime();
                while (System.nanoTime() - roundStart < 50_000_000L) {
                    sink += root(depth, i++);
                    ops += ROOT_OPS;
                }
                long bytes = threads.getCurrentThreadAllocatedBytes() - before;
                best = Math.min(best, (double) bytes / ops);
            }
            perDepth.append(' ').append(depth).append(':').append(Math.round(best));
            if (best > worst) {
                worst = best;
                worstDepth = depth;
            }
        }
        System.out.println("RESULT " + NAME + " " + worst + " " + CASE.allowedBytes + " depth=" + worstDepth
                + " sink=" + (sink == 42 ? 1 : 0));
        System.out.println("DEPTHS " + NAME + perDepth);
    }

    // Where a caller sits in C2's inlining tree decides where the 15-level limit falls in the chain below it, and
    // so which call, if any, is left in place. Each root below is compiled on its own and reaches the operation
    // through a different number of calls (0 to MAX_DEPTH), covering the depths at which builds and searches reach
    // these kernels.
    static final int MAX_DEPTH = 10;
    static final int ROOT_OPS = 8;

    // Called through a megamorphic site (eleven classes), so no caller inlines a root: each is compiled on its own.
    private static final Op[] ROOTS = {
            PanamaAllocationProbe::r00, PanamaAllocationProbe::r01, PanamaAllocationProbe::r02,
            PanamaAllocationProbe::r03, PanamaAllocationProbe::r04, PanamaAllocationProbe::r05,
            PanamaAllocationProbe::r06, PanamaAllocationProbe::r07, PanamaAllocationProbe::r08,
            PanamaAllocationProbe::r09, PanamaAllocationProbe::r10,
    };

    private static float root(int depth, int i) {
        return ROOTS[depth].run(i);
    }

    private static float r00(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += OP.run(i * ROOT_OPS + k); return s; }
    private static float r01(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d10(i * ROOT_OPS + k); return s; }
    private static float r02(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d09(i * ROOT_OPS + k); return s; }
    private static float r03(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d08(i * ROOT_OPS + k); return s; }
    private static float r04(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d07(i * ROOT_OPS + k); return s; }
    private static float r05(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d06(i * ROOT_OPS + k); return s; }
    private static float r06(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d05(i * ROOT_OPS + k); return s; }
    private static float r07(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d04(i * ROOT_OPS + k); return s; }
    private static float r08(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d03(i * ROOT_OPS + k); return s; }
    private static float r09(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d02(i * ROOT_OPS + k); return s; }
    private static float r10(int i) { float s = 0; for (int k = 0; k < ROOT_OPS; k++) s += d01(i * ROOT_OPS + k); return s; }

    // Pass-through callers: d01 reaches the operation through ten calls, d10 through one.
    private static float d01(int i) { return d02(i); }
    private static float d02(int i) { return d03(i); }
    private static float d03(int i) { return d04(i); }
    private static float d04(int i) { return d05(i); }
    private static float d05(int i) { return d06(i); }
    private static float d06(int i) { return d07(i); }
    private static float d07(int i) { return d08(i); }
    private static float d08(int i) { return d09(i); }
    private static float d09(int i) { return d10(i); }
    private static float d10(int i) { return OP.run(i); }

    private static Case select(String name) {
        if (name.isEmpty()) return new Case(i -> 0, 0);
        String[] parts = name.split("-");
        int n = Integer.parseInt(parts[parts.length - 1]);
        Random random = new Random(42);
        switch (parts[0]) {
            case "pq": return pqCase(parts[1], n, random);
            case "pqpair": return pqPairCase(parts[1], n, random);
            case "exact": return exactCase(parts[1], n, random);
            case "exact4": return exact4Case(parts[1], n, random);
            case "nvq": return nvqCase(parts[1], n, random);
            case "ash": return ashCase(parts[1], n, random);
            case "int8": return int8Case(parts[1], n, random);
            case "hamming": return hammingCase(n, random);
            case "util": return utilCase(parts[1], n, random);
            default: throw new IllegalArgumentException(name);
        }
    }

    private static VectorFloat<?> randomVector(Random random, int dim) {
        var v = VTS.createFloatVector(dim);
        for (int i = 0; i < dim; i++) v.set(i, (float) random.nextGaussian());
        return v;
    }

    /** Builds a score function per operation, as each graph insertion and search does, and scores 16 nodes. */
    private static Case pqCase(String function, int subspaceSize, Random random) {
        int dim = subspaceSize == 8 ? 768 : 96;
        int count = 2000;
        List<VectorFloat<?>> data = new ArrayList<>();
        for (int i = 0; i < count; i++) data.add(randomVector(random, dim));
        var ravv = new ListRandomAccessVectorValues(data, dim);
        var pq = ProductQuantization.compute(ravv, dim / subspaceSize, 256, true);
        PQVectors pqv = pq.encodeAll(ravv, ForkJoinPool.commonPool());
        VectorSimilarityFunction vsf = function.equals("dot") ? VectorSimilarityFunction.DOT_PRODUCT
                : function.equals("euclidean") ? VectorSimilarityFunction.EUCLIDEAN : VectorSimilarityFunction.COSINE;
        var queries = new VectorFloat<?>[16];
        for (int i = 0; i < queries.length; i++) queries[i] = randomVector(random, dim);
        Op op = i -> {
            var sf = pqv.precomputedScoreFunctionFor(queries[i & 15], vsf);
            float sum = 0;
            for (int node = 0; node < 16; node++) sum += sf.similarityTo((i * 31 + node * 97) % count);
            return sum;
        };
        // The decoder and its centered copy of the query (header, array, wrapper), with room to spare.
        return new Case(op, 4.0 * dim + 512);
    }

    private static Case pqPairCase(String function, int subspaceSize, Random random) {
        int dim = subspaceSize == 8 ? 768 : 96;
        int count = 2000;
        List<VectorFloat<?>> data = new ArrayList<>();
        for (int i = 0; i < count; i++) data.add(randomVector(random, dim));
        var ravv = new ListRandomAccessVectorValues(data, dim);
        var pq = ProductQuantization.compute(ravv, dim / subspaceSize, 256, true);
        PQVectors pqv = pq.encodeAll(ravv, ForkJoinPool.commonPool());
        VectorSimilarityFunction vsf = function.equals("dot") ? VectorSimilarityFunction.DOT_PRODUCT
                : function.equals("euclidean") ? VectorSimilarityFunction.EUCLIDEAN : VectorSimilarityFunction.COSINE;
        Op op = i -> {
            var sf = pqv.diversityFunctionFor((i * 13) % count, vsf);
            float sum = 0;
            for (int node = 0; node < 16; node++) sum += sf.similarityTo((i * 31 + node * 97) % count);
            return sum;
        };
        // The scoring lambda, with room to spare.
        return new Case(op, 64);
    }

    private static Case exactCase(String function, int dim, Random random) {
        var a = new VectorFloat<?>[16];
        var b = new VectorFloat<?>[16];
        for (int i = 0; i < 16; i++) {
            a[i] = randomVector(random, dim);
            b[i] = randomVector(random, dim);
        }
        VectorSimilarityFunction vsf = function.equals("dot") ? VectorSimilarityFunction.DOT_PRODUCT
                : function.equals("euclidean") ? VectorSimilarityFunction.EUCLIDEAN : VectorSimilarityFunction.COSINE;
        return new Case(i -> vsf.compare(a[i & 15], b[(i >>> 4) & 15]), 8);
    }

    /** The four-target bulk kernels used when a build scores a node's neighbors. */
    private static Case exact4Case(String function, int dim, Random random) {
        var q = randomVector(random, dim);
        var targets = new VectorFloat<?>[8];
        for (int i = 0; i < targets.length; i++) targets[i] = randomVector(random, dim);
        float[] results = new float[8];
        Op op;
        switch (function) {
            case "dot": op = i -> { IMPL.dotProduct(q, targets, 8, results); return results[i & 7]; }; break;
            case "euclidean": op = i -> { IMPL.squareDistance(q, targets, 8, results); return results[i & 7]; }; break;
            default: op = i -> { IMPL.cosine(q, targets, 8, results); return results[i & 7]; }; break;
        }
        return new Case(op, 8);
    }

    private static Case nvqCase(String function, int dim, Random random) {
        var q = randomVector(random, dim);
        var centroid = randomVector(random, dim);
        ByteSequence<?> bytes = VTS.createByteSequence(dim);
        for (int i = 0; i < dim; i++) bytes.set(i, (byte) random.nextInt(256));
        float growth = 1.5f, mid = 0.1f, min = -2f, max = 2f;
        switch (function) {
            case "dot": return new Case(i -> VectorUtil.nvqDotProduct8bit(q, bytes, growth, mid, min, max), 8);
            case "euclidean": return new Case(i -> VectorUtil.nvqSquareL2Distance8bit(q, bytes, growth, mid, min, max), 8);
            // The API returns a new float[2]: 24 bytes by design.
            default: return new Case(i -> VectorUtil.nvqCosine8bit(q, bytes, growth, mid, min, max, centroid)[0], 24 + 8);
        }
    }

    private static Case int8Case(String function, int dim, Random random) {
        var a = new ByteSequence<?>[16];
        var b = new ByteSequence<?>[16];
        for (int i = 0; i < 16; i++) {
            a[i] = VTS.createByteSequence(dim);
            b[i] = VTS.createByteSequence(dim);
            for (int j = 0; j < dim; j++) {
                a[i].set(j, (byte) random.nextInt(256));
                b[i].set(j, (byte) random.nextInt(256));
            }
        }
        switch (function) {
            case "dot": return new Case(i -> IMPL.dotProduct(a[i & 15], b[(i >>> 4) & 15]), 8);
            case "euclidean": return new Case(i -> IMPL.squareDistance(a[i & 15], b[(i >>> 4) & 15]), 8);
            default: return new Case(i -> IMPL.cosine(a[i & 15], b[(i >>> 4) & 15]), 8);
        }
    }

    private static Case hammingCase(int dim, Random random) {
        long[][] bits = new long[16][(dim + 63) / 64];
        for (long[] b : bits) for (int i = 0; i < b.length; i++) b[i] = random.nextLong();
        return new Case(i -> VectorUtil.hammingDistance(bits[i & 15], bits[(i >>> 4) & 15]), 8);
    }

    /** Element-wise utilities used in training and encoding. */
    private static Case utilCase(String op, int dim, Random random) {
        var a = randomVector(random, dim);
        var b = randomVector(random, dim);
        ByteSequence<?> bytes = VTS.createByteSequence(dim);
        switch (op) {
            case "sum": return new Case(i -> IMPL.sum(a), 8);
            case "scale": return new Case(i -> { IMPL.scale(a, 1.0f); return a.get(i % dim); }, 8);
            case "addinplace": return new Case(i -> { IMPL.addInPlace(a, b); IMPL.subInPlace(a, b); return a.get(i % dim); }, 8);
            case "subinplace": return new Case(i -> { IMPL.subInPlace(a, 0.5f); IMPL.addInPlace(a, 0.5f); return a.get(i % dim); }, 8);
            case "mininplace": return new Case(i -> { IMPL.minInPlace(a, b); return a.get(i % dim); }, 8);
            case "max": return new Case(i -> IMPL.max(a), 8);
            case "min": return new Case(i -> IMPL.min(a), 8);
            // A new vector per call: its array (4 bytes per float, plus headers) is by design.
            case "sub": return new Case(i -> IMPL.sub(a, 0, b, 0, dim).get(i % dim), 4.0 * dim + 64);
            case "nvqquantize": return new Case(i -> { IMPL.nvqQuantize8bit(a, 1.5f, 0.1f, -4f, 4f, bytes); return bytes.get(i % dim); }, 8);
            case "nvqloss": return new Case(i -> IMPL.nvqLoss(a, 1.5f, 0.1f, -4f, 4f, 8), 8);
            default: return new Case(i -> IMPL.nvqUniformLoss(a, -4f, 4f, 8), 8);
        }
    }

    private static Case ashCase(String kernel, int dim, Random random) {
        float[] query = new float[dim];
        for (int i = 0; i < dim; i++) query[i] = (float) random.nextGaussian();
        switch (kernel) {
            case "masked": {
                int words = (dim + 63) / 64;
                long[] packed = new long[16 * words];
                for (int i = 0; i < packed.length; i++) packed[i] = random.nextLong();
                return new Case(i -> IMPL.ashMaskedAdd_512(query, 0, packed, (i & 15) * words, dim, words), 8);
            }
            case "lut": {
                int groups = dim, stride = 32, count = 32;
                byte[] codes = new byte[(groups / 2 + groups % 2) * stride];
                random.nextBytes(codes);
                float[] lut = new float[groups * 16];
                for (int i = 0; i < lut.length; i++) lut[i] = (float) random.nextGaussian();
                float[] out = new float[count];
                return new Case(i -> { IMPL.ashLutScore(codes, 0, groups, stride, 0, count, lut, out, 0); return out[i & 31]; }, 8);
            }
            case "projection2":
            case "projection4": {
                int bits = kernel.endsWith("2") ? 2 : 4;
                byte[] codes = new byte[16 * dim * bits / 8];
                random.nextBytes(codes);
                return new Case(i -> IMPL.ashProjectionDot(query, codes, (i & 15) * dim * bits / 8, dim, bits), 8);
            }
            case "symmetric1":
            case "symmetric2":
            case "symmetric4": {
                int bits = kernel.charAt(kernel.length() - 1) - '0';
                long[] aBits = new long[(dim + 63) / 64], bBits = new long[(dim + 63) / 64];
                for (int i = 0; i < aBits.length; i++) { aBits[i] = random.nextLong(); bBits[i] = random.nextLong(); }
                byte[] aCodes = new byte[dim * bits / 8], bCodes = new byte[dim * bits / 8];
                random.nextBytes(aCodes);
                random.nextBytes(bCodes);
                return new Case(i -> IMPL.ashSymmetricDot(aBits, aCodes, bBits, bCodes, dim, bits), 8);
            }
            case "int8units": {
                int bits = 5, units = dim / 8, stride = bits;
                byte[] codes = new byte[units * stride + 32];
                random.nextBytes(codes);
                int[] digits = new int[units * 8];
                for (int i = 0; i < digits.length; i++) digits[i] = random.nextInt(255) - 127;
                return new Case(i -> IMPL.ashOffsetBinaryInt8Dot(codes, 0, stride, units, bits, digits), 8);
            }
            case "dotrow": {
                float[] row = new float[dim];
                for (int i = 0; i < dim; i++) row[i] = (float) random.nextGaussian();
                return new Case(i -> IMPL.ashDotRow(row, query), 8);
            }
            case "projectall": {
                float[][] matrix = new float[dim / 2][dim];
                for (float[] row : matrix) for (int j = 0; j < dim; j++) row[j] = (float) random.nextGaussian();
                var projection = IMPL.prepareAshProjection(matrix);
                float[] out = new float[dim / 2];
                return new Case(i -> { projection.project(query, out); return out[i % out.length]; }, 8);
            }
            case "rows2":
            case "rows4":
            case "rows5": {
                int bits = kernel.charAt(kernel.length() - 1) - '0', bytes = dim * bits / 8, count = 32;
                byte[] q = new byte[bytes], rows = new byte[count * bytes];
                random.nextBytes(q);
                random.nextBytes(rows);
                float[] out = new float[count];
                return new Case(i -> { IMPL.ashSymmetricDotRows(q, 0, rows, bytes, count, dim, bits, out); return out[i & 31]; }, 8);
            }
            case "int8neighborhood": {
                int bits = 5, units = dim / 8, blockSize = 32, capacity = 32;
                byte[] codes = new byte[units * bits * blockSize + 5 * blockSize];
                random.nextBytes(codes);
                java.util.Arrays.fill(codes, units * bits * blockSize + 4 * blockSize, codes.length, (byte) 0);
                byte[] q = new byte[units * 8];
                for (int i = 0; i < q.length; i++) q[i] = (byte) (random.nextInt(255) - 127);
                float[] out = new float[capacity];
                return new Case(i -> { IMPL.ashScoreNeighborhoodInt8(codes, bits, units, blockSize, capacity, q, 0.01f, 3f,
                        0.2f, out); return out[i & 31]; }, 8);
            }
            case "lutneighborhood": {
                int groups = dim / 4, blockSize = 32, capacity = 32, body = (groups + 1) / 2 * blockSize;
                byte[] codes = new byte[body + 5 * blockSize];
                random.nextBytes(codes);
                java.util.Arrays.fill(codes, body + 4 * blockSize, codes.length, (byte) 0);
                float[] lut = new float[groups * 16];
                for (int i = 0; i < lut.length; i++) lut[i] = (float) random.nextGaussian();
                float[] out = new float[capacity];
                return new Case(i -> { IMPL.ashScoreNeighborhood(codes, groups, blockSize, capacity, lut, 0.2f, out);
                    return out[i & 31]; }, 8);
            }
            case "canonical": {
                int groups = dim / 4;
                byte[] codes = new byte[16 * ((groups + 1) / 2)];
                random.nextBytes(codes);
                float[] lut = new float[groups * 16];
                for (int i = 0; i < lut.length; i++) lut[i] = (float) random.nextGaussian();
                return new Case(i -> IMPL.ashCanonicalLutDot(codes, (i & 15) * ((groups + 1) / 2), groups, lut), 8);
            }
            case "projectbatch":
            case "int8project":
            case "fp16project": {
                float[][] matrix = new float[dim / 2][dim];
                for (float[] row : matrix) for (int j = 0; j < dim; j++) row[j] = (float) random.nextGaussian();
                var fp32 = IMPL.prepareAshProjection(matrix);
                var projection = kernel.equals("int8project")
                        ? fp32.forQueryScoring(io.github.jbellis.jvector.quantization.ASHQueryProjection.INT8)
                        : kernel.equals("fp16project")
                        ? fp32.forQueryScoring(io.github.jbellis.jvector.quantization.ASHQueryProjection.FP16) : fp32;
                float[][] inputs = new float[8][], outputs = new float[8][dim / 2];
                for (int k = 0; k < 8; k++) {
                    inputs[k] = new float[dim];
                    for (int j = 0; j < dim; j++) inputs[k][j] = (float) random.nextGaussian();
                }
                return new Case(i -> { projection.projectBatch(inputs, outputs, 8); return outputs[i & 7][i % (dim / 2)]; }, 8);
            }
            case "offsetbinary3": {
                byte[] codes = new byte[16 * dim * 3 / 8];
                random.nextBytes(codes);
                return new Case(i -> IMPL.ashOffsetBinaryDot(query, codes, (i & 15) * dim * 3 / 8, dim, 3), 8);
            }
            default: throw new IllegalArgumentException(kernel);
        }
    }
}

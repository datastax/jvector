package io.github.jbellis.jvector.quantization.ash;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class AshDriver1Bit implements AshDriver {

    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_512;
    private static final int BITS_PER_WORD = Long.SIZE;
    private static final int BIT_DEPTH = 1;

    private final int dimension;

    public AshDriver1Bit(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    @Override
    public int getBitDepth() {
        return BIT_DEPTH;
    }

    @Override
    public PackedVectors create(int n) {
        int wordsPerVector = wordsForDims(dimension);
        return new PanamaPackedVectors(new long[n][wordsPerVector]);
    }

    @Override
    public void packInts(int[] toPack, int toPackOffset, PackedVectors out, int pvOffset) {
        if (!(out instanceof PanamaPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be PanamaPackedVectors");
        }
        PanamaPackedVectors ppv = (PanamaPackedVectors) out;
        
        if (pvOffset >= ppv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        if (toPackOffset + dimension > toPack.length) {
            throw new IllegalArgumentException("toPackOffset + dimension exceeds toPack.length");
        }

        long[] binaryVector = ppv.vectors[pvOffset];
        final int words = wordsForDims(dimension);
        if (words != binaryVector.length) {
            throw new IllegalArgumentException("binaryVector length mismatch");
        }

        // Initialize to zero
        for (int i = 0; i < binaryVector.length; i++) {
            binaryVector[i] = 0L;
        }

        // Pack bits: for 1-bit, each int should be 0 or 1
        for (int w = 0; w < words; w++) {
            long bits = 0L;
            int base = w * BITS_PER_WORD;
            int rem = Math.min(BITS_PER_WORD, dimension - base);

            for (int j = 0; j < rem; j++) {
                int bitIndex = base + j;
                int qval = toPack[toPackOffset + bitIndex];
                assert ((qval & (-1 << BIT_DEPTH)) == 0);  // must be 0 or 1
                if (qval != 0) {
                    bits |= (1L << j);
                }
            }
            binaryVector[w] = bits;
        }
    }

    @Override
    public void unpackInts(PackedVectors packed, int pvOffset, int[] out, int outOffset) {
        if (!(packed instanceof PanamaPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be PanamaPackedVectors");
        }
        PanamaPackedVectors ppv = (PanamaPackedVectors) packed;
        
        if (pvOffset >= ppv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        final long[] binaryVector = ppv.vectors[pvOffset];
        final int words = wordsForDims(dimension);
        if (words != binaryVector.length) {
            throw new IllegalArgumentException("binaryVector length mismatch");
        }

        // Unpack bits: for 1-bit, extract each bit as 0 or 1
        for (int w = 0; w < words; w++) {
            long bits = binaryVector[w];
            int base = w * BITS_PER_WORD;
            int rem = Math.min(BITS_PER_WORD, dimension - base);

            for (int j = 0; j < rem; j++) {
                int idx = base + j;
                out[outOffset + idx] = (int) ((bits >>> j) & 1L);
            }
        }
    }

    @Override
    public float asymmetricScorePackedInts(PackedVectors packed, int pvOffset, float[] query, int qOffset) {
        if (!(packed instanceof PanamaPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be PanamaPackedVectors");
        }
        PanamaPackedVectors ppv = (PanamaPackedVectors) packed;
        
        if (pvOffset >= ppv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        final long[] binaryVector = ppv.vectors[pvOffset];
        final int words = wordsForDims(dimension);
        if (words != binaryVector.length) {
            throw new IllegalArgumentException("binaryVector length mismatch");
        }

        // Use the optimized Panama SIMD masked add kernel (copied from PanamaVectorUtilSupport)
        return ashMaskedAdd_512(query, qOffset, binaryVector, 0, dimension, words);
    }

    /**
     * SIMD-optimized masked add for 1-bit vectors.
     * Copied from PanamaVectorUtilSupport.ashMaskedAdd_512
     * TODO this currently only works for dimensions >= 128 which are multiples of 64
     */
    private float ashMaskedAdd_512(float[] tildeQ, int qOffset, long[] allPackedVectors, int packedBase, int d, int words) {
        final VectorSpecies<Float> SPEC = FLOAT_SPECIES;
        int safeWords = d / 64;

        // Determine unroll strategy based on d
        int numAcc = (d >= 512) ? 8 : (d == 384 ? 6 : (d >= 128 ? 4 : 2));

        // Initialize required accumulators
        FloatVector acc0 = FloatVector.zero(SPEC), acc1 = FloatVector.zero(SPEC);
        FloatVector acc2 = null, acc3 = null, acc4 = null, acc5 = null, acc6 = null, acc7 = null;

        if (numAcc >= 4) { acc2 = FloatVector.zero(SPEC); acc3 = FloatVector.zero(SPEC); }
        if (numAcc >= 6) { acc4 = FloatVector.zero(SPEC); acc5 = FloatVector.zero(SPEC); }
        if (numAcc >= 8) { acc6 = FloatVector.zero(SPEC); acc7 = FloatVector.zero(SPEC); }

        int w = 0;

        // Main unrolled loop
        int unrollFactor = (d == 384) ? 3 : (d >= 512 ? 2 : 1);

        for (; w <= safeWords - unrollFactor; w += unrollFactor) {
            int baseIdx = qOffset + (w << 6);

            // Word 0 processing
            long wd0 = allPackedVectors[packedBase + w];
            acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx),      VectorMask.fromLong(SPEC, wd0 & 0xFFFF));
            acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 16), VectorMask.fromLong(SPEC, (wd0 >>> 16) & 0xFFFF));
            acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 32), VectorMask.fromLong(SPEC, (wd0 >>> 32) & 0xFFFF));
            acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 48), VectorMask.fromLong(SPEC, (wd0 >>> 48)));

            if (unrollFactor >= 2) {
                long wd1 = allPackedVectors[packedBase + w + 1];
                int b1 = baseIdx + 64;
                if (numAcc == 6) { // Special case for d=384 interleave
                    acc4 = acc4.add(FloatVector.fromArray(SPEC, tildeQ, b1),      VectorMask.fromLong(SPEC, wd1 & 0xFFFF));
                    acc5 = acc5.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 16), VectorMask.fromLong(SPEC, (wd1 >>> 16) & 0xFFFF));
                    acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 32), VectorMask.fromLong(SPEC, (wd1 >>> 32) & 0xFFFF));
                    acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 48), VectorMask.fromLong(SPEC, (wd1 >>> 48)));
                } else { // Standard 8-accumulator path
                    acc4 = acc4.add(FloatVector.fromArray(SPEC, tildeQ, b1),      VectorMask.fromLong(SPEC, wd1 & 0xFFFF));
                    acc5 = acc5.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 16), VectorMask.fromLong(SPEC, (wd1 >>> 16) & 0xFFFF));
                    acc6 = acc6.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 32), VectorMask.fromLong(SPEC, (wd1 >>> 32) & 0xFFFF));
                    acc7 = acc7.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 48), VectorMask.fromLong(SPEC, (wd1 >>> 48)));
                }
            }

            if (unrollFactor == 3) { // Tail of the d=384 case
                long wd2 = allPackedVectors[packedBase + w + 2];
                int b2 = baseIdx + 128;
                acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, b2),      VectorMask.fromLong(SPEC, wd2 & 0xFFFF));
                acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, b2 + 16), VectorMask.fromLong(SPEC, (wd2 >>> 16) & 0xFFFF));
                acc4 = acc4.add(FloatVector.fromArray(SPEC, tildeQ, b2 + 32), VectorMask.fromLong(SPEC, (wd2 >>> 32) & 0xFFFF));
                acc5 = acc5.add(FloatVector.fromArray(SPEC, tildeQ, b2 + 48), VectorMask.fromLong(SPEC, (wd2 >>> 48)));
            }
        }

        // Cleanup loop
        for (; w < safeWords; w++) {
            int baseIdx = qOffset + (w << 6);
            long wd = allPackedVectors[packedBase + w];
            acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx),      VectorMask.fromLong(SPEC, wd & 0xFFFF));
            acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 16), VectorMask.fromLong(SPEC, (wd >>> 16) & 0xFFFF));
            if (numAcc >= 4) {
                acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 32), VectorMask.fromLong(SPEC, (wd >>> 32) & 0xFFFF));
                acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 48), VectorMask.fromLong(SPEC, (wd >>> 48)));
            }
        }

        // Dynamic reduction tree
        return performReduction(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, numAcc);
    }

    private float performReduction(FloatVector a0, FloatVector a1, FloatVector a2, FloatVector a3,
                                   FloatVector a4, FloatVector a5, FloatVector a6, FloatVector a7, int count) {
        FloatVector res;
        if (count == 8) {
            FloatVector s01 = a0.add(a1), s23 = a2.add(a3), s45 = a4.add(a5), s67 = a6.add(a7);
            res = s01.add(s23).add(s45.add(s67));
        } else if (count == 6) {
            res = a0.add(a1).add(a2.add(a3)).add(a4.add(a5));
        } else if (count == 4) {
            res = a0.add(a1).add(a2.add(a3));
        } else {
            res = a0.add(a1);
        }
        return res.reduceLanes(VectorOperators.ADD);
    }

    @Override
    public float symmetricScorePackedInts(PackedVectors a, int aOffset, PackedVectors b, int bOffset) {
        if (!(a instanceof PanamaPackedVectors) || !(b instanceof PanamaPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be PanamaPackedVectors");
        }
        PanamaPackedVectors ppvA = (PanamaPackedVectors) a;
        PanamaPackedVectors ppvB = (PanamaPackedVectors) b;
        
        if (aOffset >= ppvA.vectors.length || bOffset >= ppvB.vectors.length) {
            throw new IllegalArgumentException("Offset out of bounds");
        }

        final long[] binaryVectorA = ppvA.vectors[aOffset];
        final long[] binaryVectorB = ppvB.vectors[bOffset];
        
        // For 1-bit vectors, symmetric score is the popcount of (A AND B)
        long innerDot = 0;

        var wordCount = binaryVectorA.length;
        assert wordCount == binaryVectorB.length;

        for (int w = 0; w < wordCount; w++) {
            long andResult = binaryVectorA[w] & binaryVectorB[w];
            innerDot += Long.bitCount(andResult);
        }

        return (float) innerDot;
    }

    @Override
    public float getRawComponentSum(PackedVectors packed, int pvOffset) {
        if (!(packed instanceof PanamaPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be PanamaPackedVectors");
        }
        PanamaPackedVectors ppv = (PanamaPackedVectors) packed;
        
        if (pvOffset >= ppv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        final long[] binaryVector = ppv.vectors[pvOffset];
        
        // For 1-bit vectors, sum is the popcount of all bits
        long sum = 0;
        for (int w = 0; w < binaryVector.length; w++) {
            sum += Long.bitCount(binaryVector[w]);
        }
        
        return (float) sum;
    }

    private static int wordsForDims(int quantizedDim) {
        return (quantizedDim + BITS_PER_WORD - 1) / BITS_PER_WORD;
    }

    /**
     * Simple holder class for packed vectors data using long arrays (1 bit per dimension)
     */
    private static class PanamaPackedVectors implements PackedVectors {
        long[][] vectors;

        PanamaPackedVectors(long[][] vectors) {
            this.vectors = vectors;
        }
    }
}

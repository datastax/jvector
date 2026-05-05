package io.github.jbellis.jvector.quantization.ash;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

public class PanamaAshDriver2BitPacked implements AshDriver {

    private static final int INT_BITS = Integer.BYTES * 8;
    private static final int BIT_DEPTH = 2;

    private final int dimension;

    public PanamaAshDriver2BitPacked(int dimension) {
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
        int wordsPerVector = wordsForDims(dimension, BIT_DEPTH);
        return new PackedVectorsImpl(new int[n][wordsPerVector]);
    }

    @Override
    public void packInts(int[] toPack, int toPackOffset, PackedVectors out, int pvOffset) {
        if (!(out instanceof PackedVectorsImpl)) {
            throw new IllegalArgumentException("Invalid PackedVectors implementation");
        }
        PackedVectorsImpl spv = (PackedVectorsImpl) out;
        
        if (pvOffset >= spv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        if (toPackOffset + dimension > toPack.length) {
            throw new IllegalArgumentException("toPackOffset + dimension exceeds toPack.length");
        }

        int[] binaryVector = spv.vectors[pvOffset];
        final int words = wordsForDims(dimension, BIT_DEPTH);
        if (words != binaryVector.length) {
            throw new IllegalArgumentException("binaryVector length mismatch");
        }

        final int dimsPerWord = getDimsPerWord(BIT_DEPTH);

        // Pack integers into long words
        for (int w = 0; w < words; w++) {
            int bits = 0;
            int base = w * dimsPerWord;
            int rem = Math.min(dimsPerWord, dimension - base);

            for (int j = 0; j < rem; j++) {
                int bitIndex = base + j;
                int qval = toPack[toPackOffset + bitIndex];
                assert ((qval & (-1 << BIT_DEPTH)) == 0);  // must be in [0, 2^bitDepth)
                bits |= (qval << (j * BIT_DEPTH));
            }
            binaryVector[w] = bits;
        }
    }

    @Override
    public void unpackInts(PackedVectors packed, int pvOffset, int[] out, int outOffset) {
        if (!(packed instanceof PackedVectorsImpl)) {
            throw new IllegalArgumentException("PackedVectors must be ScalarPackedVectors");
        }
        PackedVectorsImpl spv = (PackedVectorsImpl) packed;
        
        if (pvOffset >= spv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        final int[] bits = spv.vectors[pvOffset];
        final int d = dimension;
        final int dimsPerWord = getDimsPerWord(BIT_DEPTH);
        final int pieceMask = (1 << BIT_DEPTH) - 1;

        int dimBase = 0;
        for (int w = 0; w < bits.length && dimBase < d; w++, dimBase += dimsPerWord) {
            int word = bits[w];
            int rem = Math.min(dimsPerWord, d - dimBase);

            for (int j = 0; j < rem; j++) {
                int idx = dimBase + j;
                int piece = word & pieceMask;
                out[outOffset + idx] = (int) piece;
                word >>>= BIT_DEPTH;
            }
        }
    }

    private static final IntVector SHIFT_VEC = IntVector.fromArray(
        IntVector.SPECIES_512,
        new int[]{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
        0
    );

    @Override
    public float asymmetricScorePackedInts(PackedVectors packed, int pvOffset, float[] query, int qOffset) {
        if (!(packed instanceof PackedVectorsImpl)) {
            throw new IllegalArgumentException("PackedVectors must be ScalarPackedVectors");
        }
        PackedVectorsImpl spv = (PackedVectorsImpl) packed;
        
        if (pvOffset >= spv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        final int[] bits = spv.vectors[pvOffset];
        final int dimsPerWord = getDimsPerWord(BIT_DEPTH);
        final int pieceMask = ~(~0 << BIT_DEPTH);

        FloatVector acc0 = FloatVector.zero(FloatVector.SPECIES_512);
        FloatVector acc1 = FloatVector.zero(FloatVector.SPECIES_512);
        FloatVector acc2 = FloatVector.zero(FloatVector.SPECIES_512);
        FloatVector acc3 = FloatVector.zero(FloatVector.SPECIES_512);

        final int UNROLL_FACTOR = 4;
        final int LOOP_INC = dimsPerWord * UNROLL_FACTOR;

        int i = 0;
        for (; i + LOOP_INC <= dimension; i += LOOP_INC) {

            int word0 = bits[i / dimsPerWord + 0];
            int word1 = bits[i / dimsPerWord + 1];
            int word2 = bits[i / dimsPerWord + 2];
            int word3 = bits[i / dimsPerWord + 3];

            var b0 = IntVector.broadcast(IntVector.SPECIES_512, word0);
            var b1 = IntVector.broadcast(IntVector.SPECIES_512, word1);
            var b2 = IntVector.broadcast(IntVector.SPECIES_512, word2);
            var b3 = IntVector.broadcast(IntVector.SPECIES_512, word3);

            var shifted0 = b0.lanewise(VectorOperators.LSHR, SHIFT_VEC);
            var shifted1 = b1.lanewise(VectorOperators.LSHR, SHIFT_VEC);
            var shifted2 = b2.lanewise(VectorOperators.LSHR, SHIFT_VEC);
            var shifted3 = b3.lanewise(VectorOperators.LSHR, SHIFT_VEC);

            var masked0 = shifted0.lanewise(VectorOperators.AND, pieceMask);
            var masked1 = shifted1.lanewise(VectorOperators.AND, pieceMask);
            var masked2 = shifted2.lanewise(VectorOperators.AND, pieceMask);
            var masked3 = shifted3.lanewise(VectorOperators.AND, pieceMask);

            var cvt0 = masked0.convert(VectorOperators.I2F, 0);
            var cvt1 = masked1.convert(VectorOperators.I2F, 0);
            var cvt2 = masked2.convert(VectorOperators.I2F, 0);
            var cvt3 = masked3.convert(VectorOperators.I2F, 0);

            var q0 = FloatVector.fromArray(FloatVector.SPECIES_512, query, qOffset + i + dimsPerWord * 0);
            var q1 = FloatVector.fromArray(FloatVector.SPECIES_512, query, qOffset + i + dimsPerWord * 1);
            var q2 = FloatVector.fromArray(FloatVector.SPECIES_512, query, qOffset + i + dimsPerWord * 2);
            var q3 = FloatVector.fromArray(FloatVector.SPECIES_512, query, qOffset + i + dimsPerWord * 3);

            acc0 = q0.fma(cvt0, acc0);
            acc1 = q1.fma(cvt1, acc1);
            acc2 = q2.fma(cvt2, acc2);
            acc3 = q3.fma(cvt3, acc3);
        }
        var acc = acc0.add(acc1).add(acc2.add(acc3));
        float vectorDot = acc.reduceLanes(VectorOperators.ADD);

        float residualDot = 0f;

        // Process residuals
        for (; i < dimension; i++) {
            int wordIndex = i / dimsPerWord;
            int bitPosition = (i % dimsPerWord) * BIT_DEPTH;
            int word = bits[wordIndex];
            int piece = (word >>> bitPosition) & pieceMask;
            residualDot += piece * query[qOffset + i];
        }

        return vectorDot + residualDot;
    }

    @Override
    public float symmetricScorePackedInts(PackedVectors a, int aOffset, PackedVectors b, int bOffset) {
        if (!(a instanceof PackedVectorsImpl) || !(b instanceof PackedVectorsImpl)) {
            throw new IllegalArgumentException("PackedVectors must be ScalarPackedVectors");
        }
        PackedVectorsImpl spvA = (PackedVectorsImpl) a;
        PackedVectorsImpl spvB = (PackedVectorsImpl) b;
        
        if (aOffset >= spvA.vectors.length || bOffset >= spvB.vectors.length) {
            throw new IllegalArgumentException("Offset out of bounds");
        }

        final int[] bitsA = spvA.vectors[aOffset];
        final int[] bitsB = spvB.vectors[bOffset];
        final int pieceMask = ~(~0 << BIT_DEPTH);

        long innerDot = 0;

        int wordCount = bitsA.length;
        assert wordCount == bitsB.length;

        for (int w = 0; w < wordCount; w++) {
            int word1 = bitsA[w];
            int word2 = bitsB[w];

            while (word1 != 0 && word2 != 0) {
                int piece1 = word1 & pieceMask;
                int piece2 = word2 & pieceMask;

                innerDot += piece1 * piece2;

                word1 >>>= BIT_DEPTH;
                word2 >>>= BIT_DEPTH;
            }
        }

        return (float) innerDot;
    }

    @Override
    public float getRawComponentSum(PackedVectors packed, int pvOffset) {
        if (!(packed instanceof PackedVectorsImpl)) {
            throw new IllegalArgumentException("PackedVectors must be ScalarPackedVectors");
        }
        PackedVectorsImpl spv = (PackedVectorsImpl) packed;
        
        if (pvOffset >= spv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        final int[] bits = spv.vectors[pvOffset];
        final int d = dimension;
        final int dimsPerWord = getDimsPerWord(BIT_DEPTH);
        final int pieceMask = ~(~0 << BIT_DEPTH);

        long sum = 0;

        int dimBase = 0;
        for (int w = 0; w < bits.length && dimBase < d; w++, dimBase += dimsPerWord) {
            int word = bits[w];
            int rem = Math.min(dimsPerWord, d - dimBase);

            for (int j = 0; j < rem; j++) {
                int piece = word & pieceMask;
                sum += piece;
                word >>>= BIT_DEPTH;
            }
        }

        return (float) sum;
    }

    private int wordsForDims(int quantizedDim, int bitDepth) {
        // 64 = Integer.BITS, 63 = Integer.BITS - 1
        return (quantizedDim * bitDepth + INT_BITS - 1) / INT_BITS;
    }

    private int getDimsPerWord(int bitDepth) {
        return INT_BITS / bitDepth;
    }

    /**
     * Simple holder class for packed vectors data
     */
    private static class PackedVectorsImpl implements PackedVectors {
        int[][] vectors;

        PackedVectorsImpl(int[][] vectors) {
            this.vectors = vectors;
        }
    }
}

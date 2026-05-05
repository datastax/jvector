package io.github.jbellis.jvector.quantization.ash;

public class ScalarAshDriver implements AshDriver {

    private final int bitDepth;
    private final int dimension;

    public ScalarAshDriver(int bitDepth, int dimension) {
        this.bitDepth = bitDepth;
        this.dimension = dimension;
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    @Override
    public int getBitDepth() {
        return bitDepth;
    }

    @Override
    public PackedVectors create(int n) {
        int wordsPerVector = wordsForDims(dimension, bitDepth);
        return new ScalarPackedVectors(new long[n][wordsPerVector]);
    }

    @Override
    public void packInts(int[] toPack, int toPackOffset, PackedVectors out, int pvOffset) {
        if (!(out instanceof ScalarPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be ScalarPackedVectors");
        }
        ScalarPackedVectors spv = (ScalarPackedVectors) out;
        
        if (pvOffset >= spv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        if (toPackOffset + dimension > toPack.length) {
            throw new IllegalArgumentException("toPackOffset + dimension exceeds toPack.length");
        }

        long[] binaryVector = spv.vectors[pvOffset];
        final int words = wordsForDims(dimension, bitDepth);
        if (words != binaryVector.length) {
            throw new IllegalArgumentException("binaryVector length mismatch");
        }

        final int dimsPerWord = getDimsPerWord(bitDepth);

        // Pack integers into long words
        for (int w = 0; w < words; w++) {
            long bits = 0L;
            int base = w * dimsPerWord;
            int rem = Math.min(dimsPerWord, dimension - base);

            for (int j = 0; j < rem; j++) {
                int bitIndex = base + j;
                int qval = toPack[toPackOffset + bitIndex];
                assert ((qval & (-1 << bitDepth)) == 0);  // must be in [0, 2^bitDepth)
                bits |= ((long) qval << (j * bitDepth));
            }
            binaryVector[w] = bits;
        }
    }

    @Override
    public void unpackInts(PackedVectors packed, int pvOffset, int[] out, int outOffset) {
        if (!(packed instanceof ScalarPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be ScalarPackedVectors");
        }
        ScalarPackedVectors spv = (ScalarPackedVectors) packed;
        
        if (pvOffset >= spv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        final long[] bits = spv.vectors[pvOffset];
        final int d = dimension;
        final int dimsPerWord = getDimsPerWord(bitDepth);
        final long pieceMask = (1L << bitDepth) - 1;

        int dimBase = 0;
        for (int w = 0; w < bits.length && dimBase < d; w++, dimBase += dimsPerWord) {
            long word = bits[w];
            int rem = Math.min(dimsPerWord, d - dimBase);

            for (int j = 0; j < rem; j++) {
                int idx = dimBase + j;
                long piece = word & pieceMask;
                out[outOffset + idx] = (int) piece;
                word >>>= bitDepth;
            }
        }
    }

    @Override
    public float asymmetricScorePackedInts(PackedVectors packed, int pvOffset, float[] query, int qOffset) {
        if (!(packed instanceof ScalarPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be ScalarPackedVectors");
        }
        ScalarPackedVectors spv = (ScalarPackedVectors) packed;
        
        if (pvOffset >= spv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        final long[] bits = spv.vectors[pvOffset];
        final int d = dimension;
        final int dimsPerWord = getDimsPerWord(bitDepth);
        final long pieceMask = (1L << bitDepth) - 1;

        float innerDot = 0f;

        int dimBase = 0;
        for (int w = 0; w < bits.length && dimBase < d; w++, dimBase += dimsPerWord) {
            long word = bits[w];
            int rem = Math.min(dimsPerWord, d - dimBase);

            for (int j = 0; j < rem; j++) {
                int idx = dimBase + j;
                long piece = word & pieceMask;
                float queryPiece = query[qOffset + idx];

                innerDot += piece * queryPiece;
                word >>>= bitDepth;
            }
        }

        return innerDot;
    }

    @Override
    public float symmetricScorePackedInts(PackedVectors a, int aOffset, PackedVectors b, int bOffset) {
        if (!(a instanceof ScalarPackedVectors) || !(b instanceof ScalarPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be ScalarPackedVectors");
        }
        ScalarPackedVectors spvA = (ScalarPackedVectors) a;
        ScalarPackedVectors spvB = (ScalarPackedVectors) b;
        
        if (aOffset >= spvA.vectors.length || bOffset >= spvB.vectors.length) {
            throw new IllegalArgumentException("Offset out of bounds");
        }

        final long[] bitsA = spvA.vectors[aOffset];
        final long[] bitsB = spvB.vectors[bOffset];
        final long pieceMask = ~(~0L << bitDepth);

        long innerDot = 0;

        int wordCount = bitsA.length;
        assert wordCount == bitsB.length;

        for (int w = 0; w < wordCount; w++) {
            long word1 = bitsA[w];
            long word2 = bitsB[w];

            while (word1 != 0 && word2 != 0) {
                long piece1 = word1 & pieceMask;
                long piece2 = word2 & pieceMask;

                innerDot += piece1 * piece2;

                word1 >>>= bitDepth;
                word2 >>>= bitDepth;
            }
        }

        return (float) innerDot;
    }

    @Override
    public float getRawComponentSum(PackedVectors packed, int pvOffset) {
        if (!(packed instanceof ScalarPackedVectors)) {
            throw new IllegalArgumentException("PackedVectors must be ScalarPackedVectors");
        }
        ScalarPackedVectors spv = (ScalarPackedVectors) packed;
        
        if (pvOffset >= spv.vectors.length) {
            throw new IllegalArgumentException("pvOffset out of bounds");
        }

        final long[] bits = spv.vectors[pvOffset];
        final int d = dimension;
        final int dimsPerWord = getDimsPerWord(bitDepth);
        final long pieceMask = (1L << bitDepth) - 1;

        long sum = 0;

        int dimBase = 0;
        for (int w = 0; w < bits.length && dimBase < d; w++, dimBase += dimsPerWord) {
            long word = bits[w];
            int rem = Math.min(dimsPerWord, d - dimBase);

            for (int j = 0; j < rem; j++) {
                long piece = word & pieceMask;
                sum += piece;
                word >>>= bitDepth;
            }
        }

        return (float) sum;
    }

    private int wordsForDims(int quantizedDim, int bitDepth) {
        // 64 = Long.BITS, 63 = Long.BITS - 1
        return (quantizedDim * bitDepth + 63) / 64;
    }

    private int getDimsPerWord(int bitDepth) {
        int longBits = Long.BYTES * 8;
        int dimsPerWord = longBits / bitDepth;
        assert longBits % bitDepth == 0;
        return dimsPerWord;
    }

    /**
     * Simple holder class for packed vectors data
     */
    private static class ScalarPackedVectors implements PackedVectors {
        long[][] vectors;

        ScalarPackedVectors(long[][] vectors) {
            this.vectors = vectors;
        }
    }
}

// Made with Bob

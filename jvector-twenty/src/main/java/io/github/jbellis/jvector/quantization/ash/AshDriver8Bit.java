package io.github.jbellis.jvector.quantization.ash;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class AshDriver8Bit implements AshDriver {

    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_512;
    private static final VectorSpecies<Integer> INT_SPECIES = IntVector.SPECIES_512;
    private static final int BYTES_PER_WORD = Integer.BYTES;
    private static final int BITS_PER_WORD = BYTES_PER_WORD * 8;
    private static final int STACK_WIDTH = 16;
    private static final int FLOATS_PER_VEC = 16;
    private static final int BIT_DEPTH = 8;
    private static final int PIECE_MASK = 0b1111_1111;

    private final int dimension;

    public AshDriver8Bit(int dimension) {
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
        return new PanamaPackedVectors(new int[n][wordsPerVector]);
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

        int[] binaryVector = ppv.vectors[pvOffset];
        final int words = wordsForDims(dimension);
        if (words != binaryVector.length) {
            throw new IllegalArgumentException("binaryVector length mismatch");
        }
        final int stacks = words / STACK_WIDTH;
        final int dimsPerWord = getDimsPerWord();

        // Initialize to zero
        for (int i = 0; i < binaryVector.length; i++) {
            binaryVector[i] = 0;
        }

        // Pack using stacked layout
        for (int stack = 0; stack < stacks; stack++) {
            int stackBase = stack * STACK_WIDTH * dimsPerWord;
            for (int lvl = 0; lvl < dimsPerWord; lvl++) {
                int base = stackBase + lvl * STACK_WIDTH;
                for (int wo = 0; wo < STACK_WIDTH; wo++) {
                    int bitIndex = base + wo;
                    if (bitIndex >= dimension) {
                        break;
                    }

                    int qval = toPack[toPackOffset + bitIndex];
                    assert ((qval & (-1 << BIT_DEPTH)) == 0);  // must be in [0, 2^bitDepth)

                    int w = stack * STACK_WIDTH + wo;

                    binaryVector[w] |= (qval << (lvl * BIT_DEPTH));
                }
            }
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

        final int[] binaryVector = ppv.vectors[pvOffset];
        final int words = wordsForDims(dimension);
        if (words != binaryVector.length) {
            throw new IllegalArgumentException("binaryVector length mismatch");
        }
        final int stacks = words / STACK_WIDTH;
        final int dimsPerWord = getDimsPerWord();

        // Unpack using stacked layout
        for (int stack = 0; stack < stacks; stack++) {
            int stackBase = stack * STACK_WIDTH * dimsPerWord;
            for (int lvl = 0; lvl < dimsPerWord; lvl++) {
                int base = stackBase + lvl * STACK_WIDTH;
                for (int wo = 0; wo < STACK_WIDTH; wo++) {
                    int bitIndex = base + wo;
                    if (bitIndex >= dimension) {
                        break;
                    }

                    int w = stack * STACK_WIDTH + wo;
                    int word = binaryVector[w];
                    int piece = (word >>> (lvl * BIT_DEPTH)) & PIECE_MASK;
                    out[outOffset + bitIndex] = piece;
                }
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

        final int[] binaryVector = ppv.vectors[pvOffset];
        final int words = wordsForDims(dimension);
        if (words != binaryVector.length) {
            throw new IllegalArgumentException("binaryVector length mismatch");
        }
        final int stacks = words / STACK_WIDTH;

        final int UNROLL_FACTOR = 4;

        var dotPieces0 = FloatVector.zero(FLOAT_SPECIES);
        var dotPieces1 = FloatVector.zero(FLOAT_SPECIES);
        var dotPieces2 = FloatVector.zero(FLOAT_SPECIES);
        var dotPieces3 = FloatVector.zero(FLOAT_SPECIES);
        var extraDot = 0.f;

        int i = 0;
        int stack = 0;
        for (; stack < stacks; stack++) {
            if (i + FLOATS_PER_VEC * UNROLL_FACTOR > dimension) {
                // Handle residual
                break;
            }
            var fullPiece = IntVector.fromArray(INT_SPECIES, binaryVector, stack * STACK_WIDTH);

            var shifted0 = fullPiece.lanewise(VectorOperators.LSHR, BIT_DEPTH * 0);
            var shifted1 = fullPiece.lanewise(VectorOperators.LSHR, BIT_DEPTH * 1);
            var shifted2 = fullPiece.lanewise(VectorOperators.LSHR, BIT_DEPTH * 2);
            var shifted3 = fullPiece.lanewise(VectorOperators.LSHR, BIT_DEPTH * 3);

            var masked0 = shifted0.lanewise(VectorOperators.AND, PIECE_MASK);
            var masked1 = shifted1.lanewise(VectorOperators.AND, PIECE_MASK);
            var masked2 = shifted2.lanewise(VectorOperators.AND, PIECE_MASK);
            var masked3 = shifted3.lanewise(VectorOperators.AND, PIECE_MASK);

            var piece0 = masked0.convert(VectorOperators.I2F, 0);
            var piece1 = masked1.convert(VectorOperators.I2F, 0);
            var piece2 = masked2.convert(VectorOperators.I2F, 0);
            var piece3 = masked3.convert(VectorOperators.I2F, 0);

            var queryPiece0 = FloatVector.fromArray(FLOAT_SPECIES, query, qOffset + i + FLOATS_PER_VEC * 0);
            var queryPiece1 = FloatVector.fromArray(FLOAT_SPECIES, query, qOffset + i + FLOATS_PER_VEC * 1);
            var queryPiece2 = FloatVector.fromArray(FLOAT_SPECIES, query, qOffset + i + FLOATS_PER_VEC * 2);
            var queryPiece3 = FloatVector.fromArray(FLOAT_SPECIES, query, qOffset + i + FLOATS_PER_VEC * 3);
            i += FLOATS_PER_VEC * UNROLL_FACTOR;

            dotPieces0 = queryPiece0.fma(piece0, dotPieces0);
            dotPieces1 = queryPiece1.fma(piece1, dotPieces1);
            dotPieces2 = queryPiece2.fma(piece2, dotPieces2);
            dotPieces3 = queryPiece3.fma(piece3, dotPieces3);
        }

        var dotPieces = dotPieces0.add(dotPieces1).add(dotPieces2.add(dotPieces3));

        // Handle residual elements
        if (stack < stacks || i < dimension) {
            for (; i < dimension; i++) {
                int stackIdx = i / (STACK_WIDTH * getDimsPerWord());
                int withinStack = i % (STACK_WIDTH * getDimsPerWord());
                int lvl = withinStack / STACK_WIDTH;
                int wo = withinStack % STACK_WIDTH;
                int w = stackIdx * STACK_WIDTH + wo;
                
                if (w < binaryVector.length) {
                    var word = binaryVector[w];
                    var piece = (word >>> (BIT_DEPTH * lvl)) & PIECE_MASK;
                    extraDot += piece * query[qOffset + i];
                }
            }
        }

        var innerDot = dotPieces.reduceLanes(VectorOperators.ADD) + extraDot;

        return innerDot;
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

        final int[] binaryVectorA = ppvA.vectors[aOffset];
        final int[] binaryVectorB = ppvB.vectors[bOffset];
        
        long innerDot = 0;

        var wordCount = binaryVectorA.length;
        assert wordCount == binaryVectorB.length;

        // Order doesn't matter for symmetric inner dot
        for (int w = 0; w < wordCount; w++) {
            int word1 = binaryVectorA[w];
            int word2 = binaryVectorB[w];

            while (word1 != 0 && word2 != 0) {
                var piece1 = word1 & PIECE_MASK;
                var piece2 = word2 & PIECE_MASK;

                innerDot += piece1 * piece2;

                word1 >>>= BIT_DEPTH;
                word2 >>>= BIT_DEPTH;
            }
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

        final int[] binaryVector = ppv.vectors[pvOffset];
        final int stacks = binaryVector.length / STACK_WIDTH;
        final int dimsPerWord = getDimsPerWord();
        
        long sum = 0;
        
        // Unpack and sum using stacked layout
        for (int stack = 0; stack < stacks; stack++) {
            int stackBase = stack * STACK_WIDTH * dimsPerWord;
            for (int lvl = 0; lvl < dimsPerWord; lvl++) {
                int base = stackBase + lvl * STACK_WIDTH;
                for (int wo = 0; wo < STACK_WIDTH; wo++) {
                    int bitIndex = base + wo;
                    if (bitIndex >= dimension) {
                        break;
                    }

                    int w = stack * STACK_WIDTH + wo;
                    int word = binaryVector[w];
                    int piece = (word >>> (lvl * BIT_DEPTH)) & PIECE_MASK;
                    sum += piece;
                }
            }
        }
        
        return (float) sum;
    }

    private static int wordsForDims(int quantizedDim) {
        var mw = minWordsForDims(quantizedDim);
        return (mw + STACK_WIDTH - 1) / STACK_WIDTH * STACK_WIDTH;
    }

    private static int minWordsForDims(int quantizedDim) {
        return ((quantizedDim * BIT_DEPTH) + (BITS_PER_WORD - 1)) / BITS_PER_WORD;
    }

    private int getDimsPerWord() {
        var longBits = BYTES_PER_WORD * 8;
        var dimsPerWord = longBits / BIT_DEPTH;
        assert longBits % BIT_DEPTH == 0;
        return dimsPerWord;
    }

    /**
     * Simple holder class for packed vectors data using int arrays
     */
    private static class PanamaPackedVectors implements PackedVectors {
        int[][] vectors;

        PanamaPackedVectors(int[][] vectors) {
            this.vectors = vectors;
        }
    }
}

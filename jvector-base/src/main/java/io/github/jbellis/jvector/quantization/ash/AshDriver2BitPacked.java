package io.github.jbellis.jvector.quantization.ash;

import java.io.IOException;
import java.util.Arrays;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;

public class AshDriver2BitPacked implements NbAshDriver {

    private static final int BIT_DEPTH = 2;
    private static final int BYTES_PER_WORD = Integer.BYTES;
    private static final int BITS_PER_WORD = BYTES_PER_WORD * 8;
    private static final int PIECE_MASK = 0b11;

    public AshDriver2BitPacked() {
    }

    @Override
    public AshQuantizedVector createEmptyVector(int quantizedDim) {
        return new QuantizedVector(Float.NaN, Float.NaN, (byte) -1, new int[wordsForDims(quantizedDim)]);
    }

    @Override
    public AbstractAshVectors<?> newVectors(AsymmetricHashing ash, AshQuantizedVector[] compressedVectors) {
        var vecs = Arrays.stream(compressedVectors)
            .map(v -> {
                if (!(v instanceof QuantizedVector)) {
                    throw new RuntimeException("Passed a vector of the wrong type");
                }
                return (QuantizedVector) v;
            })
            .toArray(QuantizedVector[]::new);
        return new ASHVectors(ash, vecs);
    }

    private int wordsForDims(int quantizedDim) {
        return ((quantizedDim * BIT_DEPTH) + (BITS_PER_WORD - 1)) / BITS_PER_WORD;
    }

    private int getDimsPerWord() {
        var longBits = BYTES_PER_WORD * 8;
        var dimsPerWord = longBits / BIT_DEPTH;
        assert longBits % BIT_DEPTH == 0;
        return dimsPerWord;
    }

    private class QuantizedVector extends AshQuantizedVector {

        int[] binaryVector;

        /** create an empty QuantizedVector */
        QuantizedVector(float scale, float offset, byte landmark, int[] binaryVector) {
            super(scale, offset, landmark);
            this.binaryVector = binaryVector;
        }

        @Override
        public void writeBinaryVector(IndexWriter out, int quantizedDim) throws IOException {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'writeBinaryVector'");
        }

        @Override
        public int getBitDepth() {
            return BIT_DEPTH;
        }

        @Override
        public void binarizeIntoFromQuantized(int[] quantized) {
            var quantizedDim = quantized.length;
            final int words = wordsForDims(quantizedDim);
            if (words != binaryVector.length) {
                throw new IllegalArgumentException("binaryVector length mismatch");
            }

            final int dimsPerWord = getDimsPerWord();

            // Binarize directly from per-row projection
            for (int w = 0; w < words; w++) {
                int bits = 0;
                int base = w * dimsPerWord;
                int rem = Math.min(dimsPerWord, quantizedDim - base);

                for (int j = 0; j < rem; j++) {
                    int bitIndex = base + j;
                    int qval = quantized[bitIndex];
                    assert ((qval & (-1 << BIT_DEPTH)) == 0);  // must be in [0, 2^bitDepth)
                    bits |= (qval << (j * BIT_DEPTH));
                }
                this.binaryVector[w] = bits;
            }
        }

        @Override
        public int getRawComponentSum() {
            int sum = 0;

            for (int w = 0; w < binaryVector.length; w++) {
                int word = binaryVector[w];

                while (word != 0) {
                    var piece = word & PIECE_MASK;
                    sum += piece;
                    word >>>= BIT_DEPTH;
                }
            }

            return sum;
        }

        @Override
        public int[] toRawComponents(int quantizedDim) {
            final int words = wordsForDims(quantizedDim);
            final int dimsPerWord = getDimsPerWord();
            if (words != binaryVector.length) {
                throw new IllegalArgumentException("binaryVector length mismatch");
            }

            int[] components = new int[quantizedDim];
            int i = 0;
            for (int w = 0; w < binaryVector.length; w++) {
                var word = binaryVector[w];
                int rem = Math.min(dimsPerWord, quantizedDim - i);
                for (int j = 0; j < rem; j++) {
                    var piece = word & PIECE_MASK;
                    components[i] = (int) piece;
                    i++;
                    word >>>= BIT_DEPTH;
                }
            }

            return components;
        }
    }

    private class ASHVectors extends AbstractAshVectors<QuantizedVector> {
        public ASHVectors(AsymmetricHashing ash, QuantizedVector[] vectors) {
            super(ash, vectors);
        }

        @Override
        public long ramBytesUsed() {
            // TODO
            return 0;
        }

        @Override
        public AbstractAshScoreFunction<QuantizedVector> createScoreFunction(AshQueryPrecompute qp) {
            // return new AshScoreFunction(qp);
            return new AbstractAshScoreFunction<QuantizedVector>(qp) {

                @Override
                public float calcInnerDot(QuantizedVector v, float[] queryPool, int queryPoolOffset, int quantizedDim) {
                    return calcInnerDotAlias(v, queryPool, queryPoolOffset, quantizedDim);
                }

                @Override
                public int getBitDepth() {
                    return BIT_DEPTH;
                }
                
            };
        }

        /** just an alias, to avoid accidental recursion in createScoreFunction */
        private float calcInnerDotAlias(QuantizedVector v, float[] queryPool, int queryPoolOffset, int quantizedDim) {
            return calcInnerDot(v, queryPool, queryPoolOffset, quantizedDim);
        }

        @Override
        public float calcInnerDot(QuantizedVector v, float[] queryPool, int queryPoolOffset, int quantizedDim) {
            final var d = quantizedDim;
            final int dimsPerWord = getDimsPerWord();

            float innerDot = 0f;

            final int[] bits = v.binaryVector;
            int dimBase = 0;
            for (int w = 0; w < bits.length && dimBase < d; w++, dimBase += dimsPerWord) {
                long word = bits[w];
                int rem = Math.min(dimsPerWord, d - dimBase);

                for (int j = 0; j < rem; j++) {
                    int idx = dimBase + j;
                    var piece = word & PIECE_MASK;
                    var queryPiece = queryPool[queryPoolOffset + idx];

                    innerDot += piece * queryPiece;
                    word >>>= BIT_DEPTH;
                }
            }

            return innerDot;
        }

        @Override
        public float calcSymmetricInnerDot(QuantizedVector v1, QuantizedVector v2) {
            long innerDot = 0;

            var wordCount = v1.binaryVector.length;
            assert wordCount == v2.binaryVector.length;

            for (int w = 0; w < wordCount; w++) {
                int word1 = v1.binaryVector[w];
                int word2 = v2.binaryVector[w];

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
    }
}


package io.github.jbellis.jvector.quantization.ash;

import java.io.IOException;
import java.util.Arrays;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;

public class AshFactory1BitAlt implements NbAshFactory {

    public static final int bitDepth = 1;

    @Override
    public AshQuantizedVector createEmptyVector(int quantizedDim) {
        return QuantizedVector.createEmpty(quantizedDim);
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

    private static class QuantizedVector extends AshQuantizedVector {

        private final long[] binaryVector;

        private static int wordsForDims(int quantizedDim) {
            return (quantizedDim + 63) / 64;
        }

        public QuantizedVector(float scale, float offset, byte landmark, long[] binaryVector) {
            super(scale, offset, landmark);
            this.binaryVector = binaryVector;
        }

        public static QuantizedVector createEmpty(int quantizedDim) {
            var binaryVector = new long[wordsForDims(quantizedDim)];
            return new QuantizedVector(Float.NaN, Float.NaN, (byte) -1, binaryVector);
        }

        @Override
        public void writeBinaryVector(IndexWriter out, int quantizedDim) throws IOException {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'writeBinaryVector'");
        }

        @Override
        public int getBitDepth() {
            return bitDepth;
        }

        @Override
        public void binarizeIntoFromQuantized(int[] quantized) {
            var quantizedDim = quantized.length;
            final int words = wordsForDims(quantizedDim);
            if (words != binaryVector.length) {
                throw new IllegalArgumentException("binaryVector length mismatch");
            }

            // Binarize directly from per-row projection
            for (int w = 0; w < words; w++) {
                long bits = 0L;
                int base = w << 6;
                int rem = Math.min(64, quantizedDim - base);

                for (int j = 0; j < rem; j++) {
                    int bitIndex = base + j;
                    int qval = quantized[bitIndex];
                    assert ((qval & (-1 << 1)) == 0);  // must be 0 or 1
                    bits |= ((long) qval << j);
                }
                this.binaryVector[w] = bits;
            }
        }

        @Override
        public int getRawComponentSum() {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'getRawComponentSum'");
        }
    }

    private static class ASHVectors extends AbstractAshVectors<QuantizedVector> {

        public ASHVectors(AsymmetricHashing ash, QuantizedVector[] vectors) {
            super(ash, vectors);
        }

        @Override
        public long ramBytesUsed() {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'ramBytesUsed'");
        }

        @Override
        public AbstractAshScoreFunction<QuantizedVector> createScoreFunction(AshQueryPrecompute qp) {
            return new AbstractAshScoreFunction<QuantizedVector>(qp) {
                @Override
                public float calcInnerDot(QuantizedVector v, float[] queryPool, int queryPoolOffset, int quantizedDim) {
                    return calcInnerDotAlias(v, queryPool, queryPoolOffset, quantizedDim);
                }

                @Override
                public int getBitDepth() {
                    return bitDepth;
                }
                
            };
        }

        private float calcInnerDotAlias(QuantizedVector v, float[] queryPool, int queryPoolOffset, int quantizedDim) {
            return calcInnerDot(v, queryPool, queryPoolOffset, quantizedDim);
        }

        @Override
        public float calcInnerDot(QuantizedVector v, float[] queryPool, int queryPoolOffset, int quantizedDim) {

            final var d = quantizedDim;

            // maskedAdd = <q̃_c, b>
            // b ∈ {0,1}^d stored as packed longs
            float maskedAdd = 0f;

            final long[] bits = v.binaryVector;
            int dimBase = 0;
            for (int w = 0; w < bits.length && dimBase < d; w++, dimBase += 64) {
                long word = bits[w];
                while (word != 0L) {
                    int bit = Long.numberOfTrailingZeros(word);
                    int idx = dimBase + bit;
                    if (idx < d) {
                        maskedAdd += queryPool[queryPoolOffset + idx];
                    }
                    word &= (word - 1);
                }
            }

            return maskedAdd;
        }

        @Override
        public float calcSymmetricInnerDot(QuantizedVector v1, QuantizedVector v2) {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'calcSymmetricInnerDot'");
        }
    }
}

package io.github.jbellis.jvector.quantization.ash;

import java.io.IOException;
import java.util.Arrays;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;

class NbAshDriverTemplate implements NbAshDriver {

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

        private QuantizedVector(float scale, float offset, byte landmark) {
            super(scale, offset, landmark);
        }

        public static QuantizedVector createEmpty(int quantizedDim) {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'createEmpty'");
        }

        @Override
        public void writeBinaryVector(IndexWriter out, int quantizedDim) throws IOException {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'writeBinaryVector'");
        }

        @Override
        public int getBitDepth() {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'getBitDepth'");
        }

        @Override
        public void binarizeIntoFromQuantized(int[] quantized) {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'binarizeIntoFromQuantized'");
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
                    // TODO Auto-generated method stub
                    throw new UnsupportedOperationException("Unimplemented method 'getBitDepth'");
                }
                
            };
        }

        private float calcInnerDotAlias(QuantizedVector v, float[] queryPool, int queryPoolOffset, int quantizedDim) {
            return calcInnerDot(v, queryPool, queryPoolOffset, quantizedDim);
        }

        @Override
        public float calcInnerDot(QuantizedVector v, float[] queryPool, int queryPoolOffset, int quantizedDim) {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'calcInnerDot'");
        }

        @Override
        public float calcSymmetricInnerDot(QuantizedVector v1, QuantizedVector v2) {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'calcSymmetricInnerDot'");
        }
    }
}

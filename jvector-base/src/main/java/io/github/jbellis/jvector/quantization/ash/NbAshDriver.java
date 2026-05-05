package io.github.jbellis.jvector.quantization.ash;

import io.github.jbellis.jvector.quantization.AsymmetricHashing;

public interface NbAshDriver {
    public AshQuantizedVector createEmptyVector(int quantizedDim);

    // TODO this forces a downcast per vector
    public AbstractAshVectors<?> newVectors(AsymmetricHashing ash, AshQuantizedVector[] compressedVectors);

    public static NbAshDriver get(int bitDepth) {
        // TODO maybe cache instance?
        switch (bitDepth) {
            case 1:
                return new AshDriver1BitOriginal();
            default:
                return new AshDriverDBit(bitDepth);
        }
    }
}

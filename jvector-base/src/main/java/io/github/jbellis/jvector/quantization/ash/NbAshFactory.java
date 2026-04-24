package io.github.jbellis.jvector.quantization.ash;

import io.github.jbellis.jvector.quantization.AsymmetricHashing;

public interface NbAshFactory {
    public AshQuantizedVector createEmptyVector(int quantizedDim);

    // TODO this forces a downcast per vector
    public AbstractAshVectors<?> newVectors(AsymmetricHashing ash, AshQuantizedVector[] compressedVectors);

    public static NbAshFactory get(int bitDepth) {
        // TODO maybe cache instance?
        switch (bitDepth) {
            case 1:
                return new AshFactoryDBit(1);
                // return new AshFactory1BitAlt();
                // return new AshFactory1Bit();
            case 2:
                return new AshFactoryDBit(2);
            default:
                return new AshFactoryDBit(bitDepth);
        }
    }
}

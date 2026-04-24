package io.github.jbellis.jvector.quantization.ash;

import java.util.Arrays;

import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;

public class AshFactory1Bit implements NbAshFactory {
    @Override
    public AshQuantizedVector createEmptyVector(int quantizedDim) {
        return AsymmetricHashing.QuantizedVector.createEmpty(quantizedDim);
    }

    @Override
    public AbstractAshVectors<?> newVectors(AsymmetricHashing ash, AshQuantizedVector[] compressedVectors) {
        var vecs = Arrays.stream(compressedVectors)
            .map(v -> {
                if (!(v instanceof AsymmetricHashing.QuantizedVector)) {
                    throw new RuntimeException("Passed a vector of the wrong type");
                }
                return (AsymmetricHashing.QuantizedVector) v;
            })
            .toArray(AsymmetricHashing.QuantizedVector[]::new);
        return new ASHVectors(ash, vecs);
    }
}

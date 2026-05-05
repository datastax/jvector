package io.github.jbellis.jvector.quantization.ash;

public interface AshDriverFactory {
    /**
     * @param bitDepth number of bits per quantized dimension
     * @param dimension the quantized dimension
     */
    AshDriver createDriver(int bitDepth, int dimension);
}

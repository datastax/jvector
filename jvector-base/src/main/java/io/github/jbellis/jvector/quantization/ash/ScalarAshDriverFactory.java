package io.github.jbellis.jvector.quantization.ash;

public class ScalarAshDriverFactory implements AshDriverFactory {

    @Override
    public AshDriver createDriver(int bitDepth, int dimension) {
        switch (bitDepth) {
            case 1:
            case 2:
            case 4:
            case 8:
                return new ScalarAshDriver(bitDepth, dimension);
            default:
                throw new UnsupportedOperationException("Unsupported bit depth: " + bitDepth + " for ScalarAshDriver");
        }
    }
}

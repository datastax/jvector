package io.github.jbellis.jvector.quantization.ash;

public class PanamaAshDriverFactory implements AshDriverFactory {

    @Override
    public AshDriver createDriver(int bitDepth, int dimension) {
        switch (bitDepth) {
            case 1:
                return new AshDriver1Bit(dimension);
            case 2:
                // return new AshDriver2Bit(dimension);
                return new PanamaAshDriver2BitPacked(dimension);
            case 4:
                return new AshDriver4Bit(dimension);
            case 8:
                return new AshDriver8Bit(dimension);
            default:
                throw new UnsupportedOperationException("Unsupported bit depth: " + bitDepth);
        }
    }
}

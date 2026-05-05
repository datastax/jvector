package io.github.jbellis.jvector.quantization.ash;

import java.util.stream.Stream;

import io.github.jbellis.jvector.vector.PanamaVectorizationProvider;
import io.github.jbellis.jvector.vector.VectorizationProvider;

public class PanamaAshDriverTest extends AbstractAshDriverTest {

    private static final VectorizationProvider vts = new PanamaVectorizationProvider();

    @Override
    protected Stream<VectorizationProvider> vtsToTest() {
        assert vts.getAshDriverFactory() instanceof PanamaAshDriverFactory;
        return Stream.of(vts);
    }

    @Override
    protected Stream<Integer> bitDepthsToTest() {
        // TODO include test for 1-bit once the dimension limitations are fixed
        return Stream.of(2, 4, 8);
    }
}

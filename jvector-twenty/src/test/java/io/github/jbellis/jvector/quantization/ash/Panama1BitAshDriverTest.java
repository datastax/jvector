package io.github.jbellis.jvector.quantization.ash;

import java.util.stream.Stream;

import io.github.jbellis.jvector.vector.PanamaVectorizationProvider;
import io.github.jbellis.jvector.vector.VectorizationProvider;

public class Panama1BitAshDriverTest extends AbstractAshDriverTest {
   
    private static final VectorizationProvider vts = new PanamaVectorizationProvider();

    @Override
    protected Stream<VectorizationProvider> vtsToTest() {
        assert vts.getAshDriverFactory() instanceof PanamaAshDriverFactory;
        return Stream.of(vts);
    }

    @Override
    protected Stream<Integer> bitDepthsToTest() {
        return Stream.of(1);
    } 

    // TODO 1 bit ASH currently only supports dimesnions >= 128 that are multiples of 64
    @Override
    protected Stream<Integer> dimensionsToTest() {
        return Stream.of(128, 256, 512, 512 + 64, 1024, 2048, 4096);
    }
}

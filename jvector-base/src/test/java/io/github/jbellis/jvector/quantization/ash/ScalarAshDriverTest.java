package io.github.jbellis.jvector.quantization.ash;

import java.util.stream.IntStream;
import java.util.stream.Stream;

import io.github.jbellis.jvector.vector.DefaultVectorizationProvider;
import io.github.jbellis.jvector.vector.VectorizationProvider;

public class ScalarAshDriverTest extends AbstractAshDriverTest {

    @Override
    protected Stream<VectorizationProvider> vtsToTest() {
        var defaultVts = new DefaultVectorizationProvider();
        assert defaultVts.getAshDriverFactory() instanceof ScalarAshDriverFactory;
        return Stream.of(defaultVts);
    }

    @Override
    protected Stream<Integer> bitDepthsToTest() {
        // return IntStream.range(1, 8 + 1).boxed();
        // return Stream.of(7);
        return Stream.of(1, 2, 4, 8);
    }
}

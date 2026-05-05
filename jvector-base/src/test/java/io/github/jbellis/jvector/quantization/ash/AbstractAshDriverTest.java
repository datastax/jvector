package io.github.jbellis.jvector.quantization.ash;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.Stream;

import io.github.jbellis.jvector.vector.VectorizationProvider;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.*;

// TODO clean up unnecessary tests
// TODO test score functions under multiple vectors
// TODO include tests for query vector with offset
// PER_CLASS lifecycle required for MethodSource targets to be non-static
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public abstract class AbstractAshDriverTest {

    // TODO no longer need to stream the entire VectorizationProvider now that VectorFloat<?> is not part of the api
    protected abstract Stream<VectorizationProvider> vtsToTest();
    protected abstract Stream<Integer> bitDepthsToTest();

    protected Stream<Integer> dimensionsToTest() {
        return Stream.of(1, 2, 3, 4, 5, 10, 100, 113, 128, 202, 256, 512, 617, 1024, 1536, 2048, 3096, 4000);
        // return Stream.of(128, 202, 256, 512, 617, 1024, 1536, 2048, 3096, 4000);
        // return Stream.of(128, 256, 512, 512 + 63, 1024, 2048, 4096);
        // return Stream.of(128);
        // return Stream.of(10);
    }

    private Stream<Arguments> combinations() {
        return bitDepthsToTest()
            .flatMap(bitDepth -> vtsToTest()
                .flatMap(factory -> dimensionsToTest().map(dim -> Arguments.of(factory, bitDepth, dim)))
            );
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testGetDimensionAndBitDepth(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);

        assertEquals(bitDepth, driver.getBitDepth(), "getBitDepth() should return the configured bit depth");
        assertEquals(dimension, driver.getDimension(), "getDimension() should return the configured dimension");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testCreatePackedVectors(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        
        // Test creating packed vectors with various sizes
        for (int n : new int[]{1, 5, 10, 100}) {
            var packed = driver.create(n);
            assertNotNull(packed, "create() should return non-null PackedVectors");
        }
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testPackUnpackRoundTrip(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        // Create test data with values in valid range [0, 2^bitDepth)
        int maxValue = (1 << bitDepth) - 1;
        int[] original = new int[dimension];
        Random rand = new Random(42);
        for (int i = 0; i < dimension; i++) {
            original[i] = rand.nextInt(maxValue + 1);
        }
        
        // Pack and unpack
        driver.packInts(original, 0, packed, 0);
        int[] unpacked = new int[dimension];
        driver.unpackInts(packed, 0, unpacked, 0);
        
        assertArrayEquals(original, unpacked, "Unpacked values should match original");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testPackUnpackWithMultipleVectors(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(3);
        
        int maxValue = (1 << bitDepth) - 1;
        Random rand = new Random(42);
        
        // Pack three different vectors at different offsets
        int[][] originals = new int[3][dimension];
        for (int v = 0; v < 3; v++) {
            for (int i = 0; i < dimension; i++) {
                originals[v][i] = rand.nextInt(maxValue + 1);
            }
            driver.packInts(originals[v], 0, packed, v);
        }
        
        // Unpack and verify each one
        for (int v = 0; v < 3; v++) {
            int[] unpacked = new int[dimension];
            driver.unpackInts(packed, v, unpacked, 0);
            assertArrayEquals(originals[v], unpacked, 
                "Unpacked values at offset " + v + " should match original");
        }
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testPackUnpackAllZeros(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int[] original = new int[dimension]; // All zeros
        driver.packInts(original, 0, packed, 0);
        
        int[] unpacked = new int[dimension];
        driver.unpackInts(packed, 0, unpacked, 0);
        
        assertArrayEquals(original, unpacked, "All zeros should round-trip correctly");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testPackUnpackAllMaxValues(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int maxValue = (1 << bitDepth) - 1;
        int[] original = new int[dimension];
        Arrays.fill(original, maxValue);
        
        driver.packInts(original, 0, packed, 0);
        int[] unpacked = new int[dimension];
        driver.unpackInts(packed, 0, unpacked, 0);
        
        assertArrayEquals(original, unpacked, "All max values should round-trip correctly");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testPackUnpackAlternatingPattern(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int maxValue = (1 << bitDepth) - 1;
        int[] original = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            original[i] = (i % 2 == 0) ? 0 : maxValue;
        }
        
        driver.packInts(original, 0, packed, 0);
        int[] unpacked = new int[dimension];
        driver.unpackInts(packed, 0, unpacked, 0);

        // System.out.println(Arrays.toString(original));
        // System.out.println(Arrays.toString(unpacked));
        
        assertArrayEquals(original, unpacked, "Alternating pattern should round-trip correctly");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testPackWithInputOffset(VectorizationProvider vts, int bitDepth, int dimension) {
        // Skip for very small dimensions
        // TODO don't
        Assumptions.assumeTrue(dimension < 10);

        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int maxValue = (1 << bitDepth) - 1;
        Random rand = new Random(42);
        
        // Create array larger than needed
        int[] largeArray = new int[dimension + 20];
        for (int i = 0; i < largeArray.length; i++) {
            largeArray[i] = rand.nextInt(maxValue + 1);
        }
        
        // Pack from offset 10
        driver.packInts(largeArray, 10, packed, 0);
        
        // Unpack and verify
        int[] unpacked = new int[dimension];
        driver.unpackInts(packed, 0, unpacked, 0);
        
        int[] expected = Arrays.copyOfRange(largeArray, 10, 10 + dimension);
        assertArrayEquals(expected, unpacked, "Should pack from correct offset");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testUnpackWithOutputOffset(VectorizationProvider vts, int bitDepth, int dimension) {
        // Skip for very small dimensions
        // TODO don't
        Assumptions.assumeTrue(dimension < 10);

        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int maxValue = (1 << bitDepth) - 1;
        Random rand = new Random(42);
        
        int[] original = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            original[i] = rand.nextInt(maxValue + 1);
        }
        
        driver.packInts(original, 0, packed, 0);
        
        // Unpack to offset 5 in a larger array
        int[] largeOutput = new int[dimension + 10];
        Arrays.fill(largeOutput, -1); // Fill with sentinel value
        driver.unpackInts(packed, 0, largeOutput, 5);
        
        // Verify unpacked values at offset
        for (int i = 0; i < dimension; i++) {
            assertEquals(original[i], largeOutput[5 + i], 
                "Value at index " + i + " should match");
        }
        
        // Verify sentinel values are unchanged
        for (int i = 0; i < 5; i++) {
            assertEquals(-1, largeOutput[i], "Sentinel before offset should be unchanged");
        }
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testAsymmetricScoreBasic(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int maxValue = (1 << bitDepth) - 1;
        int[] toPack = new int[dimension];
        Random rand = new Random(42);
        for (int i = 0; i < dimension; i++) {
            toPack[i] = rand.nextInt(maxValue + 1);
        }
        driver.packInts(toPack, 0, packed, 0);
        
        // Create query vector
        float[] queryData = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            queryData[i] = rand.nextFloat() * 10;
        }

        float score = driver.asymmetricScorePackedInts(packed, 0, queryData, 0);
        assertTrue(Float.isFinite(score), "Score should be finite");
        assertTrue(score >= 0, "Score should be non-negative for non-negative inputs");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testAsymmetricScoreZeroVector(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int[] toPack = new int[dimension]; // All zeros
        driver.packInts(toPack, 0, packed, 0);
        
        float[] queryData = new float[dimension];
        Random rand = new Random(42);
        for (int i = 0; i < dimension; i++) {
            queryData[i] = rand.nextFloat() * 10;
        }
        
        float score = driver.asymmetricScorePackedInts(packed, 0, queryData, 0);
        assertEquals(0.0f, score, 1e-6f, "Score should be zero when packed vector is all zeros");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testAsymmetricScoreZeroQuery(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int maxValue = (1 << bitDepth) - 1;
        int[] toPack = new int[dimension];
        Random rand = new Random(42);
        for (int i = 0; i < dimension; i++) {
            toPack[i] = rand.nextInt(maxValue + 1);
        }
        driver.packInts(toPack, 0, packed, 0);
        
        float[] queryData = new float[dimension]; // All zeros
        
        float score = driver.asymmetricScorePackedInts(packed, 0, queryData, 0);
        assertEquals(0.0f, score, 1e-6f, "Score should be zero when query is all zeros");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testAsymmetricScoreConsistency(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int maxValue = (1 << bitDepth) - 1;
        int[] toPack = new int[dimension];
        Random rand = new Random(42);
        for (int i = 0; i < dimension; i++) {
            toPack[i] = rand.nextInt(maxValue + 1);
        }
        driver.packInts(toPack, 0, packed, 0);
        
        float[] queryData = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            queryData[i] = rand.nextFloat() * 10;
        }
        
        // Score should be consistent across multiple calls
        float score1 = driver.asymmetricScorePackedInts(packed, 0, queryData, 0);
        float score2 = driver.asymmetricScorePackedInts(packed, 0, queryData, 0);
        float tolerance = Math.max(1e-6f, Math.abs(score1) * 1e-6f);
        assertEquals(score1, score2, tolerance, "Score should be consistent");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testAsymmetricScoreManualCalculation(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int maxValue = (1 << bitDepth) - 1;
        int[] toPack = new int[dimension];
        Random rand = new Random(42);
        for (int i = 0; i < dimension; i++) {
            toPack[i] = rand.nextInt(maxValue + 1);
        }
        driver.packInts(toPack, 0, packed, 0);
        
        float[] queryData = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            queryData[i] = rand.nextFloat() * 10;
        }
        
        // Calculate expected score manually
        float expectedScore = 0f;
        for (int i = 0; i < dimension; i++) {
            expectedScore += toPack[i] * queryData[i];
        }

        float actualScore = driver.asymmetricScorePackedInts(packed, 0, queryData, 0);
        float tolerance = Math.max(1e-6f, Math.abs(expectedScore) * 1e-5f);
        assertEquals(expectedScore, actualScore, tolerance, "Score should match manual calculation");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testSymmetricScoreBasic(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(2);
        
        int maxValue = (1 << bitDepth) - 1;
        Random rand = new Random(42);
        
        int[] toPack1 = new int[dimension];
        int[] toPack2 = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            toPack1[i] = rand.nextInt(maxValue + 1);
            toPack2[i] = rand.nextInt(maxValue + 1);
        }
        
        driver.packInts(toPack1, 0, packed, 0);
        driver.packInts(toPack2, 0, packed, 1);
        
        float score = driver.symmetricScorePackedInts(packed, 0, packed, 1);
        assertTrue(Float.isFinite(score), "Score should be finite");
        assertTrue(score >= 0, "Score should be non-negative for non-negative inputs");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testSymmetricScoreSymmetry(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(2);
        
        int maxValue = (1 << bitDepth) - 1;
        Random rand = new Random(42);
        
        int[] toPack1 = new int[dimension];
        int[] toPack2 = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            toPack1[i] = rand.nextInt(maxValue + 1);
            toPack2[i] = rand.nextInt(maxValue + 1);
        }
        
        driver.packInts(toPack1, 0, packed, 0);
        driver.packInts(toPack2, 0, packed, 1);
        
        float score1 = driver.symmetricScorePackedInts(packed, 0, packed, 1);
        float score2 = driver.symmetricScorePackedInts(packed, 1, packed, 0);
        float tolerance = Math.max(1e-6f, Math.abs(score1) * 1e-6f);
        assertEquals(score1, score2, tolerance, "Symmetric score should be commutative");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testSymmetricScoreSelfSimilarity(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(1);
        
        int maxValue = (1 << bitDepth) - 1;
        Random rand = new Random(42);
        
        int[] toPack = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            toPack[i] = rand.nextInt(maxValue + 1);
        }
        
        driver.packInts(toPack, 0, packed, 0);
        
        float score = driver.symmetricScorePackedInts(packed, 0, packed, 0);
        assertTrue(Float.isFinite(score), "Self-similarity score should be finite");
        assertTrue(score >= 0, "Self-similarity score should be non-negative");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testSymmetricScoreZeroVectors(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(2);
        
        int[] toPack1 = new int[dimension]; // All zeros
        int[] toPack2 = new int[dimension]; // All zeros
        
        driver.packInts(toPack1, 0, packed, 0);
        driver.packInts(toPack2, 0, packed, 1);
        
        float score = driver.symmetricScorePackedInts(packed, 0, packed, 1);
        assertEquals(0.0f, score, 1e-6f, "Score should be zero for zero vectors");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testSymmetricScoreManualCalculation(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(2);
        
        int maxValue = (1 << bitDepth) - 1;
        Random rand = new Random(42);
        
        int[] toPack1 = new int[dimension];
        int[] toPack2 = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            toPack1[i] = rand.nextInt(maxValue + 1);
            toPack2[i] = rand.nextInt(maxValue + 1);
        }
        
        driver.packInts(toPack1, 0, packed, 0);
        driver.packInts(toPack2, 0, packed, 1);
        
        // Calculate expected score manually
        float expectedScore = 0f;
        for (int i = 0; i < dimension; i++) {
            expectedScore += toPack1[i] * toPack2[i];
        }
        
        float actualScore = driver.symmetricScorePackedInts(packed, 0, packed, 1);
        float tolerance = Math.max(1e-6f, Math.abs(expectedScore) * 1e-4f);
        assertEquals(expectedScore, actualScore, tolerance, "Score should match manual calculation");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testScoreCorrespondence(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(2);
        
        int maxValue = (1 << bitDepth) - 1;
        Random rand = new Random(42);
        
        int[] toPack1 = new int[dimension];
        int[] toPack2 = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            toPack1[i] = rand.nextInt(maxValue + 1);
            toPack2[i] = rand.nextInt(maxValue + 1);
        }
        
        driver.packInts(toPack1, 0, packed, 0);
        driver.packInts(toPack2, 0, packed, 1);
        
        // Create query vector with same values as toPack2
        float[] queryData = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            queryData[i] = (float) toPack2[i];
        }
        
        float asymmetricScore = driver.asymmetricScorePackedInts(packed, 0, queryData, 0);
        float symmetricScore = driver.symmetricScorePackedInts(packed, 0, packed, 1);
        
        // When query values match the packed values, scores should be equal
        float tolerance = Math.max(1e-6f, Math.abs(symmetricScore) * 1e-4f);
        assertEquals(symmetricScore, asymmetricScore, tolerance,
            "Asymmetric and symmetric scores should match when query equals packed vector");
    }

    @ParameterizedTest
    @MethodSource("combinations")
    public void testGetRawComponentSumMultipleVectors(VectorizationProvider vts, int bitDepth, int dimension) {
        var driver = vts.getAshDriverFactory().createDriver(bitDepth, dimension);
        var packed = driver.create(3);
        
        int maxValue = (1 << bitDepth) - 1;
        Random rand = new Random(42);
        
        long[] expectedSums = new long[3];
        for (int v = 0; v < 3; v++) {
            int[] toPack = new int[dimension];
            for (int i = 0; i < dimension; i++) {
                toPack[i] = rand.nextInt(maxValue + 1);
                expectedSums[v] += toPack[i];
            }
            driver.packInts(toPack, 0, packed, v);
        }
        
        // Verify each vector's sum independently
        for (int v = 0; v < 3; v++) {
            float actualSum = driver.getRawComponentSum(packed, v);
            assertEquals((float) expectedSums[v], actualSum, 1e-6f,
                "Sum at offset " + v + " should match expected value");
        }
    }
}

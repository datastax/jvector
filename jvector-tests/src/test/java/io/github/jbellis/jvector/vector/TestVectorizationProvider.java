/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.vector;

import com.carrotsearch.randomizedtesting.RandomizedTest;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.hamcrest.CoreMatchers.instanceOf;


public class TestVectorizationProvider extends RandomizedTest {
    static final boolean hasSimd = VectorizationProvider.vectorModulePresentAndReadable();
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    @Test
    public void testSimilarityMetricsFloat() {
        Assume.assumeTrue(hasSimd);

        VectorizationProvider a = new DefaultVectorizationProvider();
        VectorizationProvider b = VectorizationProvider.getInstance();

        VectorFloat<?> v1b = TestUtil.randomVector(getRandom(), 1021); //prime numbers
        VectorFloat<?> v2b = TestUtil.randomVector(getRandom(), 1021); //prime numbers

        var v1a = a.getVectorTypeSupport().createFloatVector(1021);
        var v2a = a.getVectorTypeSupport().createFloatVector(1021);
        for (int i = 0; i < 1021; i++) {
            v1a.set(i, v1b.get(i));
            v2a.set(i, v2b.get(i));
        }

        Assert.assertThat(v1a, instanceOf(FloatArray.class));
        Assert.assertThat(v2a, instanceOf(FloatArray.class));
        Assert.assertArrayEquals(((FloatArray) v1a).array(), getVector(v1a), 0.0001f);
        Assert.assertArrayEquals(((FloatArray) v2a).array(), getVector(v2a), 0.0001f);

        Assert.assertEquals(a.getVectorUtilSupport().dotProduct(v1a,v2a), b.getVectorUtilSupport().dotProduct(v1b, v2b), 0.0001f);
        Assert.assertEquals(a.getVectorUtilSupport().cosine(v1a,v2a), b.getVectorUtilSupport().cosine(v1b, v2b), 0.0001f);
        Assert.assertEquals(a.getVectorUtilSupport().squareDistance(v1a, v2a), b.getVectorUtilSupport().squareDistance(v1b, v2b), 0.0001f);
    }

    /**
     * PQ scoring calls dotProduct on short slices at non-zero offsets, so check every length around the SIMD widths,
     * including a slice exactly as wide as one preferred vector, at several offsets against the scalar provider.
     */
    @Test
    public void testDotProductOnShortSlicesWithOffsets() {
        Assume.assumeTrue(hasSimd);

        VectorizationProvider scalar = new DefaultVectorizationProvider();
        VectorizationProvider simd = VectorizationProvider.getInstance();

        for (int length : new int[] {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33}) {
            for (int offset1 : new int[] {0, 1, 4, 13}) {
                for (int offset2 : new int[] {0, 3, 8}) {
                    VectorFloat<?> simdA = TestUtil.randomVector(getRandom(), offset1 + length + 5);
                    VectorFloat<?> simdB = TestUtil.randomVector(getRandom(), offset2 + length + 5);
                    var scalarA = scalar.getVectorTypeSupport().createFloatVector(simdA.length());
                    var scalarB = scalar.getVectorTypeSupport().createFloatVector(simdB.length());
                    for (int i = 0; i < simdA.length(); i++) {
                        scalarA.set(i, simdA.get(i));
                    }
                    for (int i = 0; i < simdB.length(); i++) {
                        scalarB.set(i, simdB.get(i));
                    }
                    Assert.assertEquals("length=" + length + " offsets=" + offset1 + "," + offset2,
                                        scalar.getVectorUtilSupport().dotProduct(scalarA, offset1, scalarB, offset2, length),
                                        simd.getVectorUtilSupport().dotProduct(simdA, offset1, simdB, offset2, length),
                                        1e-4f);
                }
            }
        }
    }

    /**
     * dotProduct handles a slice exactly one preferred vector wide itself instead of delegating to dotProductPreferred.
     * Both must return the same bits for every input, including signed zeros and non-finite values, so that the
     * shortcut cannot change a score. dotProductPreferred is package-private, so it is reached by reflection, which
     * also keeps this test loadable when the Panama provider is not in use.
     */
    @Test
    public void testDotProductFastPathMatchesPreferredBitwise() throws Exception {
        Assume.assumeTrue(hasSimd);
        var impl = VectorizationProvider.getInstance().getVectorUtilSupport();
        java.lang.reflect.Method preferred;
        try {
            preferred = impl.getClass().getDeclaredMethod("dotProductPreferred", VectorFloat.class, int.class, VectorFloat.class, int.class, int.class);
        } catch (NoSuchMethodException e) {
            Assume.assumeNoException("not the Panama provider", e);
            return;
        }
        preferred.setAccessible(true);
        int width = Integer.parseInt(System.getProperty("io.github.jbellis.jvector.vector_bit_size")) / Float.SIZE;
        float[] specials = {0f, -0f, 1f, -1f, Float.MIN_VALUE, -Float.MIN_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE,
                            Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN, 1e-30f, -1e-30f, 1e30f};
        var random = getRandom();
        for (int length : new int[] {width, 2 * width, 3 * width + 1}) {
            for (int trial = 0; trial < 2000; trial++) {
                int offset1 = random.nextInt(9);
                int offset2 = random.nextInt(9);
                float[] x = new float[offset1 + length + 3];
                float[] y = new float[offset2 + length + 3];
                for (int i = 0; i < x.length; i++) {
                    x[i] = trial % 2 == 0 ? (float) random.nextGaussian() : specials[random.nextInt(specials.length)];
                }
                for (int i = 0; i < y.length; i++) {
                    y[i] = trial % 2 == 0 ? (float) random.nextGaussian() : specials[random.nextInt(specials.length)];
                }
                VectorFloat<?> a = vectorTypeSupport.createFloatVector(x);
                VectorFloat<?> b = vectorTypeSupport.createFloatVector(y);
                float direct = impl.dotProduct(a, offset1, b, offset2, length);
                float viaPreferred = (float) preferred.invoke(impl, a, offset1, b, offset2, length);
                Assert.assertEquals("length=" + length + " offsets=" + offset1 + "," + offset2,
                                    Float.floatToRawIntBits(viaPreferred), Float.floatToRawIntBits(direct));
            }
        }
    }

    @Test
    public void testAssembleAndSum() {
        Assume.assumeTrue(hasSimd);

        VectorizationProvider a = new DefaultVectorizationProvider();
        VectorizationProvider b = VectorizationProvider.getInstance();

        for (int i = 0; i < 1000; i++) {
            VectorFloat<?> v2 = TestUtil.randomVector(getRandom(), 256);

            VectorFloat<?> v3 = vectorTypeSupport.createFloatVector(32);
            byte[] offsets = new byte[32];
            int skipSize = 256/32;
            //Assemble v3 from bits of v2
            for (int j = 0, c = 0; j < 256; j+=skipSize, c++) {
                v3.set(c, v2.get(j));
                offsets[c] = (byte) (c * skipSize);
            }

            Assert.assertThat(v2, instanceOf(FloatArray.class));
            Assert.assertThat(v3, instanceOf(FloatArray.class));
            Assert.assertArrayEquals(((FloatArray) v2).array(), getVector(v2), 0.0001f);
            Assert.assertArrayEquals(((FloatArray) v3).array(), getVector(v3), 0.0001f);

            Assert.assertEquals(a.getVectorUtilSupport().sum(v3), b.getVectorUtilSupport().sum(v3), 0.0001);
            Assert.assertEquals(a.getVectorUtilSupport().sum(v3), a.getVectorUtilSupport().assembleAndSum(v2, 0, vectorTypeSupport.createByteSequence(offsets)), 0.0001);
            Assert.assertEquals(b.getVectorUtilSupport().sum(v3), b.getVectorUtilSupport().assembleAndSum(v2, 0, vectorTypeSupport.createByteSequence(offsets)), 0.0001);
        }
    }

	public static String REQUIRE_SPECIFIC_VECTORIZATION_PROVIDER="Test_RequireSpecificVectorizationProvider";

    /**
     * To run with native-access vector support, use
     * <pre>{@code
     *   -ea
     *   --add-modules jdk.incubator.vector
     *   --enable-native-access=ALL-UNNAMED
     * }</pre>
     *
     * To run with panama support, use
     * <pre>{@code
     *   -ea
     *   --add-modules jdk.incubator.vector
     *   -Djvector.vectorization_provider=panama
     * }</pre>
     *
     * If <pre>{@code -DTest_RequireSpecificVectorizationProvider=<simplename>}</pre>
     * is provided, then this test will error out if the chosen vector support type is different.
     */
    @Test
    public void testVectorSupportTypeIsExpected() {
        VectorizationProvider provider = VectorizationProvider.getInstance();
        System.out.println("PROVIDER: using " + provider.getClass().getSimpleName());

        boolean readable = VectorizationProvider.vectorModulePresentAndReadable();
        System.out.println("VECTOR MODULE READABLE: " + readable);

        String requiredProvider = System.getProperty(REQUIRE_SPECIFIC_VECTORIZATION_PROVIDER);
        if (requiredProvider != null) {
            System.out.println("REQUIRED PROVIDER: " + requiredProvider);
            assertEquals(
                    requiredProvider,
                    provider.getClass().getSimpleName(),
                    "Provider mismatch, " + "required " + requiredProvider + ", detected "
                            + provider.getClass().getSimpleName()
            );
        }
    }

    private static float[] getVector(VectorFloat<?> vf) {
        final float[] arr = new float[vf.length()];
        for (int i = 0; i < vf.length(); ++i) {
            arr[i] = vf.get(i);
        }
        return arr;
    }
}

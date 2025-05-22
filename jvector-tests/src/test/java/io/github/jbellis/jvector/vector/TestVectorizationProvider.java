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
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;


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

        Assert.assertEquals(a.getVectorUtilSupport().dotProduct(v1a,v2a), b.getVectorUtilSupport().dotProduct(v1b, v2b), 0.0001f);
        Assert.assertEquals(a.getVectorUtilSupport().cosine(v1a,v2a), b.getVectorUtilSupport().cosine(v1b, v2b), 0.0001f);
        Assert.assertEquals(a.getVectorUtilSupport().squareDistance(v1a, v2a), b.getVectorUtilSupport().squareDistance(v1b, v2b), 0.0001f);
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
     *   -Djvector.experimental.enable_native_vectorization=true
     * }</pre>
     *
     * To run with panama support, use
     * <pre>{@code
     *   --ea
     *   --add-modules jdk.incubator.vector
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

}

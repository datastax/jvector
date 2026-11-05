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

import java.util.Random;
import java.lang.reflect.Proxy;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TestPQCosine {
    @Test
    void preservesFrozenKernelBitsAndMatchesScalarOracle() {
        var current = new PanamaVectorUtilSupport();
        var reference = new FrozenReference();
        Random random = new Random(4331);
        for (int count : new int[] {0,1,7,8,9,15,16,17,31,32,33,96,99,103,192,193,257}) {
            for (int stride : new int[] {1,2,16,128,256}) {
                float[] sums = new float[count * stride];
                float[] mags = new float[count * stride];
                for (int i=0;i<sums.length;i++) {
                    sums[i]=random.nextFloat()*2-1;
                    mags[i]=0.25f+random.nextFloat()*4;
                }
                for (int offset : new int[] {0,1,7}) {
                    byte[] codes = new byte[offset+count];
                    for(int i=0;i<codes.length;i++) codes[i]=(byte)random.nextInt(stride);
                    check(current,reference,sums,mags,codes,offset,count,stride);
                }
            }
        }
        for (int first=0;first<256;first++) {
            int count=33,stride=256,offset=3;
            float[] sums=new float[count*stride],mags=new float[count*stride];
            byte[] codes=new byte[count+offset];
            for(int i=0;i<sums.length;i++){sums[i]=(i%stride)-128;mags[i]=1+(i%stride);}
            for(int i=0;i<count;i++)codes[i+offset]=(byte)(first+i);
            check(current,reference,sums,mags,codes,offset,count,stride);
        }
    }

    @Test
    void compilationPreservesUnsignedGatherIndices() {
        var support = new PanamaVectorUtilSupport();
        var random = new Random(755);
        int clusters = 256;
        for (int count : new int[] {16, 17, 32, 33, 96, 192, 384}) {
            int offset = 7;
            byte[] codes = new byte[count + offset];
            random.nextBytes(codes);
            float[] sums = new float[count * clusters];
            float[] magnitudes = new float[sums.length];
            for (int i = 0; i < sums.length; i++) {
                sums[i] = i % 13 - 6;
                magnitudes[i] = 1 + i % 7;
            }
            float sum = 0, magnitude = 0;
            for (int i = 0; i < count; i++) {
                int index = i * clusters + Byte.toUnsignedInt(codes[offset + i]);
                sum += sums[index];
                magnitude += magnitudes[index];
            }
            float expected = (float) (sum / Math.sqrt(magnitude * 3.25f));
            var encoded = new ArrayByteSequence(codes);
            var partials = new ArrayVectorFloat(sums);
            var norms = new ArrayVectorFloat(magnitudes);
            // Warm both gather kernels: JDK-8321215 appeared only after C2 compilation.
            // Integer lane sums are exact, so reduction ordering cannot obscure this check.
            for (int iteration = 0; iteration < 50_000; iteration++) {
                assertEquals(expected, support.pqDecodedCosineSimilarity256(
                        encoded, offset, count, clusters, partials, norms, 3.25f));
                assertEquals(expected, support.pqDecodedCosineSimilarity512(
                        encoded, offset, count, clusters, partials, norms, 3.25f));
            }
        }
    }

    private void check(PanamaVectorUtilSupport c, FrozenReference r, float[] sums,float[] mags,
                       byte[] codes,int offset,int count,int stride) {
        var s=new ArrayVectorFloat(sums);var m=new ArrayVectorFloat(mags);var b=new ArrayByteSequence(codes);
        double sum=0,mag=0;
        for(int i=0;i<count;i++){int j=i*stride+Byte.toUnsignedInt(codes[offset+i]);sum+=sums[j];mag+=mags[j];}
        float oracle=(float)(sum/Math.sqrt(mag*3.25f));
        double tolerance = Float.isFinite(oracle) ? Math.max(1e-6, Math.abs(oracle) * 1e-5) : 0;
        assertEquals(oracle, c.pqDecodedCosineSimilarity128(b, offset, count, stride, s, m, 3.25f), tolerance);
        assertEquals(oracle, c.pqDecodedCosineSimilarity(b, offset, count, stride, s, m, 3.25f), tolerance);
        var compact = new ArrayByteSequence(java.util.Arrays.copyOfRange(codes, offset, offset + count));
        assertEquals(oracle, c.pqDecodedCosineSimilarity(compact, stride, s, m, 3.25f), tolerance);
        float a=c.pqDecodedCosineSimilarity256(b,offset,count,stride,s,m,3.25f);
        float z=r.pqDecodedCosineSimilarity256(b,offset,count,stride,s,m,3.25f);
        assertEquals(z,a,tolerance,"256 count="+count+" offset="+offset);
        assertEquals(oracle,a,Float.isFinite(oracle) ? Math.max(1e-6,Math.abs(oracle)*1e-5) : 0);
        a=c.pqDecodedCosineSimilarity512(b,offset,count,stride,s,m,3.25f);
        z=r.pqDecodedCosineSimilarity512(b,offset,count,stride,s,m,3.25f);
        assertEquals(z,a,tolerance,"512 count="+count+" offset="+offset);
        assertEquals(oracle,a,Float.isFinite(oracle) ? Math.max(1e-6,Math.abs(oracle)*1e-5) : 0);
        byte[] padded = new byte[codes.length + 11];
        System.arraycopy(codes, 0, padded, 5, codes.length);
        var slice = new ArrayByteSequence(padded).slice(5, codes.length);
        assertEquals(oracle, c.pqDecodedCosineSimilarity256(slice, offset, count, stride, s, m, 3.25f), tolerance);
        assertEquals(oracle, c.pqDecodedCosineSimilarity512(slice, offset, count, stride, s, m, 3.25f), tolerance);
    }

    @Test
    void shortScalarInputsDoNotRequireBackingArrays() {
        var data = (VectorFloat<?>) Proxy.newProxyInstance(VectorFloat.class.getClassLoader(),
                new Class<?>[]{VectorFloat.class}, (proxy, method, args) -> {
                    if (method.getName().equals("get") && method.getParameterCount() == 1)
                        return ((Integer) args[0]).floatValue() + 1f;
                    throw new AssertionError("Unexpected vector access: " + method);
                });
        var codes = (ByteSequence<?>) Proxy.newProxyInstance(ByteSequence.class.getClassLoader(),
                new Class<?>[]{ByteSequence.class}, (proxy, method, args) -> {
                    if (method.getName().equals("get") && method.getParameterCount() == 1)
                        return (byte) 255;
                    throw new AssertionError("Unexpected code access: " + method);
                });
        var support = new PanamaVectorUtilSupport();
        for (int count = 0; count < 16; count++) {
            float sum = 0;
            for (int i = 0; i < count; i++) sum += 256 * i + 256;
            float expected = (float) (sum / Math.sqrt(sum * 3.25f));
            assertEquals(expected, support.pqDecodedCosineSimilarity512(codes, 3, count, 256, data, data, 3.25f));
            if (count < 8)
                assertEquals(expected, support.pqDecodedCosineSimilarity256(codes, 3, count, 256, data, data, 3.25f));
        }
    }

    // Frozen kernels are a numerical regression oracle; reduceLanes(ADD) may reorder additions.
    private static class FrozenReference extends PanamaVectorUtilSupport {
    float pqDecodedCosineSimilarity512(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        var sum = FloatVector.zero(FloatVector.SPECIES_512);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_512);

        int[] convOffsets = scratchInt512.get();
        int i = 0;
        int limit = i + ByteVector.SPECIES_128.loopBound(baseOffsetsLength);

        var scale = IntVector.zero(IntVector.SPECIES_512).addIndex(clusterCount);

        for (; i < limit; i += ByteVector.SPECIES_128.length()) {

            fromByteSequence(ByteVector.SPECIES_128, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets,0);

            var offset = i * clusterCount;
            sum = sum.add(fromVectorFloat(FloatVector.SPECIES_512, partialSums, offset, convOffsets, 0));
            vaMagnitude = vaMagnitude.add(fromVectorFloat(FloatVector.SPECIES_512, aMagnitude, offset, convOffsets, 0));
        }

        float sumResult = sum.reduceLanes(VectorOperators.ADD);
        float aMagnitudeResult = vaMagnitude.reduceLanes(VectorOperators.ADD);

        for (; i < baseOffsetsLength; i++) {
            int offset = clusterCount * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset));
            sumResult += partialSums.get(offset);
            aMagnitudeResult += aMagnitude.get(offset);
        }

        return (float) (sumResult / Math.sqrt(aMagnitudeResult * bMagnitude));
    }

    float pqDecodedCosineSimilarity256(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        var sum = FloatVector.zero(FloatVector.SPECIES_256);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_256);

        int[] convOffsets = scratchInt256.get();
        int i = 0;
        int limit = ByteVector.SPECIES_64.loopBound(baseOffsetsLength);

        var scale = IntVector.zero(IntVector.SPECIES_256).addIndex(clusterCount);

        for (; i < limit; i += ByteVector.SPECIES_64.length()) {

            fromByteSequence(ByteVector.SPECIES_64, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets,0);

            var offset = i * clusterCount;
            sum = sum.add(fromVectorFloat(FloatVector.SPECIES_256, partialSums, offset, convOffsets, 0));
            vaMagnitude = vaMagnitude.add(fromVectorFloat(FloatVector.SPECIES_256, aMagnitude, offset, convOffsets, 0));
        }

        float sumResult = sum.reduceLanes(VectorOperators.ADD);
        float aMagnitudeResult = vaMagnitude.reduceLanes(VectorOperators.ADD);

        for (; i < baseOffsetsLength; i++) {
            int offset = clusterCount * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset));
            sumResult += partialSums.get(offset);
            aMagnitudeResult += aMagnitude.get(offset);
        }

        return (float) (sumResult / Math.sqrt(aMagnitudeResult * bMagnitude));
    }

    }
}

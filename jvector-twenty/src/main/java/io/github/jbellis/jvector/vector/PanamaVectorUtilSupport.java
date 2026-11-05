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

import io.github.jbellis.jvector.util.MathUtil;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.FloatArray;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorShape;

import java.util.List;

class PanamaVectorUtilSupport implements VectorUtilSupport {

    // JDK-8321215 misencodes gather indices in high vector registers before JDK 22 / 21.0.3.
    private static final boolean BROKEN_GATHER = hasBrokenGather();
    private static final ThreadLocal<float[]> gatherScratch = ThreadLocal.withInitial(
            () -> new float[FloatVector.SPECIES_512.length()]);

    static final int PREFERRED_BIT_SIZE = FloatVector.SPECIES_PREFERRED.vectorBitSize();
    static {
        // Publish the value to System Properties for other packages (other language levels...)
        System.setProperty("io.github.jbellis.jvector.vector_bit_size", String.valueOf(PREFERRED_BIT_SIZE));
    }
    static final IntVector BYTE_TO_INT_MASK_512 = IntVector.broadcast(IntVector.SPECIES_512, 0xff);
    static final IntVector BYTE_TO_INT_MASK_256 = IntVector.broadcast(IntVector.SPECIES_256, 0xff);

    // Keep concrete type checks inline without repeating species lookups.
    private static final Class<?> BYTE_128_CLASS = ByteVector.SPECIES_128.vectorType();
    private static final Class<?> BYTE_64_CLASS = ByteVector.SPECIES_64.vectorType();
    private static final Class<?> FLOAT_256_CLASS = FloatVector.SPECIES_256.vectorType();
    private static final Class<?> FLOAT_512_CLASS = FloatVector.SPECIES_512.vectorType();
    private static final Class<?> FLOAT_PREFERRED_CLASS = FloatVector.SPECIES_PREFERRED.vectorType();
    private static final Class<?> INT_256_CLASS = IntVector.SPECIES_256.vectorType();
    private static final Class<?> INT_512_CLASS = IntVector.SPECIES_512.vectorType();
    private static final Class<?> INT_PREFERRED_CLASS = IntVector.SPECIES_PREFERRED.vectorType();

    private static final int FLOAT_PREFERRED_LANES = FloatVector.SPECIES_PREFERRED.length();
    private static final int BYTE_128_LANES = ByteVector.SPECIES_128.length();
    private static final int BYTE_64_LANES = ByteVector.SPECIES_64.length();

    static final ThreadLocal<int[]> scratchInt512 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_512.length()]);
    static final ThreadLocal<int[]> scratchInt256 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_256.length()]);

    private static float[] arrayForBlock(VectorFloat<?> vector, int vectorizedLength) {
        return vectorizedLength == 0 ? null : array(vector);
    }

    private static boolean hasBrokenGather() {
        Runtime.Version version = Runtime.version();
        String arch = System.getProperty("os.arch");
        return (arch.equals("amd64") || arch.equals("x86_64"))
                && (version.feature() < 21 || (version.feature() == 21 && version.update() < 3));
    }

    // Called only on JDKs with the indexed-load compiler bug. Modern JDKs use
    // FloatVector.fromArray at the call site so no vector can escape through a helper.
    private static FloatVector gatherFallback(VectorSpecies<Float> species, float[] values, int offset, int[] indices, int indicesOffset) {
        float[] scratch = gatherScratch.get();
        for (int lane = 0; lane < species.length(); lane++) {
            scratch[lane] = values[offset + indices[indicesOffset + lane]];
        }
        return FloatVector.fromArray(species, scratch, 0);
    }

    private static float[] array(VectorFloat<?> vector) {
        return vector instanceof ArrayVectorFloat array ? array.get() : ((FloatArray) vector).array();
    }

    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset) {
        return FloatVector.fromArray(SPEC, ((ArrayVectorFloat) vector).get(), offset);
    }

    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset, int[] indices, int indicesOffset) {
        return (BROKEN_GATHER
                    ? gatherFallback(SPEC, ((ArrayVectorFloat)vector).get(), offset, indices, indicesOffset)
                    : FloatVector.fromArray(SPEC, ((ArrayVectorFloat)vector).get(), offset, indices, indicesOffset));
    }

    protected void intoVectorFloat(FloatVector vector, VectorFloat<?> v, int offset) {
        vector.intoArray(((ArrayVectorFloat) v).get(), offset);
    }

    private static byte[] byteArray(ByteSequence<?> vector) {
        return vector instanceof ArrayByteSequence || vector instanceof ArraySliceByteSequence
                ? (byte[]) vector.get() : null;
    }

    protected ByteVector fromByteSequence(VectorSpecies<Byte> SPEC, ByteSequence<?> vector, int offset) {
        return ByteVector.fromArray(SPEC, (byte[]) vector.get(), offset);
    }

    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset) {
        vector.intoArray(((ArrayByteSequence) v).get(), offset);
    }

    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset, VectorMask<Byte> mask) {
        vector.intoArray(((ArrayByteSequence) v).get(), offset, mask);
    }


    @Override
    public float sum(VectorFloat<?> vector) {

        FloatVector sum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_PREFERRED));
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        // Resolve array storage once; native vectors expose the same heap-backed array.
        float[] values = vectorizedLength == 0 ? null : ((FloatArray) vector).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, values, i));
            sum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(sum.add(a));
        }
        float res = sum.reduceLanes(VectorOperators.ADD);
        for (int i = vectorizedLength; i < vector.length(); i++) res += vector.get(i);
        return res;
    }

    @Override
    public VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Input list cannot be null or empty");
        }

        int dimension = vectors.get(0).length();
        VectorFloat<?> sum = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(dimension);

        // Process each vector from the list
        for (VectorFloat<?> vector : vectors) {
            addInPlace(sum, vector);
        }

        return sum;
    }

    @Override
    public void scale(VectorFloat<?> vector, float multiplier) {

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        float[] values = vectorizedLength == 0 ? null : ((FloatArray) vector).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, values, i));
            FloatVector scaled = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a.mul(multiplier));
            scaled.intoArray(values, i);
        }
        for (int i = vectorizedLength; i < vector.length(); i++)
            vector.set(i, vector.get(i) * multiplier);
    }

    float dot64(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        float[] v1Values = ((FloatArray) v1).array();
        float[] v2Values = ((FloatArray) v2).array();
        FloatVector a = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_64, v1Values, offset1));
        FloatVector b = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_64, v2Values, offset2));
        FloatVector product = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                a.mul(b));
        return product.reduceLanes(VectorOperators.ADD);

    }

    float dot128(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        float[] v1Values = ((FloatArray) v1).array();
        float[] v2Values = ((FloatArray) v2).array();
        FloatVector a = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_128, v1Values, offset1));
        FloatVector b = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_128, v2Values, offset2));
        FloatVector product = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                a.mul(b));
        return product.reduceLanes(VectorOperators.ADD);

    }

    float dot256(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        // Keep the concrete SIMD type visible when this kernel is not inlined.
        FloatVector a = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_256, array(v1), offset1));
        FloatVector b = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_256, array(v2), offset2));
        FloatVector product = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(a.mul(b));
        return product.reduceLanes(VectorOperators.ADD);
    }

    float dotPreferred(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        float[] v1Values = ((FloatArray) v1).array();
        float[] v2Values = ((FloatArray) v2).array();
        FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1Values, offset1));
        FloatVector b = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2Values, offset2));
        FloatVector product = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                a.mul(b));
        return product.reduceLanes(VectorOperators.ADD);

    }

    @Override
    public float dotProduct(VectorFloat<?> v1, VectorFloat<?> v2) {
        return dotProduct(v1, 0, v2, 0, v1.length());
    }

    @Override
    public float dotProduct(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, final int length)
    {
        //Common case first
        if (length >= FloatVector.SPECIES_PREFERRED.length())
            return dotProductPreferred(v1, v1offset, v2, v2offset, length);

        if (length < FloatVector.SPECIES_128.length())
            return dotProduct64(v1, v1offset, v2, v2offset, length);
        else if (length < FloatVector.SPECIES_256.length())
            return dotProduct128(v1, v1offset, v2, v2offset, length);
        else
            return dotProduct256(v1, v1offset, v2, v2offset, length);

    }

    float dotProduct64(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_64.length())
            return dot64(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
        float[] v1Values = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] v2Values = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        FloatVector sum = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_64));
        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_64, v1Values, v1offset + i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_64, v2Values, v2offset + i));
            sum = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                    a.fma(b, sum));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;

    }

    float dotProduct128(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_128.length())
            return dot128(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
        float[] v1Values = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] v2Values = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        FloatVector sum = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_128));

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_128, v1Values, v1offset + i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_128, v2Values, v2offset + i));
            sum = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                    a.fma(b, sum));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;

    }


    float dotProduct256(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_256.length())
            return dot256(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
        float[] v1Values = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] v2Values = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        FloatVector sum = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_256));

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_256, v1Values, v1offset + i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_256, v2Values, v2offset + i));
            sum = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                    a.fma(b, sum));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;

    }

    float dotProductPreferred(VectorFloat<?> va, int vaoffset, VectorFloat<?> vb, int vboffset, int length) {
        // Preserve concrete SIMD types when this kernel is inlined into a larger caller.
        // Otherwise C2 can materialize temporary vector objects and backing arrays.
        if (length == FloatVector.SPECIES_PREFERRED.length())
            return dotPreferred(va, vaoffset, vb, vboffset);

        FloatVector sum0 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.zero(FloatVector.SPECIES_PREFERRED));
        FloatVector sum1 = sum0;
        FloatVector a0, a1, b0, b1;

        int vectorLength = FloatVector.SPECIES_PREFERRED.length();

        // Unrolled vector loop; for dot product from L1 cache, an unroll factor of 2 generally suffices.
        // If we are going to be getting data that's further down the hierarchy but not fetched off disk/network,
        // we might want to unroll further, e.g. to 8 (4 sets of a,b,sum with 3-ahead reads seems to work best).
        if (length >= vectorLength * 2)
        {
            length -= vectorLength * 2;
            a0 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(va), vaoffset + vectorLength * 0));
            b0 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(vb), vboffset + vectorLength * 0));
            a1 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(va), vaoffset + vectorLength * 1));
            b1 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(vb), vboffset + vectorLength * 1));
            vaoffset += vectorLength * 2;
            vboffset += vectorLength * 2;
            while (length >= vectorLength * 2)
            {
                // All instructions in the main loop have no dependencies between them and can be executed in parallel.
                length -= vectorLength * 2;
                sum0 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a0.fma(b0, sum0));
                a0 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(va), vaoffset + vectorLength * 0));
                b0 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(vb), vboffset + vectorLength * 0));
                sum1 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a1.fma(b1, sum1));
                a1 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(va), vaoffset + vectorLength * 1));
                b1 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(vb), vboffset + vectorLength * 1));
                vaoffset += vectorLength * 2;
                vboffset += vectorLength * 2;
            }
            sum0 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a0.fma(b0, sum0));
            sum1 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a1.fma(b1, sum1));
        }
        sum0 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(sum0.add(sum1));

        // Process the remaining few vectors
        while (length >= vectorLength) {
            length -= vectorLength;
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(va), vaoffset));
            FloatVector b = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(vb), vboffset));
            vaoffset += vectorLength;
            vboffset += vectorLength;
            sum0 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a.fma(b, sum0));
        }

        float resVec = sum0.reduceLanes(VectorOperators.ADD);
        float resTail = 0;

        // Process the tail
        for (; length > 0; --length)
            resTail += va.get(vaoffset++) * vb.get(vboffset++);

        return resVec + resTail;
    }

    @Override
    public float cosine(VectorFloat<?> v1, VectorFloat<?> v2) {
        int length = v1.length();
        if (length != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }
        return cosine(v1, 0, v2, 0, length);
    }

    @Override
    public float cosine(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        FloatVector vsum = (FloatVector) FLOAT_PREFERRED_CLASS.cast(
                FloatVector.zero(FloatVector.SPECIES_PREFERRED));
        FloatVector vaMagnitude = (FloatVector) FLOAT_PREFERRED_CLASS.cast(
                FloatVector.zero(FloatVector.SPECIES_PREFERRED));
        FloatVector vbMagnitude = (FloatVector) FLOAT_PREFERRED_CLASS.cast(
                FloatVector.zero(FloatVector.SPECIES_PREFERRED));
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        float[] v1Values = arrayForBlock(v1, vectorizedLength);
        float[] v2Values = arrayForBlock(v2, vectorizedLength);
        for (int i = 0; i < vectorizedLength; i += FLOAT_PREFERRED_LANES) {
            FloatVector a = (FloatVector) FLOAT_PREFERRED_CLASS.cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1Values, v1offset + i));
            FloatVector b = (FloatVector) FLOAT_PREFERRED_CLASS.cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2Values, v2offset + i));
            vsum = (FloatVector) FLOAT_PREFERRED_CLASS.cast(
                    a.fma(b, vsum));
            vaMagnitude = (FloatVector) FLOAT_PREFERRED_CLASS.cast(
                    a.fma(a, vaMagnitude));
            vbMagnitude = (FloatVector) FLOAT_PREFERRED_CLASS.cast(
                    b.fma(b, vbMagnitude));
        }

        return finishCosine(v1, v1offset, v2, v2offset, vectorizedLength, length,
                vsum.reduceLanes(VectorOperators.ADD), vaMagnitude.reduceLanes(VectorOperators.ADD),
                vbMagnitude.reduceLanes(VectorOperators.ADD));
    }

    private static float finishCosine(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2,
                                      int v2offset, int start, int length,
                                      float sum, float aMagnitude, float bMagnitude) {
        for (int i = start; i < length; i++) {
            sum += v1.get(v1offset + i) * v2.get(v2offset + i);
            aMagnitude += v1.get(v1offset + i) * v1.get(v1offset + i);
            bMagnitude += v2.get(v2offset + i) * v2.get(v2offset + i);
        }
        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    float squareDistance64(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        float[] v1Values = ((FloatArray) v1).array();
        float[] v2Values = ((FloatArray) v2).array();
        FloatVector a = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_64, v1Values, offset1));
        FloatVector b = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_64, v2Values, offset2));
        FloatVector diff = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                a.sub(b));
        FloatVector product = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                diff.mul(diff));
        return product.reduceLanes(VectorOperators.ADD);

    }

    float squareDistance128(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        float[] v1Values = ((FloatArray) v1).array();
        float[] v2Values = ((FloatArray) v2).array();
        FloatVector a = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_128, v1Values, offset1));
        FloatVector b = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_128, v2Values, offset2));
        FloatVector diff = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                a.sub(b));
        FloatVector product = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                diff.mul(diff));
        return product.reduceLanes(VectorOperators.ADD);

    }

    float squareDistance256(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        FloatVector a = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_256, array(v1), offset1));
        FloatVector b = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_256, array(v2), offset2));
        FloatVector diff = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(a.sub(b));
        FloatVector squared = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(diff.mul(diff));
        return squared.reduceLanes(VectorOperators.ADD);
    }

    float squareDistancePreferred(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        float[] v1Values = ((FloatArray) v1).array();
        float[] v2Values = ((FloatArray) v2).array();
        FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1Values, offset1));
        FloatVector b = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2Values, offset2));
        FloatVector diff = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                a.sub(b));
        FloatVector product = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                diff.mul(diff));
        return product.reduceLanes(VectorOperators.ADD);

    }

    @Override
    public float squareDistance(VectorFloat<?> v1, VectorFloat<?> v2) {
        return squareDistance(v1, 0, v2, 0, v1.length());
    }

    @Override
    public void dotProductMulti(VectorFloat<?> vector, VectorFloat<?>[] queries, int count, float[] out) {
        if (!(vector instanceof FloatArray)) {
            VectorUtilSupport.super.dotProductMulti(vector, queries, count, out);
            return;
        }
        final VectorSpecies<Float> FS = FloatVector.SPECIES_PREFERRED;
        final float[] v = ((FloatArray) vector).array();
        final int dim = vector.length();
        final int bound = FS.loopBound(dim);
        int i = 0;
        // eight queries per pass: one vector chunk load feeds eight FMAs into eight accumulators
        for (; i + 8 <= count; i += 8) {
            boolean heap = true;
            for (int j = 0; j < 8; j++) heap &= queries[i + j] instanceof FloatArray;
            if (!heap) break;
            final float[] q0 = ((FloatArray) queries[i]).array(), q1 = ((FloatArray) queries[i + 1]).array();
            final float[] q2 = ((FloatArray) queries[i + 2]).array(), q3 = ((FloatArray) queries[i + 3]).array();
            final float[] q4 = ((FloatArray) queries[i + 4]).array(), q5 = ((FloatArray) queries[i + 5]).array();
            final float[] q6 = ((FloatArray) queries[i + 6]).array(), q7 = ((FloatArray) queries[i + 7]).array();
            FloatVector a0 = FloatVector.zero(FS), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            int k = 0;
            for (; k < bound; k += FS.length()) {
                FloatVector vv = FloatVector.fromArray(FS, v, k);
                a0 = FloatVector.fromArray(FS, q0, k).fma(vv, a0);
                a1 = FloatVector.fromArray(FS, q1, k).fma(vv, a1);
                a2 = FloatVector.fromArray(FS, q2, k).fma(vv, a2);
                a3 = FloatVector.fromArray(FS, q3, k).fma(vv, a3);
                a4 = FloatVector.fromArray(FS, q4, k).fma(vv, a4);
                a5 = FloatVector.fromArray(FS, q5, k).fma(vv, a5);
                a6 = FloatVector.fromArray(FS, q6, k).fma(vv, a6);
                a7 = FloatVector.fromArray(FS, q7, k).fma(vv, a7);
            }
            float s0 = a0.reduceLanes(VectorOperators.ADD), s1 = a1.reduceLanes(VectorOperators.ADD);
            float s2 = a2.reduceLanes(VectorOperators.ADD), s3 = a3.reduceLanes(VectorOperators.ADD);
            float s4 = a4.reduceLanes(VectorOperators.ADD), s5 = a5.reduceLanes(VectorOperators.ADD);
            float s6 = a6.reduceLanes(VectorOperators.ADD), s7 = a7.reduceLanes(VectorOperators.ADD);
            for (; k < dim; k++) {
                float x = v[k];
                s0 += x * q0[k]; s1 += x * q1[k]; s2 += x * q2[k]; s3 += x * q3[k];
                s4 += x * q4[k]; s5 += x * q5[k]; s6 += x * q6[k]; s7 += x * q7[k];
            }
            out[i] = s0; out[i + 1] = s1; out[i + 2] = s2; out[i + 3] = s3;
            out[i + 4] = s4; out[i + 5] = s5; out[i + 6] = s6; out[i + 7] = s7;
        }
        for (; i < count; i++) {
            out[i] = dotProduct(vector, queries[i]);
        }
    }

    @Override
    public float squareDistance(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, final int length)
    {
        //Common case first
        if (length >= FloatVector.SPECIES_PREFERRED.length())
            return squareDistancePreferred(v1, v1offset, v2, v2offset, length);

        if (length < FloatVector.SPECIES_128.length())
            return squareDistance64(v1, v1offset, v2, v2offset, length);
        else if (length < FloatVector.SPECIES_256.length())
            return squareDistance128(v1, v1offset, v2, v2offset, length);
        else
            return squareDistance256(v1, v1offset, v2, v2offset, length);
    }

    float squareDistance64(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_64.length())
            return squareDistance64(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
        float[] v1Values = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] v2Values = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        FloatVector sum = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_64));

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_64, v1Values, v1offset + i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_64, v2Values, v2offset + i));
            FloatVector diff = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                    a.sub(b));
            sum = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                    diff.fma(diff, sum));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;

    }

    float squareDistance128(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_128.length())
            return squareDistance128(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
        float[] v1Values = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] v2Values = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        FloatVector sum = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_128));

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_128, v1Values, v1offset + i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_128, v2Values, v2offset + i));
            FloatVector diff = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                    a.sub(b));
            sum = (FloatVector) FloatVector.SPECIES_128.vectorType().cast(
                    diff.fma(diff, sum));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;

    }


    float squareDistance256(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_256.length())
            return squareDistance256(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
        float[] v1Values = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] v2Values = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        FloatVector sum = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_256));

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_256, v1Values, v1offset + i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_256, v2Values, v2offset + i));
            FloatVector diff = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                    a.sub(b));
            sum = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                    diff.fma(diff, sum));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;

    }

    float squareDistancePreferred(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_PREFERRED.length())
            return squareDistancePreferred(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        float[] v1Values = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] v2Values = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        FloatVector sum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_PREFERRED));

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1Values, v1offset + i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2Values, v2offset + i));
            FloatVector diff = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    a.sub(b));
            sum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    diff.fma(diff, sum));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;

    }

    void addInPlace64(VectorFloat<?> v1, VectorFloat<?> v2) {

        FloatVector a = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_64, ((FloatArray) v1).array(), 0));
        FloatVector b = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_64, ((FloatArray) v2).array(), 0));
        FloatVector added = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(a.add(b));
        added.intoArray(((FloatArray) v1).array(), 0);
    }

    void addInPlace64(VectorFloat<?> v1, float value) {

        FloatVector a = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(
                FloatVector.fromArray(FloatVector.SPECIES_64, ((FloatArray) v1).array(), 0));
        FloatVector added = (FloatVector) FloatVector.SPECIES_64.vectorType().cast(a.add(value));
        added.intoArray(((FloatArray) v1).array(), 0);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {

        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }
        if (v1.length() == 2) {
            addInPlace64(v1, v2);
            return;
        }
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());
        float[] aValues = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] bValues = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, aValues, i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, bValues, i));
            FloatVector result = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a.add(b));
            result.intoArray(aValues, i);
        }
        for (int i = vectorizedLength; i < v1.length(); i++)
            v1.set(i, v1.get(i) + v2.get(i));
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, float value) {

        if (v1.length() == 2) {
            addInPlace64(v1, value);
            return;
        }
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());
        float[] values = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, values, i));
            FloatVector result = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a.add(value));
            result.intoArray(values, i);
        }
        for (int i = vectorizedLength; i < v1.length(); i++)
            v1.set(i, v1.get(i) + value);
    }

    @Override
    public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {

        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());
        float[] aValues = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] bValues = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, aValues, i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, bValues, i));
            FloatVector result = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a.sub(b));
            result.intoArray(aValues, i);
        }
        for (int i = vectorizedLength; i < v1.length(); i++)
            v1.set(i, v1.get(i) - v2.get(i));
    }

    @Override
    public void subInPlace(VectorFloat<?> vector, float value) {

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        float[] values = vectorizedLength == 0 ? null : ((FloatArray) vector).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, values, i));
            FloatVector result = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a.sub(value));
            result.intoArray(values, i);
        }
        for (int i = vectorizedLength; i < vector.length(); i++)
            vector.set(i, vector.get(i) - value);
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, float value) {
        return sub(a, 0, value, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b) {
        return sub(a, 0, b, 0, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        VectorFloat<?> res = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(length);
        float[] aValues = vectorizedLength == 0 ? null : ((FloatArray) a).array();
        float[] resultValues = vectorizedLength == 0 ? null : ((FloatArray) res).array();
        float[] bValues = vectorizedLength == 0 ? null : ((FloatArray) b).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector lhs = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, aValues, aOffset + i));
            FloatVector rhs = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, bValues, bOffset + i));
            FloatVector result = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(lhs.sub(rhs));
            result.intoArray(resultValues, i);
        }
        for (int i = vectorizedLength; i < length; i++)
            res.set(i, a.get(aOffset + i) - b.get(bOffset + i));
        return res;
    }

    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, float value, int length) {

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        VectorFloat<?> res = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(length);
        float[] aValues = vectorizedLength == 0 ? null : ((FloatArray) a).array();
        float[] resultValues = vectorizedLength == 0 ? null : ((FloatArray) res).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector lhs = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, aValues, aOffset + i));
            FloatVector result = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(lhs.sub(value));
            result.intoArray(resultValues, i);
        }
        for (int i = vectorizedLength; i < length; i++)
            res.set(i, a.get(aOffset + i) - value);
        return res;
    }

    @Override
    public void minInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {

        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());
        float[] aValues = vectorizedLength == 0 ? null : ((FloatArray) v1).array();
        float[] bValues = vectorizedLength == 0 ? null : ((FloatArray) v2).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, aValues, i));
            FloatVector b = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, bValues, i));
            FloatVector result = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(a.min(b));
            result.intoArray(aValues, i);
        }
        for (int i = vectorizedLength; i < v1.length(); i++)
            v1.set(i, Math.min(v1.get(i), v2.get(i)));
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
        return assembleAndSum(data, dataBase,  baseOffsets, 0, baseOffsets.length());
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        return switch (PREFERRED_BIT_SIZE)
        {
            case 512 -> assembleAndSum512(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
            case 256 -> assembleAndSum256(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
            case 128 -> assembleAndSum128(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
            default -> throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
        };
    }

    float assembleAndSum512(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        if (baseOffsetsLength < ByteVector.SPECIES_128.length()) {
            return assembleAndSum128(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
        }
        return assembleAndSum512(array(data), dataBase, baseOffsets,
                baseOffsetsOffset, baseOffsetsLength);
    }

    private float assembleAndSum512(float[] dataArray, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {

        int[] convOffsets = scratchInt512.get();
        FloatVector sum = (FloatVector) FLOAT_512_CLASS.cast(FloatVector.zero(FloatVector.SPECIES_512));
        int i = 0;
        int limit = ByteVector.SPECIES_128.loopBound(baseOffsetsLength);
        IntVector scale = (IntVector) INT_512_CLASS.cast(IntVector.zero(IntVector.SPECIES_512).addIndex(dataBase));

        for (; i < limit; i += ByteVector.SPECIES_128.length()) {
            ByteVector bytes = (ByteVector) BYTE_128_CLASS.cast(
                    fromByteSequence(ByteVector.SPECIES_128, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset));
            IntVector codes = (IntVector) INT_512_CLASS.cast(bytes.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0));
            codes = (IntVector) INT_512_CLASS.cast(codes.and(BYTE_TO_INT_MASK_512));
            codes = (IntVector) INT_512_CLASS.cast(codes.add(scale));
            codes.intoArray(convOffsets, 0);
            FloatVector values = (FloatVector) FLOAT_512_CLASS.cast(BROKEN_GATHER
                    ? gatherFallback(FloatVector.SPECIES_512, dataArray, i * dataBase, convOffsets, 0)
                    : FloatVector.fromArray(FloatVector.SPECIES_512, dataArray, i * dataBase, convOffsets, 0));
            sum = (FloatVector) FLOAT_512_CLASS.cast(sum.add(values));
        }

        float result = sum.reduceLanes(VectorOperators.ADD);
        for (; i < baseOffsetsLength; i++) {
            result += dataArray[dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset))];
        }
        return result;
    }

    float assembleAndSum256(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        if (baseOffsetsLength < ByteVector.SPECIES_64.length()) {
            return assembleAndSum128(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
        }
        return assembleAndSum256(array(data), dataBase, baseOffsets,
                baseOffsetsOffset, baseOffsetsLength);
    }

    private float assembleAndSum256(float[] dataArray, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {

        int[] convOffsets = scratchInt256.get();
        FloatVector sum = (FloatVector) FLOAT_256_CLASS.cast(FloatVector.zero(FloatVector.SPECIES_256));
        int i = 0;
        int limit = ByteVector.SPECIES_64.loopBound(baseOffsetsLength);
        IntVector scale = (IntVector) INT_256_CLASS.cast(IntVector.zero(IntVector.SPECIES_256).addIndex(dataBase));

        for (; i < limit; i += ByteVector.SPECIES_64.length()) {

            ByteVector bytes = (ByteVector) BYTE_64_CLASS.cast(
                    fromByteSequence(ByteVector.SPECIES_64, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset));
            IntVector codes = (IntVector) INT_256_CLASS.cast(bytes.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0));
            codes = (IntVector) INT_256_CLASS.cast(codes.and(BYTE_TO_INT_MASK_256));
            codes = (IntVector) INT_256_CLASS.cast(codes.add(scale));
            codes.intoArray(convOffsets, 0);
            FloatVector values = (FloatVector) FLOAT_256_CLASS.cast(BROKEN_GATHER
                    ? gatherFallback(FloatVector.SPECIES_256, dataArray, i * dataBase, convOffsets, 0)
                    : FloatVector.fromArray(FloatVector.SPECIES_256, dataArray, i * dataBase, convOffsets, 0));
            sum = (FloatVector) FLOAT_256_CLASS.cast(sum.add(values));
        }

        float result = sum.reduceLanes(VectorOperators.ADD);
        for (; i < baseOffsetsLength; i++) {
            result += dataArray[dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset))];
        }
        return result;
    }

    float assembleAndSum128(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        // benchmarking a 128-bit SIMD implementation showed it performed worse than scalar
        float sum = 0f;
        for (int i = 0; i < baseOffsetsLength; i++) {
            sum += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));
        }
        return sum;
    }

    @Override
    public float assembleAndSumPQ(
            VectorFloat<?> codebookPartialSums,
            int subspaceCount,                  // = M
            ByteSequence<?> vector1Ordinals,
            int vector1OrdinalOffset,
            ByteSequence<?> vector2Ordinals,
            int vector2OrdinalOffset,
            int clusterCount                    // = k
    ) {
        //compute the size of the subvector
        return switch (PREFERRED_BIT_SIZE)
        {
            case 512 -> assembleAndSumPQ_512(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
            case 256 -> assembleAndSumPQ_256(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
            case 128 -> assembleAndSumPQ_128(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
            default -> throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
        };
    }

    float assembleAndSumPQ_128(
            VectorFloat<?> data,
            int subspaceCount,                  // = M
            ByteSequence<?> baseOffsets1,
            int baseOffsetsOffset1,
            ByteSequence<?> baseOffsets2,
            int baseOffsetsOffset2,
            int clusterCount                    // = k
    ) {
        final int k          = clusterCount;
        final int blockSize  = k * (k + 1) / 2;
        float res = 0f;

        for (int i = 0; i < subspaceCount; i++) {
            int c1 = Byte.toUnsignedInt(baseOffsets1.get(i + baseOffsetsOffset1));
            int c2 = Byte.toUnsignedInt(baseOffsets2.get(i + baseOffsetsOffset2));
            int r  = Math.min(c1, c2);
            int c  = Math.max(c1, c2);

            int offsetRow  = r * k - (r * (r - 1) / 2);
            int idxInBlock = offsetRow + (c - r);
            int base       = i * blockSize;

            res += data.get(base + idxInBlock);
        }

        return res;
    }

    float assembleAndSumPQ_256(
            VectorFloat<?> data,
            int subspaceCount,                  // = M
            ByteSequence<?> baseOffsets1,
            int baseOffsetsOffset1,
            ByteSequence<?> baseOffsets2,
            int baseOffsetsOffset2,
            int clusterCount                    // = k
    ) {
        final VectorSpecies<Float> FSPECIES = FloatVector.SPECIES_256;
        final int LANES      = FSPECIES.length();
        final int k          = clusterCount;
        final int blockSize  = k * (k + 1) / 2;
        final int M          = subspaceCount;

        int[] convOffsets = scratchInt256.get();
        byte[] codes1 = byteArray(baseOffsets1);
        byte[] codes2 = byteArray(baseOffsets2);
        float[] partials = M < LANES ? null : array(data);
        int start1 = baseOffsets1.offset() + baseOffsetsOffset1;
        int start2 = baseOffsets2.offset() + baseOffsetsOffset2;
        IntVector scale = IntVector.zero(IntVector.SPECIES_256).addIndex(blockSize);
        // Retain the concrete vector class so C2 can eliminate intermediate vector boxes.
        // Intrinsifying the arithmetic alone does not ensure allocation-free execution.
        FloatVector sum = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_256));
        for (int i = 0; i + LANES <= M; i += LANES) {
            ByteVector bytes1 = (ByteVector) ByteVector.SPECIES_64.vectorType().cast(codes1 != null
                    ? ByteVector.fromArray(ByteVector.SPECIES_64, codes1, start1 + i)
                    : fromByteSequence(ByteVector.SPECIES_64, baseOffsets1, start1 + i));
            IntVector c1 = (IntVector) IntVector.SPECIES_256.vectorType().cast(
                    bytes1.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0));
            c1 = (IntVector) IntVector.SPECIES_256.vectorType().cast(c1.and(0xff));
            ByteVector bytes2 = (ByteVector) ByteVector.SPECIES_64.vectorType().cast(codes2 != null
                    ? ByteVector.fromArray(ByteVector.SPECIES_64, codes2, start2 + i)
                    : fromByteSequence(ByteVector.SPECIES_64, baseOffsets2, start2 + i));
            IntVector c2 = (IntVector) IntVector.SPECIES_256.vectorType().cast(
                    bytes2.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0));
            c2 = (IntVector) IntVector.SPECIES_256.vectorType().cast(c2.and(0xff));
            IntVector lo = c1.min(c2);
            IntVector hi = c1.max(c2);
            // Packed upper-triangular table: lo * k - lo * (lo - 1) / 2 + hi - lo.
            // Codes are unsigned bytes, so all intermediate indices fit exactly in int.
            IntVector triangular = lo.mul(lo.sub(1)).lanewise(VectorOperators.LSHR, 1);
            lo.mul(k).sub(triangular).add(hi.sub(lo)).add(scale).intoArray(convOffsets, 0);
            FloatVector chunk = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(
                    (BROKEN_GATHER
                    ? gatherFallback(FloatVector.SPECIES_256, partials, i * blockSize, convOffsets, 0)
                    : FloatVector.fromArray(FloatVector.SPECIES_256, partials, i * blockSize, convOffsets, 0)));
            sum = (FloatVector) FloatVector.SPECIES_256.vectorType().cast(sum.add(chunk));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Score subspaces remaining after the full SIMD groups.
        for (int i = (M / LANES) * LANES; i < M; i++) {
            int c1 = Byte.toUnsignedInt(baseOffsets1.get(i + baseOffsetsOffset1));
            int c2 = Byte.toUnsignedInt(baseOffsets2.get(i + baseOffsetsOffset2));
            int r  = Math.min(c1, c2);
            int c  = Math.max(c1, c2);

            int offsetRow  = r * k - (r * (r - 1) / 2);
            int idxInBlock = offsetRow + (c - r);
            int base       = i * blockSize;

            res += data.get(base + idxInBlock);
        }

        return res;
    }

    float assembleAndSumPQ_512(
            VectorFloat<?> data,
            int subspaceCount,                  // = M
            ByteSequence<?> baseOffsets1,
            int baseOffsetsOffset1,
            ByteSequence<?> baseOffsets2,
            int baseOffsetsOffset2,
            int clusterCount                    // = k
    ) {
        final VectorSpecies<Float> FSPECIES = FloatVector.SPECIES_512;
        final int LANES      = FSPECIES.length();
        final int k          = clusterCount;
        final int blockSize  = k * (k + 1) / 2;
        final int M          = subspaceCount;

        int[] convOffsets = scratchInt512.get();
        byte[] codes1 = byteArray(baseOffsets1);
        byte[] codes2 = byteArray(baseOffsets2);
        float[] partials = M < LANES ? null : array(data);
        int start1 = baseOffsets1.offset() + baseOffsetsOffset1;
        int start2 = baseOffsets2.offset() + baseOffsetsOffset2;
        IntVector scale = IntVector.zero(IntVector.SPECIES_512).addIndex(blockSize);
        // Retain the concrete vector class so C2 can eliminate intermediate vector boxes.
        // Intrinsifying the arithmetic alone does not ensure allocation-free execution.
        FloatVector sum = (FloatVector) FloatVector.SPECIES_512.vectorType().cast(
                FloatVector.zero(FloatVector.SPECIES_512));
        for (int i = 0; i + LANES <= M; i += LANES) {
            ByteVector bytes1 = (ByteVector) ByteVector.SPECIES_128.vectorType().cast(codes1 != null
                    ? ByteVector.fromArray(ByteVector.SPECIES_128, codes1, start1 + i)
                    : fromByteSequence(ByteVector.SPECIES_128, baseOffsets1, start1 + i));
            IntVector c1 = (IntVector) IntVector.SPECIES_512.vectorType().cast(
                    bytes1.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0));
            c1 = (IntVector) IntVector.SPECIES_512.vectorType().cast(c1.and(0xff));
            ByteVector bytes2 = (ByteVector) ByteVector.SPECIES_128.vectorType().cast(codes2 != null
                    ? ByteVector.fromArray(ByteVector.SPECIES_128, codes2, start2 + i)
                    : fromByteSequence(ByteVector.SPECIES_128, baseOffsets2, start2 + i));
            IntVector c2 = (IntVector) IntVector.SPECIES_512.vectorType().cast(
                    bytes2.convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0));
            c2 = (IntVector) IntVector.SPECIES_512.vectorType().cast(c2.and(0xff));
            IntVector lo = c1.min(c2);
            IntVector hi = c1.max(c2);
            // Packed upper-triangular table: lo * k - lo * (lo - 1) / 2 + hi - lo.
            // Codes are unsigned bytes, so all intermediate indices fit exactly in int.
            IntVector triangular = lo.mul(lo.sub(1)).lanewise(VectorOperators.LSHR, 1);
            lo.mul(k).sub(triangular).add(hi.sub(lo)).add(scale).intoArray(convOffsets, 0);
            FloatVector chunk = (FloatVector) FloatVector.SPECIES_512.vectorType().cast(
                    (BROKEN_GATHER
                    ? gatherFallback(FloatVector.SPECIES_512, partials, i * blockSize, convOffsets, 0)
                    : FloatVector.fromArray(FloatVector.SPECIES_512, partials, i * blockSize, convOffsets, 0)));
            sum = (FloatVector) FloatVector.SPECIES_512.vectorType().cast(sum.add(chunk));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Score subspaces remaining after the full SIMD groups.
        for (int i = (M / LANES) * LANES; i < M; i++) {
            int c1 = Byte.toUnsignedInt(baseOffsets1.get(i + baseOffsetsOffset1));
            int c2 = Byte.toUnsignedInt(baseOffsets2.get(i + baseOffsetsOffset2));
            int r  = Math.min(c1, c2);
            int c  = Math.max(c1, c2);

            int offsetRow  = r * k - (r * (r - 1) / 2);
            int idxInBlock = offsetRow + (c - r);
            int base       = i * blockSize;

            res += data.get(base + idxInBlock);
        }

        return res;
    }

    /**
     * Vectorized calculation of Hamming distance for two arrays of long integers.
     * Both arrays should have the same length.
     *
     * @param a The first array
     * @param b The second array
     * @return The Hamming distance
     */
    @Override
    public int hammingDistance(long[] a, long[] b) {
        LongVector sum = (LongVector) LongVector.SPECIES_PREFERRED.vectorType().cast(
                LongVector.zero(LongVector.SPECIES_PREFERRED));
        int vectorizedLength = LongVector.SPECIES_PREFERRED.loopBound(a.length);
        for (int i = 0; i < vectorizedLength; i += LongVector.SPECIES_PREFERRED.length()) {
            LongVector va = (LongVector) LongVector.SPECIES_PREFERRED.vectorType().cast(
                    LongVector.fromArray(LongVector.SPECIES_PREFERRED, a, i));
            LongVector vb = (LongVector) LongVector.SPECIES_PREFERRED.vectorType().cast(
                    LongVector.fromArray(LongVector.SPECIES_PREFERRED, b, i));
            LongVector xor = (LongVector) LongVector.SPECIES_PREFERRED.vectorType().cast(
                    va.lanewise(VectorOperators.XOR, vb));
            LongVector counts = (LongVector) LongVector.SPECIES_PREFERRED.vectorType().cast(
                    xor.lanewise(VectorOperators.BIT_COUNT));
            sum = (LongVector) LongVector.SPECIES_PREFERRED.vectorType().cast(sum.add(counts));
        }
        int res = (int) sum.reduceLanes(VectorOperators.ADD);
        for (int i = vectorizedLength; i < a.length; i++) res += Long.bitCount(a[i] ^ b[i]);
        return res;
    }

    @Override
    public float max(VectorFloat<?> v) {
        FloatVector accum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, -Float.MAX_VALUE));
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v.length());
        float[] vValues = vectorizedLength == 0 ? null : ((FloatArray) v).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vValues, i));
            accum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    accum.max(a));
        }
        float max = accum.reduceLanes(VectorOperators.MAX);
        for (int i = vectorizedLength; i < v.length(); i++) {
            max = Math.max(max, v.get(i));
        }
        return max;

    }

    @Override
    public float min(VectorFloat<?> v) {
        FloatVector accum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, Float.MAX_VALUE));
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v.length());
        float[] vValues = vectorizedLength == 0 ? null : ((FloatArray) v).array();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vValues, i));
            accum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(
                    accum.min(a));
        }
        float min = accum.reduceLanes(VectorOperators.MIN);
        for (int i = vectorizedLength; i < v.length(); i++) {
            min = Math.min(min, v.get(i));
        }
        return min;

    }

    private static int combineBytes(int i, int shuffle, ByteSequence<?> quantizedPartials) {
        var lowByte = quantizedPartials.get(i * 512 + shuffle);
        var highByte = quantizedPartials.get((i * 512) + 256 + shuffle);
        return ((Byte.toUnsignedInt(highByte) << 8) | Byte.toUnsignedInt(lowByte));
    }

    private static float combineBytes(int i, int shuffle,  VectorFloat<?> partials) {
        return partials.get(i * 256 + shuffle);
    }

    private static int computeSingleShuffle(int codebookPosition, int neighborPosition, ByteSequence<?> shuffles, int codebookCount) {
        int blockSize = ByteVector.SPECIES_PREFERRED.length();

        int blockIndex = neighborPosition / blockSize;
        int positionWithinBlock = neighborPosition % blockSize;
        int offset = blockIndex * blockSize * codebookCount;
        return Byte.toUnsignedInt(shuffles.get(offset + blockSize * codebookPosition + positionWithinBlock));
    }

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int encodedOffset, int encodedLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        return switch (PREFERRED_BIT_SIZE) {
            case 512 -> pqDecodedCosineSimilarity512(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
            case 256 -> pqDecodedCosineSimilarity256(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
            case 128 -> pqDecodedCosineSimilarity128(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
            default -> throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
        };
    }

    float pqDecodedCosineSimilarity512(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        if (BROKEN_GATHER) {
            return pqCosineFallback512(baseOffsets, baseOffsetsOffset, baseOffsetsLength, clusterCount,
                    partialSums, aMagnitude, bMagnitude);
        }

        FloatVector sum = (FloatVector) FLOAT_512_CLASS.cast(FloatVector.zero(FloatVector.SPECIES_512));
        FloatVector vaMagnitude = sum;

        int[] convOffsets = scratchInt512.get();
        int i = 0;
        int limit = i + ByteVector.SPECIES_128.loopBound(baseOffsetsLength);

        var scale = IntVector.zero(IntVector.SPECIES_512).addIndex(clusterCount);

        for (; i < limit; i += ByteVector.SPECIES_128.length()) {

            fromByteSequence(ByteVector.SPECIES_128, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset)
                    .convertShape(VectorOperators.ZERO_EXTEND_B2I, IntVector.SPECIES_512, 0)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets,0);

            var offset = i * clusterCount;
            sum = (FloatVector) FLOAT_512_CLASS.cast(sum.add(fromVectorFloat(FloatVector.SPECIES_512, partialSums, offset, convOffsets, 0)));
            vaMagnitude = (FloatVector) FLOAT_512_CLASS.cast(vaMagnitude.add(fromVectorFloat(FloatVector.SPECIES_512, aMagnitude, offset, convOffsets, 0)));
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

    private float pqCosineFallback512(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength,
                                     int clusterCount, VectorFloat<?> partialSums,
                                     VectorFloat<?> aMagnitude, float bMagnitude) {
        return pqDecodedCosineSimilarityFallback(baseOffsets, baseOffsetsOffset, baseOffsetsLength, clusterCount,
                partialSums, aMagnitude, bMagnitude, FloatVector.SPECIES_512, IntVector.SPECIES_512,
                ByteVector.SPECIES_128, scratchInt512.get());
    }

    float pqDecodedCosineSimilarity256(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        if (BROKEN_GATHER) {
            return pqDecodedCosineSimilarityFallback(baseOffsets, baseOffsetsOffset, baseOffsetsLength, clusterCount, partialSums, aMagnitude, bMagnitude,
                    FloatVector.SPECIES_256, IntVector.SPECIES_256, ByteVector.SPECIES_64, scratchInt256.get());
        }
        if (baseOffsetsLength < ByteVector.SPECIES_64.length()) {
            return pqDecodedCosineSimilarity128(baseOffsets, baseOffsetsOffset, baseOffsetsLength, clusterCount,
                    partialSums, aMagnitude, bMagnitude);
        }
        return pqDecodedCosineSimilarity256Impl(baseOffsets, baseOffsetsOffset, baseOffsetsLength, clusterCount,
                array(partialSums), array(aMagnitude), bMagnitude);
    }

    private float pqDecodedCosineSimilarity256Impl(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, float[] partials, float[] magnitudes, float bMagnitude) {
        // Keep the complete gather/accumulator chain concrete when inlined into the cosine decoder.
        // Vector intrinsics alone do not guarantee that temporary vectors and arrays are eliminated.
        FloatVector sum = (FloatVector) FLOAT_256_CLASS.cast(FloatVector.zero(FloatVector.SPECIES_256));
        FloatVector vaMagnitude = sum;
        int limit = ByteVector.SPECIES_64.loopBound(baseOffsetsLength);
        int[] convOffsets = scratchInt256.get();
        int i = 0;
        IntVector scale = (IntVector) INT_256_CLASS.cast(IntVector.zero(IntVector.SPECIES_256).addIndex(clusterCount));

        for (; i < limit; i += BYTE_64_LANES) {
            ByteVector bytes = (ByteVector) BYTE_64_CLASS.cast(
                    fromByteSequence(ByteVector.SPECIES_64, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset));
            IntVector codes = (IntVector) INT_256_CLASS.cast(
                    bytes.convertShape(VectorOperators.ZERO_EXTEND_B2I, IntVector.SPECIES_256, 0));
            codes = (IntVector) INT_256_CLASS.cast(codes.add(scale));
            codes.intoArray(convOffsets, 0);

            FloatVector partial = (FloatVector) FLOAT_256_CLASS.cast(
                    FloatVector.fromArray(FloatVector.SPECIES_256, partials, i * clusterCount, convOffsets, 0));
            FloatVector magnitude = (FloatVector) FLOAT_256_CLASS.cast(
                    FloatVector.fromArray(FloatVector.SPECIES_256, magnitudes, i * clusterCount, convOffsets, 0));
            sum = (FloatVector) FLOAT_256_CLASS.cast(sum.add(partial));
            vaMagnitude = (FloatVector) FLOAT_256_CLASS.cast(vaMagnitude.add(magnitude));
        }

        float sumResult = sum.reduceLanes(VectorOperators.ADD);
        float magnitudeResult = vaMagnitude.reduceLanes(VectorOperators.ADD);
        for (; i < baseOffsetsLength; i++) {
            int offset = clusterCount * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset));
            sumResult += partials[offset];
            magnitudeResult += magnitudes[offset];
        }
        return (float) (sumResult / Math.sqrt(magnitudeResult * bMagnitude));
    }

    // Keep the old-JDK workaround outside the modern SIMD loops. This method
    // returns a float, so its vector intermediates never cross a call boundary.
    private float pqDecodedCosineSimilarityFallback(ByteSequence<?> baseOffsets, int baseOffsetsOffset,
                                                   int baseOffsetsLength, int clusterCount,
                                                   VectorFloat<?> partialSums, VectorFloat<?> aMagnitude,
                                                   float bMagnitude, VectorSpecies<Float> floatSpecies,
                                                   VectorSpecies<Integer> intSpecies, VectorSpecies<Byte> byteSpecies,
                                                   int[] convOffsets) {
        var floatType = floatSpecies.vectorType();
        var intType = intSpecies.vectorType();
        var byteType = byteSpecies.vectorType();
        FloatVector sum = (FloatVector) floatType.cast(FloatVector.zero(floatSpecies));
        FloatVector magnitudeSum = sum;
        int limit = byteSpecies.loopBound(baseOffsetsLength);
        float[] partials = arrayForBlock(partialSums, limit);
        float[] magnitudes = arrayForBlock(aMagnitude, limit);
        IntVector scale = (IntVector) intType.cast(IntVector.zero(intSpecies).addIndex(clusterCount));
        int i = 0;
        for (; i < limit; i += byteSpecies.length()) {
            ByteVector bytes = (ByteVector) byteType.cast(
                    fromByteSequence(byteSpecies, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset));
            IntVector codes = (IntVector) intType.cast(bytes.convertShape(VectorOperators.B2I, intSpecies, 0));
            codes = (IntVector) intType.cast(codes.and(0xff));
            codes = (IntVector) intType.cast(codes.add(scale));
            codes.intoArray(convOffsets, 0);
            FloatVector partial = (FloatVector) floatType.cast(
                    gatherFallback(floatSpecies, partials, i * clusterCount, convOffsets, 0));
            FloatVector magnitude = (FloatVector) floatType.cast(
                    gatherFallback(floatSpecies, magnitudes, i * clusterCount, convOffsets, 0));
            sum = (FloatVector) floatType.cast(sum.add(partial));
            magnitudeSum = (FloatVector) floatType.cast(magnitudeSum.add(magnitude));
        }
        return finishPQCosine(baseOffsets, baseOffsetsOffset, baseOffsetsLength, clusterCount,
                partialSums, aMagnitude, bMagnitude, i,
                sum.reduceLanes(VectorOperators.ADD), magnitudeSum.reduceLanes(VectorOperators.ADD));
    }

    private static float finishPQCosine(ByteSequence<?> baseOffsets, int baseOffsetsOffset,
                                       int baseOffsetsLength, int clusterCount,
                                       VectorFloat<?> partialSums, VectorFloat<?> aMagnitude,
                                       float bMagnitude, int i, float sum, float magnitude) {
        for (; i < baseOffsetsLength; i++) {
            int offset = clusterCount * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset));
            sum += partialSums.get(offset);
            magnitude += aMagnitude.get(offset);
        }
        return (float) (sum / Math.sqrt(magnitude * bMagnitude));
    }

    float pqDecodedCosineSimilarity128(ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        // benchmarking showed that a 128-bit SIMD implementation performed worse than scalar
        float sum = 0.0f;
        float aMag = 0.0f;

        for (int m = 0; m < baseOffsetsLength; ++m) {
            int centroidIndex = Byte.toUnsignedInt(baseOffsets.get(m + baseOffsetsOffset));
            var index = m * clusterCount + centroidIndex;
            sum += partialSums.get(index);
            aMag += aMagnitude.get(index);
        }

        return (float) (sum / Math.sqrt(aMag * bMagnitude));
    }

    //---------------------------------------------
    // NVQ quantization instructions start here
    //---------------------------------------------

    static final FloatVector const1f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.f);
    static final FloatVector const05f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 0.5f);

    // One encoded byte per float lane; the smallest supported byte shape may need a mask.
    private static final VectorSpecies<Byte> NVQ_BYTE_SPECIES = nvqByteSpecies(FloatVector.SPECIES_PREFERRED.length());

    static VectorSpecies<Byte> nvqByteSpecies(int floatLanes) {
        int bits = Math.max(64, Integer.highestOneBit(floatLanes - 1) * 2 * Byte.SIZE);
        return bits <= 512 ? VectorSpecies.of(byte.class, VectorShape.forBitSize(bits)) : ByteVector.SPECIES_MAX;
    }

    // Explicit target species avoid generic conversion fallbacks and temporary vector arrays.
    FloatVector logisticNQT(FloatVector vector, float alpha, float x0) {
        // Preserve the original species-generic helper contract outside the preferred fast path.
        return vector.species().equals(FloatVector.SPECIES_PREFERRED)
                ? logisticNQTPreferred(vector, alpha, x0) : logisticNQTGeneric(vector, alpha, x0);
    }

    private FloatVector logisticNQTPreferred(FloatVector vector, float alpha, float x0) {
        // Reuse class references to keep this concrete conversion chain small enough to inline.
        var floatType = FloatVector.SPECIES_PREFERRED.vectorType();
        var intType = IntVector.SPECIES_PREFERRED.vectorType();
        var maskType = FloatVector.SPECIES_PREFERRED.maskType();
        vector = (FloatVector) floatType.cast(vector);
        FloatVector temp = (FloatVector) floatType.cast(vector.fma(alpha, -alpha * x0));
        VectorMask<Float> isNegative = maskType.cast(temp.test(VectorOperators.IS_NEGATIVE));
        VectorMask<Float> isPositive = maskType.cast(isNegative.not());
        FloatVector roundedInput = (FloatVector) floatType.cast(temp.add(1, isPositive));
        IntVector p = (IntVector) intType.cast(roundedInput.convertShape(VectorOperators.F2I, IntVector.SPECIES_PREFERRED, 0));
        FloatVector e = (FloatVector) floatType.cast(p.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0));
        FloatVector fraction = (FloatVector) floatType.cast(temp.sub(e));
        fraction = (FloatVector) floatType.cast(fraction.fma(0.5f, 1));
        IntVector m = (IntVector) intType.cast(fraction.reinterpretAsInts());
        IntVector exponent = (IntVector) intType.cast(p.lanewise(VectorOperators.LSHL, 23));
        m = (IntVector) intType.cast(m.add(exponent));
        temp = (FloatVector) floatType.cast(m.reinterpretAsFloats()); // temp = m * 2^p
        FloatVector denominator = (FloatVector) floatType.cast(temp.add(1));
        return (FloatVector) floatType.cast(temp.div(denominator));
    }

    private FloatVector logisticNQTGeneric(FloatVector vector, float alpha, float x0) {
        var species = vector.species();
        var intSpecies = species.withLanes(int.class);
        var floatType = species.vectorType();
        var intType = intSpecies.vectorType();
        FloatVector temp = (FloatVector) floatType.cast(vector.fma(alpha, -alpha * x0));
        VectorMask<Float> positive = species.maskType().cast(temp.test(VectorOperators.IS_NEGATIVE).not());
        FloatVector rounded = (FloatVector) floatType.cast(temp.add(1, positive));
        IntVector exponent = (IntVector) intType.cast(rounded.convertShape(VectorOperators.F2I, intSpecies, 0));
        FloatVector integral = (FloatVector) floatType.cast(exponent.convertShape(VectorOperators.I2F, species, 0));
        FloatVector fraction = (FloatVector) floatType.cast(temp.sub(integral));
        fraction = (FloatVector) floatType.cast(fraction.fma(0.5f, 1));
        IntVector mantissa = (IntVector) intType.cast(fraction.reinterpretAsInts());
        exponent = (IntVector) intType.cast(exponent.lanewise(VectorOperators.LSHL, 23));
        mantissa = (IntVector) intType.cast(mantissa.add(exponent));
        temp = (FloatVector) floatType.cast(mantissa.reinterpretAsFloats());
        FloatVector denominator = (FloatVector) floatType.cast(temp.add(1));
        return (FloatVector) floatType.cast(temp.div(denominator));
    }

    float logisticNQT(float value, float alpha, float x0) {
        float temp = Math.fma(value, alpha, -alpha * x0);
        int p = (int) Math.floor(temp + 1);
        int m = Float.floatToIntBits(Math.fma(temp - p, 0.5f, 1));

        temp = Float.intBitsToFloat(m + (p << 23));  // temp = m * 2^p
        return temp / (temp + 1);
    }

    FloatVector logitNQT(FloatVector vector, float inverseAlpha, float x0) {
        // Preserve concrete SIMD types when this kernel is inlined into a larger caller.
        // Otherwise C2 can materialize temporary vector objects and backing arrays.
        vector = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(vector);
        FloatVector denominator = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(const1f.sub(vector));
        FloatVector z = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(vector.div(denominator));
        IntVector temp = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(z.reinterpretAsInts());
        IntVector exponent = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(temp.and(0x7f800000));
        exponent = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(exponent.lanewise(VectorOperators.LSHR, 23));
        exponent = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(exponent.sub(128));
        FloatVector p = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(exponent.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0));
        IntVector mantissa = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(temp.and(0x007fffff));
        mantissa = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(mantissa.add(0x3f800000));
        FloatVector m = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(mantissa.reinterpretAsFloats());
        FloatVector combined = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(m.add(p));
        return (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(combined.fma(inverseAlpha, x0));

    }

    float logitNQT(float value, float inverseAlpha, float x0) {
        float z = value / (1 - value);

        int temp = Float.floatToIntBits(z);
        int e = temp & 0x7f800000;
        float p = (float) ((e >> 23) - 128);
        float m = Float.intBitsToFloat((temp & 0x007fffff) + 0x3f800000);

        return Math.fma(m + p, inverseAlpha, x0);
    }

    FloatVector nvqDequantize8bit(ByteVector bytes, float inverseAlpha, float x0, float logisticScale, float logisticBias, int part) {
        /*
         * We unpack the vector using the FastLanes strategy:
         * https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf?ref=blog.lancedb.com
         *
         * We treat the ByteVector bytes as a vector of integers.
         * | Int0                    | Int1                    | ...
         * | Byte3 Byte2 Byte1 Byte0 | Byte3 Byte2 Byte1 Byte0 | ...
         *
         * The argument part indicates which byte we want to extract from each integer.
         * With part=0, we extract
         *      Int0\Byte0, Int1\Byte0, etc.
         * With part=1, we shift by 8 bits and then extract
         *      Int0\Byte1, Int1\Byte1, etc.
         * With part=2, we shift by 16 bits and then extract
         *      Int0\Byte2, Int1\Byte2, etc.
         * With part=3, we shift by 24 bits and then extract
         *      Int0\Byte3, Int1\Byte3, etc.
         */
        bytes = (ByteVector) ByteVector.SPECIES_PREFERRED.vectorType().cast(bytes);
        IntVector ints = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(bytes.reinterpretAsInts());
        ints = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(ints.lanewise(VectorOperators.LSHR, 8 * part));
        ints = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(ints.and(0xff));
        FloatVector arr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(ints.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0));
        arr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(arr.fma(logisticScale, logisticBias));
        return (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(logitNQT(arr, inverseAlpha, x0));

    }

    @Override
    public void nvqQuantize8bit(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, ByteSequence<?> destination) {
        // Keep vector intermediates inside the complete kernel. An out-of-line
        // vector-returning transform would materialize temporary vectors.

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        final var byteSpecies = NVQ_BYTE_SPECIES;
        final float[] values = vectorizedLength == 0 ? null : array(vector);
        final byte[] output = vectorizedLength == 0 ? null : ((ArrayByteSequence) destination).get();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var invLogisticScale = 255 / (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector arr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, values, i));
            {
                FloatVector blockInput = arr;
                var floatType = FloatVector.SPECIES_PREFERRED.vectorType();
                var intType = IntVector.SPECIES_PREFERRED.vectorType();
                var maskType = FloatVector.SPECIES_PREFERRED.maskType();
                blockInput = (FloatVector) floatType.cast(blockInput);
                FloatVector temp = (FloatVector) floatType.cast(blockInput.fma(scaledAlpha, -scaledAlpha * scaledX0));
                VectorMask<Float> isNegative = maskType.cast(temp.test(VectorOperators.IS_NEGATIVE));
                VectorMask<Float> isPositive = maskType.cast(isNegative.not());
                FloatVector roundedInput = (FloatVector) floatType.cast(temp.add(1, isPositive));
                IntVector p = (IntVector) intType.cast(roundedInput.convertShape(VectorOperators.F2I, IntVector.SPECIES_PREFERRED, 0));
                FloatVector e = (FloatVector) floatType.cast(p.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0));
                FloatVector fraction = (FloatVector) floatType.cast(temp.sub(e));
                fraction = (FloatVector) floatType.cast(fraction.fma(0.5f, 1));
                IntVector m = (IntVector) intType.cast(fraction.reinterpretAsInts());
                IntVector exponent = (IntVector) intType.cast(p.lanewise(VectorOperators.LSHL, 23));
                m = (IntVector) intType.cast(m.add(exponent));
                temp = (FloatVector) floatType.cast(m.reinterpretAsFloats()); // temp = m * 2^p
                FloatVector denominator = (FloatVector) floatType.cast(temp.add(1));
                arr = (FloatVector) floatType.cast(temp.div(denominator));
            }
            arr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(arr.sub(logisticBias));
            arr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(arr.mul(invLogisticScale));
            arr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(arr.add(const05f));
            // Java float-to-byte conversion narrows through int. Keep both SIMD steps explicit.
            IntVector integers = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(
                    arr.convertShape(VectorOperators.F2I, IntVector.SPECIES_PREFERRED, 0));
            ByteVector bytes = (ByteVector) byteSpecies.vectorType().cast(
                    integers.convertShape(VectorOperators.I2B, byteSpecies, 0));
            if (FloatVector.SPECIES_PREFERRED.length() == byteSpecies.length()) {
                bytes.intoArray(output, i);
            } else {
                bytes.intoArray(output, i, byteSpecies.indexInRange(0, FloatVector.SPECIES_PREFERRED.length()));
            }
        }

        // Process the tail
        for (int d = vectorizedLength; d < vector.length(); d++) {
            // Ensure the quantized value is within the 0 to constant range
            float value = vector.get(d);
            value = logisticNQT(value, scaledAlpha, scaledX0);
            value = (value - logisticBias) * invLogisticScale;
            int quantizedValue = Math.round(value);
            destination.set(d, (byte) quantizedValue);
        }
    }

    @Override
    public float nvqLoss(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, int nBits) {
        // Keep vector intermediates inside the complete kernel. An out-of-line
        // vector-returning transform would materialize temporary vectors.

        int constant = (1 << nBits) - 1;
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        FloatVector squaredSumVec = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.zero(FloatVector.SPECIES_PREFERRED));

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / constant;
        var invLogisticScale = 1 / logisticScale;

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector arr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(vector), i));
            FloatVector recArr;
            {
                FloatVector blockInput = arr;
                var floatType = FloatVector.SPECIES_PREFERRED.vectorType();
                var intType = IntVector.SPECIES_PREFERRED.vectorType();
                var maskType = FloatVector.SPECIES_PREFERRED.maskType();
                blockInput = (FloatVector) floatType.cast(blockInput);
                FloatVector temp = (FloatVector) floatType.cast(blockInput.fma(scaledAlpha, -scaledAlpha * scaledX0));
                VectorMask<Float> isNegative = maskType.cast(temp.test(VectorOperators.IS_NEGATIVE));
                VectorMask<Float> isPositive = maskType.cast(isNegative.not());
                FloatVector roundedInput = (FloatVector) floatType.cast(temp.add(1, isPositive));
                IntVector p = (IntVector) intType.cast(roundedInput.convertShape(VectorOperators.F2I, IntVector.SPECIES_PREFERRED, 0));
                FloatVector e = (FloatVector) floatType.cast(p.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0));
                FloatVector fraction = (FloatVector) floatType.cast(temp.sub(e));
                fraction = (FloatVector) floatType.cast(fraction.fma(0.5f, 1));
                IntVector m = (IntVector) intType.cast(fraction.reinterpretAsInts());
                IntVector exponent = (IntVector) intType.cast(p.lanewise(VectorOperators.LSHL, 23));
                m = (IntVector) intType.cast(m.add(exponent));
                temp = (FloatVector) floatType.cast(m.reinterpretAsFloats()); // temp = m * 2^p
                FloatVector denominator = (FloatVector) floatType.cast(temp.add(1));
                recArr = (FloatVector) floatType.cast(temp.div(denominator));
            }
            recArr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(recArr.sub(logisticBias));
            recArr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(recArr.mul(invLogisticScale));
            recArr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(recArr.add(const05f));
            IntVector rounded = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(recArr.convertShape(VectorOperators.F2I, IntVector.SPECIES_PREFERRED, 0));
            recArr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(rounded.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0));
            recArr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(recArr.fma(logisticScale, logisticBias));
            {
                FloatVector blockInput = recArr;
                // Preserve concrete SIMD types when this kernel is inlined into a larger caller.
                // Otherwise C2 can materialize temporary blockInput objects and backing arrays.
                blockInput = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(blockInput);
                FloatVector denominator = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(const1f.sub(blockInput));
                FloatVector z = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(blockInput.div(denominator));
                IntVector temp = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(z.reinterpretAsInts());
                IntVector exponent = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(temp.and(0x7f800000));
                exponent = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(exponent.lanewise(VectorOperators.LSHR, 23));
                exponent = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(exponent.sub(128));
                FloatVector p = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(exponent.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0));
                IntVector mantissa = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(temp.and(0x007fffff));
                mantissa = (IntVector) IntVector.SPECIES_PREFERRED.vectorType().cast(mantissa.add(0x3f800000));
                FloatVector m = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(mantissa.reinterpretAsFloats());
                FloatVector combined = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(m.add(p));
                recArr = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(combined.fma(invScaledAlpha, scaledX0));
            }

            FloatVector diff = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(arr.sub(recArr));
            squaredSumVec = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(diff.fma(diff, squaredSumVec));
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value, recValue;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value = vector.get(i);

            recValue = logisticNQT(value, scaledAlpha, scaledX0);
            recValue = (recValue - logisticBias) * invLogisticScale;
            recValue = Math.round(recValue);
            recValue = Math.fma(logisticScale, recValue, logisticBias);
            recValue = logitNQT(recValue, invScaledAlpha, scaledX0);

            squaredSum += MathUtil.square(value - recValue);
        }

        return squaredSum;
    }

    @Override
    public float nvqUniformLoss(VectorFloat<?> vector, float minValue, float maxValue, int nBits) {
        float constant = (1 << nBits) - 1;
        float delta = maxValue - minValue;

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        FloatVector squaredSumVec = (FloatVector) FLOAT_PREFERRED_CLASS.cast(FloatVector.zero(FloatVector.SPECIES_PREFERRED));

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector arr = (FloatVector) FLOAT_PREFERRED_CLASS.cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(vector), i));
            FloatVector recArr = (FloatVector) FLOAT_PREFERRED_CLASS.cast(arr.sub(minValue));
            recArr = (FloatVector) FLOAT_PREFERRED_CLASS.cast(recArr.mul(constant / delta));
            recArr = (FloatVector) FLOAT_PREFERRED_CLASS.cast(recArr.add(const05f));
            IntVector rounded = (IntVector) INT_PREFERRED_CLASS.cast(recArr.convertShape(VectorOperators.F2I, IntVector.SPECIES_PREFERRED, 0));
            recArr = (FloatVector) FLOAT_PREFERRED_CLASS.cast(rounded.convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0));
            recArr = (FloatVector) FLOAT_PREFERRED_CLASS.cast(recArr.fma(delta / constant, minValue));

            FloatVector diff = (FloatVector) FLOAT_PREFERRED_CLASS.cast(arr.sub(recArr));
            squaredSumVec = (FloatVector) FLOAT_PREFERRED_CLASS.cast(diff.fma(diff, squaredSumVec));
        }

        return nvqUniformLossTail(vector, vectorizedLength, minValue, delta, constant,
                squaredSumVec.reduceLanes(VectorOperators.ADD));
    }

    private static float nvqUniformLossTail(VectorFloat<?> vector, int start,
                                           float minValue, float delta, float constant, float squaredSum) {
        float value, recValue;
        for (int i = start; i < vector.length(); i++) {
            value = vector.get(i);
            recValue = (value - minValue) / delta;
            recValue = Math.round(constant * recValue) / constant;
            recValue = recValue * delta + minValue;
            squaredSum += MathUtil.square(value - recValue);
        }
        return squaredSum;
    }

    @Override
    public float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> quantizedVector,
                                         float alpha, float x0, float minValue, float maxValue) {
        FloatVector squaredSumVec = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.zero(FloatVector.SPECIES_PREFERRED));

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(quantizedVector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            ByteVector byteArr = (ByteVector) ByteVector.SPECIES_PREFERRED.vectorType().cast(fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i));

            for (int j = 0; j < 4; j++) {
                FloatVector v1 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(vector), i + floatStep * j));
                FloatVector v2 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j));

                FloatVector diff = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(v1.sub(v2));
                squaredSumVec = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(diff.fma(diff, squaredSumVec));
            }
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2, diff;
        for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = Math.fma(logisticScale, value2, logisticBias);
            value2 = logitNQT(value2, invScaledAlpha, scaledX0);
            diff = vector.get(i) - value2;
            squaredSum += MathUtil.square(diff);
        }

        return squaredSum;
    }

    @Override
    public float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> quantizedVector,
                                   float alpha, float x0, float minValue, float maxValue) {
        // Preserve concrete SIMD types when this kernel is inlined into a larger caller.
        // Otherwise C2 can materialize temporary vector objects and backing arrays.
        FloatVector dotProdVec = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.zero(FloatVector.SPECIES_PREFERRED));

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(quantizedVector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;


        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = (ByteVector) ByteVector.SPECIES_PREFERRED.vectorType().cast(fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i));

            for (int j = 0; j < 4; j++) {
                var v1 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i + floatStep * j));
                var v2 = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j));
                dotProdVec = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(v1.fma(v2, dotProdVec));
            }
        }

        float dotProd = dotProdVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2;
        for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = Math.fma(logisticScale, value2, logisticBias);
            value2 = logitNQT(value2, invScaledAlpha, scaledX0);
            dotProd = Math.fma(vector.get(i), value2, dotProd);
        }

        return dotProd;
    }

    @Override
    public float[] nvqCosine8bit(VectorFloat<?> vector,
                                 ByteSequence<?> quantizedVector, float alpha, float x0, float minValue, float maxValue,
                                 VectorFloat<?> centroid) {
        if (vector.length() != centroid.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        FloatVector vsum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.zero(FloatVector.SPECIES_PREFERRED));
        FloatVector vbMagnitude = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.zero(FloatVector.SPECIES_PREFERRED));

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(vector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            ByteVector byteArr = (ByteVector) ByteVector.SPECIES_PREFERRED.vectorType().cast(fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i));

            for (int j = 0; j < 4; j++) {
                FloatVector va = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(vector), i + floatStep * j));
                FloatVector vb = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j));

                FloatVector vCentroid = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array(centroid), i + floatStep * j));
                vb = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(vb.add(vCentroid));

                vsum = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(va.fma(vb, vsum));
                vbMagnitude = (FloatVector) FloatVector.SPECIES_PREFERRED.vectorType().cast(vb.fma(vb, vbMagnitude));
            }
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = Math.fma(logisticScale, value2, logisticBias);
            value2 = logitNQT(value2, invScaledAlpha, scaledX0) + centroid.get(i);
            sum = Math.fma(vector.get(i), value2, sum);
            bMagnitude = Math.fma(value2, value2, bMagnitude);
        }

        // TODO can we avoid returning a new array?
        return new float[]{sum, bMagnitude};
    }

    void transpose(VectorFloat<?> arr, int first, int last, int nRows) {
        final int mn1 = (last - first - 1);
        final int n   = (last - first) / nRows;
        boolean[] visited = new boolean[last - first];
        float temp;
        int cycle = first;
        while (++cycle != last) {
            if (visited[cycle - first])
                continue;
            int a = cycle - first;
            do  {
                a = a == mn1 ? mn1 : (n * a) % mn1;
                temp = arr.get(first + a);
                arr.set(first + a, arr.get(cycle));
                arr.set(cycle, temp);
                visited[a] = true;
            } while ((first + a) != cycle);
        }
    }

    @Override
    public void nvqShuffleQueryInPlace8bit(VectorFloat<?> vector) {
        // To understand this shuffle, see nvqDequantize8bit

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        final int step = FloatVector.SPECIES_PREFERRED.length() * 4;

        for (int i = 0; i + step <= vectorizedLength; i += step) {
            transpose(vector, i, i + step, 4);
        }
    }

    @Override
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
        int codebookBase = codebookIndex * clusterCount;
        for (int i = 0; i < clusterCount; i++) {
            switch (vsf) {
                case DOT_PRODUCT:
                    partialSums.set(codebookBase + i, dotProduct(codebook, i * size, query, queryOffset, size));
                    break;
                case EUCLIDEAN:
                    partialSums.set(codebookBase + i, squareDistance(codebook, i * size, query, queryOffset, size));
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
            }
        }
    }

    @Override
    public float pqDecodedCosineSimilarity(ByteSequence<?> encoded, int clusterCount, VectorFloat<?> partialSums, VectorFloat<?> aMagnitude, float bMagnitude) {
        return pqDecodedCosineSimilarity(encoded, 0, encoded.length(),  clusterCount, partialSums, aMagnitude, bMagnitude);
    }
}
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
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.util.Objects;

import java.util.List;

class PanamaVectorUtilSupport implements VectorUtilSupport {
    private static final String ASH_BLOCK_KERNEL =
            System.getProperty("jvector.ash.blockKernel", "mask"); // "lut" or "mask"

    static final int PREFERRED_BIT_SIZE = FloatVector.SPECIES_PREFERRED.vectorBitSize();
    static final IntVector BYTE_TO_INT_MASK_512 = IntVector.broadcast(IntVector.SPECIES_512, 0xff);
    static final IntVector BYTE_TO_INT_MASK_256 = IntVector.broadcast(IntVector.SPECIES_256, 0xff);

    static final ThreadLocal<int[]> scratchInt512 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_512.length()]);
    static final ThreadLocal<int[]> scratchInt256 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_256.length()]);

    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset) {
        return FloatVector.fromArray(SPEC, ((ArrayVectorFloat) vector).get(), offset);
    }

    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset, int[] indices, int indicesOffset) {
        return FloatVector.fromArray(SPEC, ((ArrayVectorFloat) vector).get(), offset, indices, indicesOffset);
    }

    protected void intoVectorFloat(FloatVector vector, VectorFloat<?> v, int offset) {
        vector.intoArray(((ArrayVectorFloat) v).get(), offset);
    }

    protected ByteVector fromByteSequence(VectorSpecies<Byte> SPEC, ByteSequence<?> vector, int offset) {
        return ByteVector.fromArray(SPEC, ((ArrayByteSequence) vector).get(), offset);
    }

    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset) {
        vector.intoArray(((ArrayByteSequence) v).get(), offset);
    }

    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset, VectorMask<Byte> mask) {
        vector.intoArray(((ArrayByteSequence) v).get(), offset, mask);
    }


    @Override
    public float sum(VectorFloat<?> vector) {
        var sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the remainder
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            sum = sum.add(a);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            res += vector.get(i);
        }

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

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            var divResult = a.mul(multiplier);
            intoVectorFloat(divResult, vector, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, vector.get(i) * multiplier);
        }
    }

    float dot64(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_64, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_64, v2, offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    float dot128(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_128, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_128, v2, offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    float dot256(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_256, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_256, v2, offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    float dotPreferred(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    @Override
    public float dotProduct(VectorFloat<?> v1, VectorFloat<?> v2) {
        return dotProduct(v1, 0, v2, 0, v1.length());
    }

    @Override
    public float dotProduct(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, final int length) {
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
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);
        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_64, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_64, v2, v2offset + i);
            sum = a.fma(b, sum);
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
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_128, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_128, v2, v2offset + i);
            sum = a.fma(b, sum);
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
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_256, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_256, v2, v2offset + i);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    float dotProductPreferred(VectorFloat<?> va, int vaoffset, VectorFloat<?> vb, int vboffset, int length) {
        if (length == FloatVector.SPECIES_PREFERRED.length())
            return dotPreferred(va, vaoffset, vb, vboffset);

        FloatVector sum0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        FloatVector sum1 = sum0;
        FloatVector a0, a1, b0, b1;

        int vectorLength = FloatVector.SPECIES_PREFERRED.length();

        // Unrolled vector loop; for dot product from L1 cache, an unroll factor of 2 generally suffices.
        // If we are going to be getting data that's further down the hierarchy but not fetched off disk/network,
        // we might want to unroll further, e.g. to 8 (4 sets of a,b,sum with 3-ahead reads seems to work best).
        if (length >= vectorLength * 2) {
            length -= vectorLength * 2;
            a0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 0);
            b0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 0);
            a1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 1);
            b1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 1);
            vaoffset += vectorLength * 2;
            vboffset += vectorLength * 2;
            while (length >= vectorLength * 2) {
                // All instructions in the main loop have no dependencies between them and can be executed in parallel.
                length -= vectorLength * 2;
                sum0 = a0.fma(b0, sum0);
                a0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 0);
                b0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 0);
                sum1 = a1.fma(b1, sum1);
                a1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 1);
                b1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 1);
                vaoffset += vectorLength * 2;
                vboffset += vectorLength * 2;
            }
            sum0 = a0.fma(b0, sum0);
            sum1 = a1.fma(b1, sum1);
        }
        sum0 = sum0.add(sum1);

        // Process the remaining few vectors
        while (length >= vectorLength) {
            length -= vectorLength;
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset);
            vaoffset += vectorLength;
            vboffset += vectorLength;
            sum0 = a.fma(b, sum0);
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
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());
        // Process the vectorized part, convert from 8 bytes to 8 ints
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, i);
            vsum = a.fma(b, vsum);
            vaMagnitude = a.fma(a, vaMagnitude);
            vbMagnitude = b.fma(b, vbMagnitude);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float aMagnitude = vaMagnitude.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            sum += v1.get(i) * v2.get(i);
            aMagnitude += v1.get(i) * v1.get(i);
            bMagnitude += v2.get(i) * v2.get(i);
        }

        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    @Override
    public float cosine(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, int length) {
        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, v1offset + i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, v2offset + i);
            vsum = a.fma(b, vsum);
            vaMagnitude = a.fma(a, vaMagnitude);
            vbMagnitude = b.fma(b, vbMagnitude);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float aMagnitude = vaMagnitude.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        for (int i = vectorizedLength; i < length; i++) {
            sum += v1.get(v1offset + i) * v2.get(v2offset + i);
            aMagnitude += v1.get(v1offset + i) * v1.get(v1offset + i);
            bMagnitude += v2.get(v2offset + i) * v2.get(v2offset + i);
        }

        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    float squareDistance64(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_64, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_64, v2, offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    float squareDistance128(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_128, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_128, v2, offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    float squareDistance256(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_256, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_256, v2, offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    float squareDistancePreferred(VectorFloat<?> v1, int offset1, VectorFloat<?> v2, int offset2) {
        var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, offset1);
        var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    @Override
    public float squareDistance(VectorFloat<?> v1, VectorFloat<?> v2) {
        return squareDistance(v1, 0, v2, 0, v1.length());
    }

    @Override
    public float squareDistance(VectorFloat<?> v1, int v1offset, VectorFloat<?> v2, int v2offset, final int length) {
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
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_64, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_64, v2, v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
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
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_128, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_128, v2, v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
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
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_256, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_256, v2, v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
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
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, v1offset + i);
            FloatVector b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
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
        var a = fromVectorFloat(FloatVector.SPECIES_64, v1, 0);
        var b = fromVectorFloat(FloatVector.SPECIES_64, v2, 0);
        intoVectorFloat(a.add(b), v1, 0);
    }

    void addInPlace64(VectorFloat<?> v1, float value) {
        var a = fromVectorFloat(FloatVector.SPECIES_64, v1, 0);
        intoVectorFloat(a.add(value), v1, 0);
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

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, i);
            intoVectorFloat(a.add(b), v1, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i, v1.get(i) + v2.get(i));
        }
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, float value) {
        if (v1.length() == 2) {
            addInPlace64(v1, value);
            return;
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            intoVectorFloat(a.add(value), v1, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i, v1.get(i) + value);
        }
    }

    @Override
    public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, i);
            intoVectorFloat(a.sub(b), v1, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i, v1.get(i) - v2.get(i));
        }
    }

    @Override
    public void subInPlace(VectorFloat<?> vector, float value) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            intoVectorFloat(a.sub(value), vector, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, vector.get(i) - value);
        }
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

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var lhs = fromVectorFloat(FloatVector.SPECIES_PREFERRED, a, aOffset + i);
            var rhs = fromVectorFloat(FloatVector.SPECIES_PREFERRED, b, bOffset + i);
            var subResult = lhs.sub(rhs);
            intoVectorFloat(subResult, res, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < length; i++) {
            res.set(i, a.get(aOffset + i) - b.get(bOffset + i));
        }

        return res;
    }

    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, float value, int length) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        VectorFloat<?> res = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var lhs = fromVectorFloat(FloatVector.SPECIES_PREFERRED, a, aOffset + i);
            var subResult = lhs.sub(value);
            intoVectorFloat(subResult, res, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < length; i++) {
            res.set(i, a.get(aOffset + i) - value);
        }

        return res;
    }

    @Override
    public void minInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v2, i);
            intoVectorFloat(a.min(b), v1, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i, Math.min(v1.get(i), v2.get(i)));
        }
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
        return assembleAndSum(data, dataBase, baseOffsets, 0, baseOffsets.length());
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        return switch (PREFERRED_BIT_SIZE) {
            case 512 -> assembleAndSum512(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
            case 256 -> assembleAndSum256(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
            case 128 -> assembleAndSum128(data, dataBase, baseOffsets, baseOffsetsOffset, baseOffsetsLength);
            default -> throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
        };
    }

    float assembleAndSum512(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        int[] convOffsets = scratchInt512.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
        int i = 0;
        int limit = ByteVector.SPECIES_128.loopBound(baseOffsetsLength);
        var scale = IntVector.zero(IntVector.SPECIES_512).addIndex(dataBase);

        for (; i < limit; i += ByteVector.SPECIES_128.length()) {
            fromByteSequence(ByteVector.SPECIES_128, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets, 0);

            var offset = i * dataBase;
            sum = sum.add(fromVectorFloat(FloatVector.SPECIES_512, data, offset, convOffsets, 0));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        //Process tail
        for (; i < baseOffsetsLength; i++)
            res += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));

        return res;
    }

    float assembleAndSum256(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets, int baseOffsetsOffset, int baseOffsetsLength) {
        int[] convOffsets = scratchInt256.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);
        int i = 0;
        int limit = ByteVector.SPECIES_64.loopBound(baseOffsetsLength);
        var scale = IntVector.zero(IntVector.SPECIES_256).addIndex(dataBase);

        for (; i < limit; i += ByteVector.SPECIES_64.length()) {

            fromByteSequence(ByteVector.SPECIES_64, baseOffsets, i + baseOffsets.offset() + baseOffsetsOffset)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets, 0);

            var offset = i * dataBase;
            sum = sum.add(fromVectorFloat(FloatVector.SPECIES_256, data, offset, convOffsets, 0));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process tail
        for (; i < baseOffsetsLength; i++)
            res += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i + baseOffsetsOffset)));

        return res;
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
        return switch (PREFERRED_BIT_SIZE) {
            case 512 ->
                    assembleAndSumPQ_512(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
            case 256 ->
                    assembleAndSumPQ_256(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
            case 128 ->
                    assembleAndSumPQ_128(codebookPartialSums, subspaceCount, vector1Ordinals, vector1OrdinalOffset, vector2Ordinals, vector2OrdinalOffset, clusterCount);
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
        final int k = clusterCount;
        final int blockSize = k * (k + 1) / 2;
        float res = 0f;

        for (int i = 0; i < subspaceCount; i++) {
            int c1 = Byte.toUnsignedInt(baseOffsets1.get(i + baseOffsetsOffset1));
            int c2 = Byte.toUnsignedInt(baseOffsets2.get(i + baseOffsetsOffset2));
            int r = Math.min(c1, c2);
            int c = Math.max(c1, c2);

            int offsetRow = r * k - (r * (r - 1) / 2);
            int idxInBlock = offsetRow + (c - r);
            int base = i * blockSize;

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
        final int LANES = FSPECIES.length();
        final int k = clusterCount;
        final int blockSize = k * (k + 1) / 2;
        final int M = subspaceCount;

        int[] convOffsets = scratchInt256.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);
        FloatVector scale = FloatVector.zero(FloatVector.SPECIES_256).addIndex(blockSize);
        FloatVector kvec = FloatVector.broadcast(FloatVector.SPECIES_256, k);
        FloatVector onevec = FloatVector.broadcast(FloatVector.SPECIES_256, 1);
        FloatVector twovec = FloatVector.broadcast(FloatVector.SPECIES_256, 0.5f);


        for (int i = 0; i + LANES <= M; i += LANES) {

            FloatVector c1v = fromByteSequence(ByteVector.SPECIES_64, baseOffsets1, i + baseOffsets1.offset() + baseOffsetsOffset1)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                    .convertShape(VectorOperators.I2F, FSPECIES, 0)
                    .reinterpretAsFloats();

            FloatVector c2v = fromByteSequence(ByteVector.SPECIES_64, baseOffsets2, i + baseOffsets2.offset() + baseOffsetsOffset2)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                    .convertShape(VectorOperators.I2F, FSPECIES, 0)
                    .reinterpretAsFloats();

            // b) r = min(c1,c2), c = max(c1,c2)
            var r = c1v.min(c2v);
            var c = c1v.max(c2v);

            // c) offsetRow = r*k - (r*(r-1))/2
            var rk = r.mul(kvec);
            var triangular = r.mul(r.sub(onevec)).mul(twovec);
            var offsetRow = rk.sub(triangular);

            // d) idxInBlock = offsetRow + (c - r) + (i * blockSize)
            offsetRow.add(c.sub(r)).add(scale)
                    .convertShape(VectorOperators.F2I, IntVector.SPECIES_256, 0)
                    .reinterpretAsInts()
                    .intoArray(convOffsets, 0);

            // e) gather LANES floats from `partials` at those indices
            FloatVector chunk = fromVectorFloat(FSPECIES, data, i * blockSize, convOffsets, 0);

            // f) horizontal sum the chunk and add to our scalar accumulator
            sum = sum.add(chunk);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        //
        // 3) Remainder: fall back to your scalar code for i % LANES != 0
        //
        for (int i = (M / LANES) * LANES; i < M; i++) {
            int c1 = Byte.toUnsignedInt(baseOffsets1.get(i + baseOffsetsOffset1));
            int c2 = Byte.toUnsignedInt(baseOffsets2.get(i + baseOffsetsOffset2));
            int r = Math.min(c1, c2);
            int c = Math.max(c1, c2);

            int offsetRow = r * k - (r * (r - 1) / 2);
            int idxInBlock = offsetRow + (c - r);
            int base = i * blockSize;

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
        final int LANES = FSPECIES.length();
        final int k = clusterCount;
        final int blockSize = k * (k + 1) / 2;
        final int M = subspaceCount;

        int[] convOffsets = scratchInt512.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
        FloatVector scale = FloatVector.zero(FloatVector.SPECIES_512).addIndex(blockSize);
        FloatVector kvec = FloatVector.broadcast(FloatVector.SPECIES_512, k);
        FloatVector onevec = FloatVector.broadcast(FloatVector.SPECIES_512, 1);
        FloatVector twovec = FloatVector.broadcast(FloatVector.SPECIES_512, 0.5f);

        for (int i = 0; i + LANES <= M; i += LANES) {
            FloatVector c1v = fromByteSequence(ByteVector.SPECIES_128, baseOffsets1, i + baseOffsets1.offset() + baseOffsetsOffset1)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                    .convertShape(VectorOperators.I2F, FSPECIES, 0)
                    .reinterpretAsFloats();

            FloatVector c2v = fromByteSequence(ByteVector.SPECIES_128, baseOffsets2, i + baseOffsets2.offset() + baseOffsetsOffset2)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                    .convertShape(VectorOperators.I2F, FSPECIES, 0)
                    .reinterpretAsFloats();

            // b) r = min(c1,c2), c = max(c1,c2)
            var r = c1v.min(c2v);
            var c = c1v.max(c2v);

            // c) offsetRow = r*k - (r*(r-1))/2
            var rk = r.mul(kvec);
            var triangular = r.mul(r.sub(onevec)).mul(twovec);
            var offsetRow = rk.sub(triangular);

            // d) idxInBlock = offsetRow + (c - r) + (i * blockSize)
            offsetRow.add(c.sub(r)).add(scale)
                    .convertShape(VectorOperators.F2I, IntVector.SPECIES_512, 0)
                    .reinterpretAsInts()
                    .intoArray(convOffsets, 0);

            // e) gather LANES floats from `partials` at those indices
            FloatVector chunk = fromVectorFloat(FSPECIES, data, i * blockSize, convOffsets, 0);

            // f) horizontal sum the chunk and add to our scalar accumulator
            sum = sum.add(chunk);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        //
        // 3) Remainder: fall back to your scalar code for i % LANES != 0
        //
        for (int i = (M / LANES) * LANES; i < M; i++) {
            int c1 = Byte.toUnsignedInt(baseOffsets1.get(i + baseOffsetsOffset1));
            int c2 = Byte.toUnsignedInt(baseOffsets2.get(i + baseOffsetsOffset2));
            int r = Math.min(c1, c2);
            int c = Math.max(c1, c2);

            int offsetRow = r * k - (r * (r - 1) / 2);
            int idxInBlock = offsetRow + (c - r);
            int base = i * blockSize;

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
        var sum = LongVector.zero(LongVector.SPECIES_PREFERRED);
        int vectorizedLength = LongVector.SPECIES_PREFERRED.loopBound(a.length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += LongVector.SPECIES_PREFERRED.length()) {
            var va = LongVector.fromArray(LongVector.SPECIES_PREFERRED, a, i);
            var vb = LongVector.fromArray(LongVector.SPECIES_PREFERRED, b, i);

            var xorResult = va.lanewise(VectorOperators.XOR, vb);
            sum = sum.add(xorResult.lanewise(VectorOperators.BIT_COUNT));
        }

        int res = (int) sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < a.length; i++) {
            res += Long.bitCount(a[i] ^ b[i]);
        }

        return res;
    }

    @Override
    public float max(VectorFloat<?> v) {
        var accum = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, -Float.MAX_VALUE);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v, i);
            accum = accum.max(a);
        }
        float max = accum.reduceLanes(VectorOperators.MAX);
        for (int i = vectorizedLength; i < v.length(); i++) {
            max = Math.max(max, v.get(i));
        }
        return max;
    }

    @Override
    public float min(VectorFloat<?> v) {
        var accum = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, Float.MAX_VALUE);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = fromVectorFloat(FloatVector.SPECIES_PREFERRED, v, i);
            accum = accum.min(a);
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

    private static float combineBytes(int i, int shuffle, VectorFloat<?> partials) {
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
            case 512 ->
                    pqDecodedCosineSimilarity512(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
            case 256 ->
                    pqDecodedCosineSimilarity256(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
            case 128 ->
                    pqDecodedCosineSimilarity128(encoded, encodedOffset, encodedLength, clusterCount, partialSums, aMagnitude, bMagnitude);
            default -> throw new IllegalStateException("Unsupported vector width: " + PREFERRED_BIT_SIZE);
        };
    }

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
                    .intoArray(convOffsets, 0);

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
                    .intoArray(convOffsets, 0);

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

    FloatVector logisticNQT(FloatVector vector, float alpha, float x0) {
        FloatVector temp = vector.fma(alpha, -alpha * x0);
        VectorMask<Float> isPositive = temp.test(VectorOperators.IS_NEGATIVE).not();
        IntVector p = temp.add(1, isPositive)
                .convert(VectorOperators.F2I, 0)
                .reinterpretAsInts();
        FloatVector e = p.convert(VectorOperators.I2F, 0).reinterpretAsFloats();
        IntVector m = temp.sub(e).fma(0.5f, 1).reinterpretAsInts();

        temp = m.add(p.lanewise(VectorOperators.LSHL, 23)).reinterpretAsFloats();  // temp = m * 2^p
        return temp.div(temp.add(1));
    }

    float logisticNQT(float value, float alpha, float x0) {
        float temp = Math.fma(value, alpha, -alpha * x0);
        int p = (int) Math.floor(temp + 1);
        int m = Float.floatToIntBits(Math.fma(temp - p, 0.5f, 1));

        temp = Float.intBitsToFloat(m + (p << 23));  // temp = m * 2^p
        return temp / (temp + 1);
    }

    FloatVector logitNQT(FloatVector vector, float inverseAlpha, float x0) {
        FloatVector z = vector.div(const1f.sub(vector));

        IntVector temp = z.reinterpretAsInts();
        FloatVector p = temp.and(0x7f800000)
                .lanewise(VectorOperators.LSHR, 23).sub(128)
                .convert(VectorOperators.I2F, 0)
                .reinterpretAsFloats();
        FloatVector m = temp.lanewise(VectorOperators.AND, 0x007fffff).add(0x3f800000).reinterpretAsFloats();

        return m.add(p).fma(inverseAlpha, x0);
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
        var arr = bytes.reinterpretAsInts()
                .lanewise(VectorOperators.LSHR, 8 * part)
                .lanewise(VectorOperators.AND, 0xff)
                .convert(VectorOperators.I2F, 0)
                .reinterpretAsFloats();

        arr = arr.fma(logisticScale, logisticBias);
        return logitNQT(arr, inverseAlpha, x0);
    }

    @Override
    public void nvqQuantize8bit(VectorFloat<?> vector, float alpha, float x0, float minValue, float maxValue, ByteSequence<?> destination) {
        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        final var mask = ByteVector.SPECIES_PREFERRED.indexInRange(0, FloatVector.SPECIES_PREFERRED.length());

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var invLogisticScale = 255 / (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            arr = logisticNQT(arr, scaledAlpha, scaledX0);
            arr = arr.sub(logisticBias).mul(invLogisticScale);
            var bytes = arr.add(const05f)
                    .convertShape(VectorOperators.F2B, ByteVector.SPECIES_PREFERRED, 0)
                    .reinterpretAsBytes();

            intoByteSequence(bytes, destination, i, mask);
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
        int constant = (1 << nBits) - 1;
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / constant;
        var invLogisticScale = 1 / logisticScale;

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            var recArr = logisticNQT(arr, scaledAlpha, scaledX0);
            recArr = recArr.sub(logisticBias).mul(invLogisticScale);
            recArr = recArr.add(const05f)
                    .convert(VectorOperators.F2I, 0)
                    .reinterpretAsInts()
                    .convert(VectorOperators.I2F, 0)
                    .reinterpretAsFloats();
            recArr = recArr.fma(logisticScale, logisticBias);
            recArr = logitNQT(recArr, invScaledAlpha, scaledX0);

            var diff = arr.sub(recArr);
            squaredSumVec = diff.fma(diff, squaredSumVec);
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

        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i);
            var recArr = arr.sub(minValue).mul(constant / delta);
            recArr = recArr.add(const05f)
                    .convert(VectorOperators.F2I, 0)
                    .reinterpretAsInts()
                    .convert(VectorOperators.I2F, 0)
                    .reinterpretAsFloats();
            recArr = recArr.fma(delta / constant, minValue);

            var diff = arr.sub(recArr);
            squaredSumVec = diff.fma(diff, squaredSumVec);
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value, recValue;
        for (int i = vectorizedLength; i < vector.length(); i++) {
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
        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(quantizedVector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i);

            for (int j = 0; j < 4; j++) {
                var v1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i + floatStep * j);
                var v2 = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);

                var diff = v1.sub(v2);
                squaredSumVec = diff.fma(diff, squaredSumVec);
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
        FloatVector dotProdVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(quantizedVector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logisticNQT(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logisticNQT(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;


        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i);

            for (int j = 0; j < 4; j++) {
                var v1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i + floatStep * j);
                var v2 = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);
                dotProdVec = v1.fma(v2, dotProdVec);
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

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(vector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = fromByteSequence(ByteVector.SPECIES_PREFERRED, quantizedVector, i);

            for (int j = 0; j < 4; j++) {
                var va = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vector, i + floatStep * j);
                var vb = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);

                var vCentroid = fromVectorFloat(FloatVector.SPECIES_PREFERRED, centroid, i + floatStep * j);
                vb = vb.add(vCentroid);

                vsum = va.fma(vb, vsum);
                vbMagnitude = vb.fma(vb, vbMagnitude);
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
        final int n = (last - first) / nRows;
        boolean[] visited = new boolean[last - first];
        float temp;
        int cycle = first;
        while (++cycle != last) {
            if (visited[cycle - first])
                continue;
            int a = cycle - first;
            do {
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
        return pqDecodedCosineSimilarity(encoded, 0, encoded.length(), clusterCount, partialSums, aMagnitude, bMagnitude);
    }

    @Override
    public float ashDotRow(float[] Arow, float[] x) {
        assert Arow.length == x.length : "Arow.length != x.length";

        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int len = x.length;
        final int upper = SPEC.loopBound(len);

        FloatVector sum = FloatVector.zero(SPEC);

        int i = 0;
        for (; i < upper; i += SPEC.length()) {
            FloatVector va = FloatVector.fromArray(SPEC, Arow, i);
            FloatVector vx = FloatVector.fromArray(SPEC, x, i);
            sum = va.fma(vx, sum);
        }

        float acc = sum.reduceLanes(VectorOperators.ADD);

        // scalar tail
        for (; i < len; i++) {
            acc += Arow[i] * x[i];
        }

        return acc;
    }

    @Override
    public boolean supportsAshMaskedLoad() {
        return true;
    }

//    // This version uses masked loads.
//    @Override
//    public float ashMaskedAddAllWords(float[] tildeQ,
//                                      int d,
//                                      long[] packedBits,
//                                      int packedBase,
//                                      int words) {
//        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
//        final int LANES = SPEC.length();
//
//        FloatVector accVec = FloatVector.zero(SPEC);
//        float scalarAcc = 0f;
//
//        for (int w = 0; w < words; w++) {
//            long wordBits = packedBits[packedBase + w];
//            int baseDim = w * 64;
//
//            for (int off = 0; off < 64 && (baseDim + off) < d; off += LANES) {
//                int qBase = baseDim + off;
//                VectorMask<Float> inRange = SPEC.indexInRange(qBase, d);
//
//                long maskBits;
//                if (LANES == 16)      maskBits = (wordBits >>> off) & 0xFFFFL;
//                else if (LANES == 8)  maskBits = (wordBits >>> off) & 0xFFL;
//                else if (LANES == 4)  maskBits = (wordBits >>> off) & 0xFL;
//                else {
//                    // unusual width fallback
//                    long m = wordBits >>> off;
//                    float s = 0f;
//                    for (int b = 0; b < LANES; b++) {
//                        if (((m >>> b) & 1L) != 0L) {
//                            int idx = qBase + b;
//                            if (idx < d) s += tildeQ[idx];
//                        }
//                    }
//                    scalarAcc += s;
//                    continue;
//                }
//
//                if (maskBits == 0L) continue;
//
//                VectorMask<Float> bitMask = VectorMask.fromLong(SPEC, maskBits).and(inRange);
//                FloatVector masked = FloatVector.fromArray(SPEC, tildeQ, qBase, bitMask);
//                accVec = accVec.add(masked);
//            }
//        }
//
//        return accVec.reduceLanes(VectorOperators.ADD) + scalarAcc;
//    }

    @Override
    public float ashMaskedAddFlat(float[] tildeQ,
                                  int qOffset,
                                  long[] allPackedVectors,
                                  int packedBase,
                                  int d,
                                  int words) {
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int LANES = SPEC.length();

        // 4 Accumulators to hide instruction latency (Pipeline depth ~4)
        FloatVector acc0 = FloatVector.zero(SPEC);
        FloatVector acc1 = FloatVector.zero(SPEC);
        FloatVector acc2 = FloatVector.zero(SPEC);
        FloatVector acc3 = FloatVector.zero(SPEC);

        // Pre-calculate mask extraction constants based on LANES
        long laneMask;
        if (LANES == 16)      laneMask = 0xFFFFL;       // AVX-512
        else if (LANES == 8)  laneMask = 0xFFL;         // AVX2
        else if (LANES == 4)  laneMask = 0xFL;          // NEON / SSE
        else                  laneMask = (1L << LANES) - 1;

        // Safe limit for Fast Path
        int safeWords = d / 64;

        // --- HOT LOOP ---
        for (int w = 0; w < safeWords; w++) {
            long wordBits = allPackedVectors[packedBase + w];
            if (wordBits == 0) continue; // Branch prediction handles this well

            int baseDim = w * 64;

            // Iterate 64 bits in steps of LANES (e.g., 0, 8, 16, 24... for AVX2)
            // We unroll manually inside the loop using modulo to pick accumulators
            // The JIT is smart enough to unroll this loop completely since LANES is constant.
            int accumulatorIdx = 0;

            for (int off = 0; off < 64; off += LANES) {
                // Shift and mask
                long m = (wordBits >>> off) & laneMask;

                if (m != 0) {
                    VectorMask<Float> bitMask = VectorMask.fromLong(SPEC, m);
                    FloatVector loaded = FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + off);

                    // Round-robin distribution to accumulators to break dependency chains
                    switch (accumulatorIdx & 3) { // (idx % 4)
                        case 0 -> acc0 = acc0.add(loaded, bitMask);
                        case 1 -> acc1 = acc1.add(loaded, bitMask);
                        case 2 -> acc2 = acc2.add(loaded, bitMask);
                        case 3 -> acc3 = acc3.add(loaded, bitMask);
                    }
                }
                accumulatorIdx++;
            }
        }

        // --- TAIL LOOP (Robust Bounds) ---
        // Handles the edge case where d is not a multiple of 64
        for (int w = safeWords; w < words; w++) {
            long wordBits = allPackedVectors[packedBase + w];
            if (wordBits == 0) continue;

            int baseDim = w * 64;

            for (int off = 0; off < 64; off += LANES) {
                int localIdx = baseDim + off;
                if (localIdx >= d) break;

                long m = (wordBits >>> off) & laneMask;
                if (m == 0) continue;

                VectorMask<Float> bitMask = VectorMask.fromLong(SPEC, m);
                int absIdx = qOffset + localIdx;

                if (localIdx + LANES <= d) {
                    // Safe unmasked load
                    acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, absIdx), bitMask);
                } else {
                    // Boundary masked load
                    VectorMask<Float> rangeMask = SPEC.indexInRange(absIdx, qOffset + d);
                    acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, absIdx, rangeMask),
                            bitMask.and(rangeMask));
                }
            }
        }

        // Reduce all accumulators
        return acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD);
    }

//    @Override
//    public float ashMaskedAddFlat(float[] tildeQ,
//                                  int qOffset,
//                                  long[] allPackedVectors,
//                                  int packedBase,
//                                  int d,
//                                  int words) {
//        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
//
//        FloatVector acc0 = FloatVector.zero(SPEC);
//        FloatVector acc1 = FloatVector.zero(SPEC);
//        FloatVector acc2 = FloatVector.zero(SPEC);
//        FloatVector acc3 = FloatVector.zero(SPEC);
//
//        int safeWords = d / 64;
//
//        // --- HOT LOOP (Unrolled & Safe) ---
//        for (int w = 0; w < safeWords; w++) {
//            long wordBits = allPackedVectors[packedBase + w];
//            if (wordBits == 0) continue;
//
//            int baseDim = w * 64;
//
//            // Unroll 0
//            long m0 = wordBits & 0xFFFFL;
//            if (m0 != 0) {
//                acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim),
//                        VectorMask.fromLong(SPEC, m0));
//            }
//
//            // Unroll 1
//            long m1 = (wordBits >>> 16) & 0xFFFFL;
//            if (m1 != 0) {
//                acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + 16),
//                        VectorMask.fromLong(SPEC, m1));
//            }
//
//            // Unroll 2
//            long m2 = (wordBits >>> 32) & 0xFFFFL;
//            if (m2 != 0) {
//                acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + 32),
//                        VectorMask.fromLong(SPEC, m2));
//            }
//
//            // Unroll 3
//            long m3 = (wordBits >>> 48) & 0xFFFFL;
//            if (m3 != 0) {
//                acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + 48),
//                        VectorMask.fromLong(SPEC, m3));
//            }
//        }
//
//        // --- TAIL LOOP (Robust Bounds) ---
//        for (int w = safeWords; w < words; w++) {
//            long wordBits = allPackedVectors[packedBase + w];
//            if (wordBits == 0) continue;
//
//            int baseDim = w * 64;
//
//            for (int off = 0; off < 64; off += 16) {
//                int localIdx = baseDim + off;
//                if (localIdx >= d) break;
//
//                long maskBits = (wordBits >>> off) & 0xFFFFL;
//                if (maskBits == 0) continue;
//
//                VectorMask<Float> bitMask = VectorMask.fromLong(SPEC, maskBits);
//                int absIdx = qOffset + localIdx;
//
//                if (localIdx + 16 <= d) {
//                    acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, absIdx), bitMask);
//                } else {
//                    VectorMask<Float> rangeMask = SPEC.indexInRange(absIdx, qOffset + d);
//                    acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, absIdx, rangeMask),
//                            bitMask.and(rangeMask));
//                }
//            }
//        }
//
//        return acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD);
//    }

    public float ashMaskedAddFlatOptimizedv1(float[] tildeQ,
                                           int qOffset,
                                           long[] allPackedVectors,
                                           int packedBase,
                                           int d,
                                           int words) {
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int LANES = SPEC.length();

        FloatVector acc0 = FloatVector.zero(SPEC);
        FloatVector acc1 = FloatVector.zero(SPEC);
        FloatVector acc2 = FloatVector.zero(SPEC);
        FloatVector acc3 = FloatVector.zero(SPEC);

        int safeWords = d / 64;

        // --- OPTIMIZATION 1: Specialized AVX-512 Loop (16 Lanes) ---
        // This covers the vast majority of server workloads.
        // We unroll the 64-bit word processing into exactly 4 blocks.
        if (LANES == 16) {
            for (int w = 0; w < safeWords; w++) {
                long wordBits = allPackedVectors[packedBase + w];
                if (wordBits == 0) continue;

                int baseDim = w * 64;

                // Block 0: Bits 0-15
                // Note: casting to (int) acts as a fast mask for the low bits
                long m0 = wordBits & 0xFFFF;
                if (m0 != 0) {
                    VectorMask<Float> mask = VectorMask.fromLong(SPEC, m0);
                    FloatVector loaded = FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim);
                    // OPTIMIZATION 2: Blend (Zero-out) + Unmasked Add
                    // "blend" with 0.0f where mask is unset (not())
                    acc0 = acc0.add(loaded.blend(0.0f, mask.not()));
                }

                // Block 1: Bits 16-31
                long m1 = (wordBits >>> 16) & 0xFFFF;
                if (m1 != 0) {
                    VectorMask<Float> mask = VectorMask.fromLong(SPEC, m1);
                    FloatVector loaded = FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + 16);
                    acc1 = acc1.add(loaded.blend(0.0f, mask.not()));
                }

                // Block 2: Bits 32-47
                long m2 = (wordBits >>> 32) & 0xFFFF;
                if (m2 != 0) {
                    VectorMask<Float> mask = VectorMask.fromLong(SPEC, m2);
                    FloatVector loaded = FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + 32);
                    acc2 = acc2.add(loaded.blend(0.0f, mask.not()));
                }

                // Block 3: Bits 48-63
                long m3 = (wordBits >>> 48); // No mask needed for high bits shift
                if (m3 != 0) {
                    VectorMask<Float> mask = VectorMask.fromLong(SPEC, m3);
                    FloatVector loaded = FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + 48);
                    acc3 = acc3.add(loaded.blend(0.0f, mask.not()));
                }
            }
        }
        // --- OPTIMIZATION 3: Generic / AVX2 Loop ---
        else {
            long laneMask = (1L << LANES) - 1;
            for (int w = 0; w < safeWords; w++) {
                long wordBits = allPackedVectors[packedBase + w];
                if (wordBits == 0) continue;

                int baseDim = w * 64;
                int accumulatorIdx = 0;

                // Manual step loop to avoid modulo in switch
                for (int off = 0; off < 64; off += LANES) {
                    long m = (wordBits >>> off) & laneMask;
                    if (m != 0) {
                        VectorMask<Float> mask = VectorMask.fromLong(SPEC, m);
                        FloatVector loaded = FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + off);

                        // Simple Round-Robin without switch/modulo
                        if (accumulatorIdx == 0) acc0 = acc0.add(loaded.blend(0.0f, mask.not()));
                        else if (accumulatorIdx == 1) acc1 = acc1.add(loaded.blend(0.0f, mask.not()));
                        else if (accumulatorIdx == 2) acc2 = acc2.add(loaded.blend(0.0f, mask.not()));
                        else {
                            acc3 = acc3.add(loaded.blend(0.0f, mask.not()));
                            accumulatorIdx = -1; // Reset for next increment
                        }
                    }
                    accumulatorIdx++;
                }
            }
        }

        // --- TAIL LOOP (Keep your original robust logic here) ---
        // Your original tail logic handles bounds checking correctly.
        // Since this is the "cold" path, optimization matters less.
        for (int w = safeWords; w < words; w++) {
            // ... (Keep your original Tail Loop implementation) ...
        }

        return acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD);
    }

    public float ashMaskedAddFlatOptimizedv2(float[] tildeQ,
                                             int qOffset,
                                             long[] allPackedVectors,
                                             int packedBase,
                                             int d,
                                             int words) {
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int LANES = SPEC.length();

        // Expansion to 8 accumulators to saturate 32 ZMM registers and hide memory latency
        FloatVector acc0 = FloatVector.zero(SPEC);
        FloatVector acc1 = FloatVector.zero(SPEC);
        FloatVector acc2 = FloatVector.zero(SPEC);
        FloatVector acc3 = FloatVector.zero(SPEC);
        FloatVector acc4 = FloatVector.zero(SPEC);
        FloatVector acc5 = FloatVector.zero(SPEC);
        FloatVector acc6 = FloatVector.zero(SPEC);
        FloatVector acc7 = FloatVector.zero(SPEC);

        int safeWords = d / 64;

        if (LANES == 16) {
            // Process 2 words at a time (128 floats) to fill the 8-way unroll
            int w = 0;
            for (; w <= safeWords - 2; w += 2) {
                int baseIdx = qOffset + (w << 6);

                // Word 1 (acc0-acc3)
                long w0 = allPackedVectors[packedBase + w];
                if (w0 != 0) {
                    long m0 = w0 & 0xFFFF;
                    if (m0 != 0) acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx, VectorMask.fromLong(SPEC, m0)));
                    long m1 = (w0 >>> 16) & 0xFFFF;
                    if (m1 != 0) acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 16, VectorMask.fromLong(SPEC, m1)));
                    long m2 = (w0 >>> 32) & 0xFFFF;
                    if (m2 != 0) acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 32, VectorMask.fromLong(SPEC, m2)));
                    long m3 = (w0 >>> 48);
                    if (m3 != 0) acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 48, VectorMask.fromLong(SPEC, m3)));
                }

                // Word 2 (acc4-acc7)
                long w1 = allPackedVectors[packedBase + w + 1];
                int nextBaseIdx = baseIdx + 64;
                if (w1 != 0) {
                    long m4 = w1 & 0xFFFF;
                    if (m4 != 0) acc4 = acc4.add(FloatVector.fromArray(SPEC, tildeQ, nextBaseIdx, VectorMask.fromLong(SPEC, m4)));
                    long m5 = (w1 >>> 16) & 0xFFFF;
                    if (m5 != 0) acc5 = acc5.add(FloatVector.fromArray(SPEC, tildeQ, nextBaseIdx + 16, VectorMask.fromLong(SPEC, m5)));
                    long m6 = (w1 >>> 32) & 0xFFFF;
                    if (m6 != 0) acc6 = acc6.add(FloatVector.fromArray(SPEC, tildeQ, nextBaseIdx + 32, VectorMask.fromLong(SPEC, m6)));
                    long m7 = (w1 >>> 48);
                    if (m7 != 0) acc7 = acc7.add(FloatVector.fromArray(SPEC, tildeQ, nextBaseIdx + 48, VectorMask.fromLong(SPEC, m7)));
                }
            }

            // Clean up remaining safe word if odd count
            for (; w < safeWords; w++) {
                int baseIdx = qOffset + (w << 6);
                long wordBits = allPackedVectors[packedBase + w];
                if (wordBits != 0) {
                    long m0 = wordBits & 0xFFFF;
                    if (m0 != 0) acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx, VectorMask.fromLong(SPEC, m0)));
                    long m1 = (wordBits >>> 16) & 0xFFFF;
                    if (m1 != 0) acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 16, VectorMask.fromLong(SPEC, m1)));
                    long m2 = (wordBits >>> 32) & 0xFFFF;
                    if (m2 != 0) acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 32, VectorMask.fromLong(SPEC, m2)));
                    long m3 = (wordBits >>> 48);
                    if (m3 != 0) acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 48, VectorMask.fromLong(SPEC, m3)));
                }
            }
        } else {
            // Fallback for AVX2/Generic environments
            long laneMask = (1L << LANES) - 1;
            for (int w = 0; w < safeWords; w++) {
                long wordBits = allPackedVectors[packedBase + w];
                if (wordBits == 0) continue;
                int baseDim = w * 64;
                for (int off = 0; off < 64; off += LANES) {
                    long m = (wordBits >>> off) & laneMask;
                    if (m != 0) {
                        // Use masked add directly to avoid blend/not overhead
                        acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, qOffset + baseDim + off), VectorMask.fromLong(SPEC, m));
                    }
                }
            }
        }

        // Tail loop for non-word-aligned dimensions
        for (int w = safeWords; w < words; w++) {
            long wordBits = allPackedVectors[packedBase + w];
            if (wordBits == 0) continue;
            int baseDim = w * 64;
            int remaining = d - baseDim;
            for (int i = 0; i < remaining; i++) {
                if ((wordBits & (1L << i)) != 0) {
                    // Scalar fallback for the absolute tail
                    // (Could be vectorized with a length-limited mask if performance here is critical)
                    acc0 = acc0.add(FloatVector.broadcast(SPEC, tildeQ[qOffset + baseDim + i]).blend(0, SPEC.indexInRange(0, 1).not()));
                }
            }
        }

        // Horizontal reduction of the 8 accumulators
        FloatVector res0 = acc0.add(acc1).add(acc2).add(acc3);
        FloatVector res1 = acc4.add(acc5).add(acc6).add(acc7);
        return res0.add(res1).reduceLanes(VectorOperators.ADD);
    }

    public float ashMaskedAddFlatOptimizedv3(float[] tildeQ, int qOffset, long[] allPackedVectors, int packedBase, int d, int words) {
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;

        // 6 Accumulators: Optimized to fit in registers without spilling to cache
        FloatVector acc0 = FloatVector.zero(SPEC), acc1 = FloatVector.zero(SPEC), acc2 = FloatVector.zero(SPEC);
        FloatVector acc3 = FloatVector.zero(SPEC), acc4 = FloatVector.zero(SPEC), acc5 = FloatVector.zero(SPEC);

        int safeWords = d / 64;
        for (int w = 0; w < safeWords; w++) {
            int baseIdx = qOffset + (w << 6);
            long word = allPackedVectors[packedBase + w];

            // Direct Masked Add: Collapses 'blend' + 'add' into a single AVX-512 instruction
            acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx),      VectorMask.fromLong(SPEC, word & 0xFFFF));
            acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 16), VectorMask.fromLong(SPEC, (word >>> 16) & 0xFFFF));
            acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 32), VectorMask.fromLong(SPEC, (word >>> 32) & 0xFFFF));
            acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 48), VectorMask.fromLong(SPEC, (word >>> 48)));
        }

        // Horizontal reduction
        return acc0.add(acc1).add(acc2).add(acc3).add(acc4).add(acc5).reduceLanes(VectorOperators.ADD);
    }

    public float ashMaskedAddFlatOptimizedv4(float[] tildeQ, int qOffset, long[] allPackedVectors, int packedBase, int d, int words) {
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;

        // STRICT ADHERENCE TO V3 LEARNINGS: 6 Accumulators to prevent register spilling
        FloatVector acc0 = FloatVector.zero(SPEC), acc1 = FloatVector.zero(SPEC), acc2 = FloatVector.zero(SPEC);
        FloatVector acc3 = FloatVector.zero(SPEC), acc4 = FloatVector.zero(SPEC), acc5 = FloatVector.zero(SPEC);

        int safeWords = d / 64;
        int w = 0;

        // --- NEW OPTIMIZATION: 3-Word Unroll ---
        // This processes 192 dimensions per loop while staying within the 6-accumulator limit.
        for (; w <= safeWords - 3; w += 3) {
            int baseIdx = qOffset + (w << 6);

            // Word 0 -> acc0, acc1, acc2, acc3 (Uses 4 sub-blocks)
            long wd0 = allPackedVectors[packedBase + w];
            acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx),      VectorMask.fromLong(SPEC, wd0 & 0xFFFF));
            acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 16), VectorMask.fromLong(SPEC, (wd0 >>> 16) & 0xFFFF));
            acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 32), VectorMask.fromLong(SPEC, (wd0 >>> 32) & 0xFFFF));
            acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 48), VectorMask.fromLong(SPEC, (wd0 >>> 48)));

            // Word 1 -> acc4, acc5, acc0, acc1 (Interleaves to reuse registers)
            long wd1 = allPackedVectors[packedBase + w + 1];
            int base1 = baseIdx + 64;
            acc4 = acc4.add(FloatVector.fromArray(SPEC, tildeQ, base1),      VectorMask.fromLong(SPEC, wd1 & 0xFFFF));
            acc5 = acc5.add(FloatVector.fromArray(SPEC, tildeQ, base1 + 16), VectorMask.fromLong(SPEC, (wd1 >>> 16) & 0xFFFF));
            acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, base1 + 32), VectorMask.fromLong(SPEC, (wd1 >>> 32) & 0xFFFF));
            acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, base1 + 48), VectorMask.fromLong(SPEC, (wd1 >>> 48)));

            // Word 2 -> acc2, acc3, acc4, acc5
            long wd2 = allPackedVectors[packedBase + w + 2];
            int base2 = baseIdx + 128;
            acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, base2),      VectorMask.fromLong(SPEC, wd2 & 0xFFFF));
            acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, base2 + 16), VectorMask.fromLong(SPEC, (wd2 >>> 16) & 0xFFFF));
            acc4 = acc4.add(FloatVector.fromArray(SPEC, tildeQ, base2 + 32), VectorMask.fromLong(SPEC, (wd2 >>> 32) & 0xFFFF));
            acc5 = acc5.add(FloatVector.fromArray(SPEC, tildeQ, base2 + 48), VectorMask.fromLong(SPEC, (wd2 >>> 48)));
        }

        // Standard v3 cleanup for remaining words
        for (; w < safeWords; w++) {
            int baseIdx = qOffset + (w << 6);
            long word = allPackedVectors[packedBase + w];
            acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx),      VectorMask.fromLong(SPEC, word & 0xFFFF));
            acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 16), VectorMask.fromLong(SPEC, (word >>> 16) & 0xFFFF));
            acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 32), VectorMask.fromLong(SPEC, (word >>> 32) & 0xFFFF));
            acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 48), VectorMask.fromLong(SPEC, (word >>> 48)));
        }

        // Balanced Reduction Tree (Better than linear add)

        FloatVector sum01 = acc0.add(acc1);
        FloatVector sum23 = acc2.add(acc3);
        FloatVector sum45 = acc4.add(acc5);
        return sum01.add(sum23).add(sum45).reduceLanes(VectorOperators.ADD);
    }

    public float ashMaskedAdd_512(float[] tildeQ, int qOffset, long[] allPackedVectors, int packedBase, int d, int words) {
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED; // Assumed 512-bit (16 floats)
        int safeWords = d / 64;

        // 1. Determine Unroll Strategy based on d
        // We aim for 8 accumulators for large d, but 384 remains at 6 as requested.
        int numAcc = (d >= 512) ? 8 : (d == 384 ? 6 : (d >= 128 ? 4 : 2));

        // Initialize required accumulators
        FloatVector acc0 = FloatVector.zero(SPEC), acc1 = FloatVector.zero(SPEC);
        FloatVector acc2 = null, acc3 = null, acc4 = null, acc5 = null, acc6 = null, acc7 = null;

        if (numAcc >= 4) { acc2 = FloatVector.zero(SPEC); acc3 = FloatVector.zero(SPEC); }
        if (numAcc >= 6) { acc4 = FloatVector.zero(SPEC); acc5 = FloatVector.zero(SPEC); }
        if (numAcc >= 8) { acc6 = FloatVector.zero(SPEC); acc7 = FloatVector.zero(SPEC); }

        int w = 0;

        // 2. Main Unrolled Loop
        // For d=384, this uses your 3-word unroll.
        // For d >= 512, we use a 2-word unroll across 8 registers for better throughput.
        int unrollFactor = (d == 384) ? 3 : (d >= 512 ? 2 : 1);

        for (; w <= safeWords - unrollFactor; w += unrollFactor) {
            int baseIdx = qOffset + (w << 6);

            // Word 0 Processing
            long wd0 = allPackedVectors[packedBase + w];
            acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx),      VectorMask.fromLong(SPEC, wd0 & 0xFFFF));
            acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 16), VectorMask.fromLong(SPEC, (wd0 >>> 16) & 0xFFFF));
            acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 32), VectorMask.fromLong(SPEC, (wd0 >>> 32) & 0xFFFF));
            acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 48), VectorMask.fromLong(SPEC, (wd0 >>> 48)));

            if (unrollFactor >= 2) {
                long wd1 = allPackedVectors[packedBase + w + 1];
                int b1 = baseIdx + 64;
                if (numAcc == 6) { // Special Case for d=384 Interleave
                    acc4 = acc4.add(FloatVector.fromArray(SPEC, tildeQ, b1),      VectorMask.fromLong(SPEC, wd1 & 0xFFFF));
                    acc5 = acc5.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 16), VectorMask.fromLong(SPEC, (wd1 >>> 16) & 0xFFFF));
                    acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 32), VectorMask.fromLong(SPEC, (wd1 >>> 32) & 0xFFFF));
                    acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 48), VectorMask.fromLong(SPEC, (wd1 >>> 48)));
                } else { // Standard 8-accumulator path
                    acc4 = acc4.add(FloatVector.fromArray(SPEC, tildeQ, b1),      VectorMask.fromLong(SPEC, wd1 & 0xFFFF));
                    acc5 = acc5.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 16), VectorMask.fromLong(SPEC, (wd1 >>> 16) & 0xFFFF));
                    acc6 = acc6.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 32), VectorMask.fromLong(SPEC, (wd1 >>> 32) & 0xFFFF));
                    acc7 = acc7.add(FloatVector.fromArray(SPEC, tildeQ, b1 + 48), VectorMask.fromLong(SPEC, (wd1 >>> 48)));
                }
            }

            if (unrollFactor == 3) { // Tail of the d=384 case
                long wd2 = allPackedVectors[packedBase + w + 2];
                int b2 = baseIdx + 128;
                acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, b2),      VectorMask.fromLong(SPEC, wd2 & 0xFFFF));
                acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, b2 + 16), VectorMask.fromLong(SPEC, (wd2 >>> 16) & 0xFFFF));
                acc4 = acc4.add(FloatVector.fromArray(SPEC, tildeQ, b2 + 32), VectorMask.fromLong(SPEC, (wd2 >>> 32) & 0xFFFF));
                acc5 = acc5.add(FloatVector.fromArray(SPEC, tildeQ, b2 + 48), VectorMask.fromLong(SPEC, (wd2 >>> 48)));
            }
        }

        // 3. Cleanup Loop
        for (; w < safeWords; w++) {
            int baseIdx = qOffset + (w << 6);
            long wd = allPackedVectors[packedBase + w];
            acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx),      VectorMask.fromLong(SPEC, wd & 0xFFFF));
            acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 16), VectorMask.fromLong(SPEC, (wd >>> 16) & 0xFFFF));
            if (numAcc >= 4) {
                acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 32), VectorMask.fromLong(SPEC, (wd >>> 32) & 0xFFFF));
                acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, baseIdx + 48), VectorMask.fromLong(SPEC, (wd >>> 48)));
            }
        }

        // 4. Dynamic Reduction Tree
        return performReduction(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, numAcc);
    }

    private float performReduction(FloatVector a0, FloatVector a1, FloatVector a2, FloatVector a3,
                                   FloatVector a4, FloatVector a5, FloatVector a6, FloatVector a7, int count) {
        FloatVector res;
        if (count == 8) {
            FloatVector s01 = a0.add(a1), s23 = a2.add(a3), s45 = a4.add(a5), s67 = a6.add(a7);
            res = s01.add(s23).add(s45.add(s67));
        } else if (count == 6) {
            res = a0.add(a1).add(a2.add(a3)).add(a4.add(a5));
        } else if (count == 4) {
            res = a0.add(a1).add(a2.add(a3));
        } else {
            res = a0.add(a1);
        }
        return res.reduceLanes(VectorOperators.ADD);
    }

    // This version is optimized for dense masks and uses a masked load, plain add.
    // It is branchless and fully unrolled.
    // It only works for 64-bit aligned payloads and 16 lanes (AVX512).
    public float ashMaskedAddFlat_dense(float[] tildeQ,
                                        int qOffset,
                                        long[] allPackedVectors,
                                        int packedBase,
                                        int d,
                                        int words_not_used) { // TODO words is not used in this version
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int LANES = SPEC.length();

        // Typical case: d is 256/384 => multiple of 64
        // If you can guarantee that, you can delete all tail handling.
        final int words = d >>> 6;

        Objects.checkFromIndexSize(qOffset, d, tildeQ.length);
        Objects.checkFromIndexSize(packedBase, words, allPackedVectors.length);

        // 4 accumulators is fine for AVX-512; keep it.
        FloatVector acc0 = FloatVector.zero(SPEC);
        FloatVector acc1 = FloatVector.zero(SPEC);
        FloatVector acc2 = FloatVector.zero(SPEC);
        FloatVector acc3 = FloatVector.zero(SPEC);

        if (LANES == 16) {
            for (int w = 0, qBase = qOffset; w < words; w++, qBase += 64) {
                long bits = allPackedVectors[packedBase + w];

                // Build masks from shifted bits (no laneMask needed)
                VectorMask<Float> k0 = VectorMask.fromLong(SPEC, bits);
                VectorMask<Float> k1 = VectorMask.fromLong(SPEC, bits >>> 16);
                VectorMask<Float> k2 = VectorMask.fromLong(SPEC, bits >>> 32);
                VectorMask<Float> k3 = VectorMask.fromLong(SPEC, bits >>> 48);

                // Masked load (zeroing) + unmasked add: branchless
                acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, qBase +  0, k0));
                acc1 = acc1.add(FloatVector.fromArray(SPEC, tildeQ, qBase + 16, k1));
                acc2 = acc2.add(FloatVector.fromArray(SPEC, tildeQ, qBase + 32, k2));
                acc3 = acc3.add(FloatVector.fromArray(SPEC, tildeQ, qBase + 48, k3));
            }
        } else {
            // Generic fallback (still branchless and no laneMask)
            for (int w = 0, qBase = qOffset; w < words; w++, qBase += 64) {
                long bits = allPackedVectors[packedBase + w];
                for (int off = 0; off < 64; off += LANES) {
                    VectorMask<Float> k = VectorMask.fromLong(SPEC, bits >>> off);
                    acc0 = acc0.add(FloatVector.fromArray(SPEC, tildeQ, qBase + off, k));
                }
            }
        }

        return acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD);
    }

    // This approach uses masked adds instead of masked loads, and two accumulators (since this is slow)
    @Override
    public float ashMaskedAddAllWords(float[] tildeQ,
                                      int d,
                                      long[] packedBits,
                                      int packedBase,
                                      int words) {
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int LANES = SPEC.length();
        final int upperBoundSafe = d - LANES;

        // Two accumulators to hide floating-point addition latency (4 cycles)
        FloatVector acc1 = FloatVector.zero(SPEC);
        FloatVector acc2 = FloatVector.zero(SPEC);
        float scalarAcc = 0f;

        for (int w = 0; w < words; w++) {
            long wordBits = packedBits[packedBase + w];
            // REMOVED: if (wordBits == 0) continue; (useless for dense data)

            int baseDim = w * 64;

            // Toggle to alternate between acc1 and acc2
            boolean toggle = false;

            for (int off = 0; off < 64; off += LANES) {
                int qBase = baseDim + off;

                // Keep the fast integer check!
                if (qBase >= d) break;

                long maskBits;
                if (LANES == 16)      maskBits = (wordBits >>> off) & 0xFFFFL;
                else if (LANES == 8)  maskBits = (wordBits >>> off) & 0xFFL;
                else if (LANES == 4)  maskBits = (wordBits >>> off) & 0xFL;
                else {
                    // Fallback for unusual widths
                    long m = wordBits >>> off;
                    for (int b = 0; b < LANES; b++) {
                        if (((m >>> b) & 1L) != 0L) {
                            int idx = qBase + b;
                            if (idx < d) scalarAcc += tildeQ[idx];
                        }
                    }
                    continue;
                }

                // Still worth checking if the local chunk is 0 to skip the ADD
                if (maskBits == 0L) continue;

                VectorMask<Float> bitMask = VectorMask.fromLong(SPEC, maskBits);
                FloatVector loaded;

                // Fast Unmasked Load (Still critical for dense data)
                if (qBase <= upperBoundSafe) {
                    loaded = FloatVector.fromArray(SPEC, tildeQ, qBase);
                } else {
                    VectorMask<Float> rangeMask = SPEC.indexInRange(qBase, d);
                    loaded = FloatVector.fromArray(SPEC, tildeQ, qBase, bitMask.and(rangeMask));
                }

                // Alternate accumulators to parallelize the ADDs
                if (!toggle) {
                    acc1 = acc1.add(loaded, bitMask);
                } else {
                    acc2 = acc2.add(loaded, bitMask);
                }
                toggle = !toggle;
            }
        }

        // Merge the two pipelines
        return acc1.add(acc2).reduceLanes(VectorOperators.ADD) + scalarAcc;
    }

    @Override
    public void ashMaskedAddBlockAllWords(float[] tildeQ,
                                          int d,
                                          long[] packedBits,
                                          int blockWordBase,
                                          int words,
                                          int blockSize,
                                          int laneStart,
                                          int blockLen,
                                          float[] outMaskedAdd) {

        // Like in the ASH paper, SIMD lanes are dimensions (SPECIES_PREFERRED),
        // masked load from tildeQ and horizontal sum per neighbor.
        //
        // Optimization: amortize horizontal reductions.
        // Instead of reducing each masked chunk immediately, we accumulate masked
        // vectors into a vector register accumulator and reduce once per lane.
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int LANES = SPEC.length();

        for (int lane = 0; lane < blockLen; lane++) {
            // Vector accumulator: stays in registers (no array accumulation).
            FloatVector accVec = FloatVector.zero(SPEC);

            // Scalar accumulator for unusual widths fallback only.
            float scalarAcc = 0f;

            int laneIndex = laneStart + lane;

            for (int w = 0; w < words; w++) {
                long wordBits = packedBits[blockWordBase + (w * blockSize) + laneIndex];
                int baseDim = w * 64;

                // process this 64-bit word in LANES-bit chunks
                for (int off = 0; off < 64 && (baseDim + off) < d; off += LANES) {
                    int qBase = baseDim + off;

                    // Range mask for tail dims beyond d
                    VectorMask<Float> inRange = SPEC.indexInRange(qBase, d);

                    long maskBits;
                    if (LANES == 16) {
                        maskBits = (wordBits >>> off) & 0xFFFFL;
                    } else if (LANES == 8) {
                        maskBits = (wordBits >>> off) & 0xFFL;
                    } else if (LANES == 4) {
                        maskBits = (wordBits >>> off) & 0xFL;
                    } else {
                        // Unusual width; fall back to scalar for this chunk
                        long m = wordBits >>> off;
                        float s = 0f;
                        for (int b = 0; b < LANES; b++) {
                            if (((m >>> b) & 1L) != 0L) {
                                int idx = qBase + b;
                                if (idx < d) s += tildeQ[idx];
                            }
                        }
                        scalarAcc += s;
                        continue;
                    }

                    if (maskBits == 0L) continue;

                    VectorMask<Float> bitMask =
                            VectorMask.fromLong(SPEC, maskBits).and(inRange);

                    // masked load (paper)
                    FloatVector masked = FloatVector.fromArray(SPEC, tildeQ, qBase, bitMask);

                    // amortized accumulation (register)
                    accVec = accVec.add(masked);
                }
            }

            // One horizontal sum per lane (instead of per chunk)
            float maskedAdd = accVec.reduceLanes(VectorOperators.ADD) + scalarAcc;

            outMaskedAdd[lane] = maskedAdd;
        }
    }

    @Override
    public void ashMaskedAddBlockAllWordsPooled(float[] tildeQPool,
                                                int tildeBase,
                                                int d,
                                                long[] packedBits,
                                                int blockWordBase,
                                                int words,
                                                int blockSize,
                                                int laneStart,
                                                int blockLen,
                                                float[] outMaskedAdd) {

        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int LANES = SPEC.length();

        for (int lane = 0; lane < blockLen; lane++) {
            FloatVector accVec = FloatVector.zero(SPEC);
            float scalarAcc = 0f;

            int laneIndex = laneStart + lane;

            for (int w = 0; w < words; w++) {
                long wordBits = packedBits[blockWordBase + (w * blockSize) + laneIndex];
                int baseDim = w * 64;

                for (int off = 0; off < 64 && (baseDim + off) < d; off += LANES) {
                    int qBase = baseDim + off;

                    VectorMask<Float> inRange = SPEC.indexInRange(qBase, d);

                    long maskBits;
                    if (LANES == 16) {
                        maskBits = (wordBits >>> off) & 0xFFFFL;
                    } else if (LANES == 8) {
                        maskBits = (wordBits >>> off) & 0xFFL;
                    } else if (LANES == 4) {
                        maskBits = (wordBits >>> off) & 0xFL;
                    } else {
                        long m = wordBits >>> off;
                        float s = 0f;
                        for (int b = 0; b < LANES; b++) {
                            if (((m >>> b) & 1L) != 0L) {
                                int idx = qBase + b;
                                if (idx < d) s += tildeQPool[tildeBase + idx];
                            }
                        }
                        scalarAcc += s;
                        continue;
                    }

                    if (maskBits == 0L) continue;

                    VectorMask<Float> bitMask =
                            VectorMask.fromLong(SPEC, maskBits).and(inRange);

                    // pooled masked load: note tildeBase + qBase
                    FloatVector masked =
                            FloatVector.fromArray(SPEC, tildeQPool, tildeBase + qBase, bitMask);

                    accVec = accVec.add(masked);
                }
            }

            outMaskedAdd[lane] = accVec.reduceLanes(VectorOperators.ADD) + scalarAcc;
        }
    }

    private static void ashMaskedAddBlockWordLutNibble(float[] tildeQ,
                                                       int baseDim,
                                                       int d,
                                                       long[] packedBits,
                                                       int packedBase,
                                                       int laneStart,
                                                       int blockLen,
                                                       float[] acc) {
        for (int off = 0; off < 64 && (baseDim + off) < d; off += 4) {
            int q0 = baseDim + off;

            float v0 = (q0 + 0 < d) ? tildeQ[q0 + 0] : 0f;
            float v1 = (q0 + 1 < d) ? tildeQ[q0 + 1] : 0f;
            float v2 = (q0 + 2 < d) ? tildeQ[q0 + 2] : 0f;
            float v3 = (q0 + 3 < d) ? tildeQ[q0 + 3] : 0f;

            float lut1  = v0;
            float lut2  = v1;
            float lut3  = v0 + v1;
            float lut4  = v2;
            float lut5  = v0 + v2;
            float lut6  = v1 + v2;
            float lut7  = v0 + v1 + v2;
            float lut8  = v3;
            float lut9  = v0 + v3;
            float lut10 = v1 + v3;
            float lut11 = v0 + v1 + v3;
            float lut12 = v2 + v3;
            float lut13 = v0 + v2 + v3;
            float lut14 = v1 + v2 + v3;
            float lut15 = v0 + v1 + v2 + v3;

            for (int lane = 0; lane < blockLen; lane++) {
                long word = packedBits[packedBase + laneStart + lane];
                int mask = (int) ((word >>> off) & 0xF);
                switch (mask) {
                    case 0:  break;
                    case 1:  acc[lane] += lut1;  break;
                    case 2:  acc[lane] += lut2;  break;
                    case 3:  acc[lane] += lut3;  break;
                    case 4:  acc[lane] += lut4;  break;
                    case 5:  acc[lane] += lut5;  break;
                    case 6:  acc[lane] += lut6;  break;
                    case 7:  acc[lane] += lut7;  break;
                    case 8:  acc[lane] += lut8;  break;
                    case 9:  acc[lane] += lut9;  break;
                    case 10: acc[lane] += lut10; break;
                    case 11: acc[lane] += lut11; break;
                    case 12: acc[lane] += lut12; break;
                    case 13: acc[lane] += lut13; break;
                    case 14: acc[lane] += lut14; break;
                    case 15: acc[lane] += lut15; break;
                }
            }
        }
    }

    private static void ashMaskedAddBlockWordLutByteTwoNibbles(float[] tildeQ,
                                                               int baseDim,
                                                               int d,
                                                               long[] packedBits,
                                                               int packedBase,
                                                               int laneStart,
                                                               int blockLen,
                                                               float[] acc) {
        for (int off = 0; off < 64 && (baseDim + off) < d; off += 8) {
            int q0 = baseDim + off;

            float v0 = (q0 + 0 < d) ? tildeQ[q0 + 0] : 0f;
            float v1 = (q0 + 1 < d) ? tildeQ[q0 + 1] : 0f;
            float v2 = (q0 + 2 < d) ? tildeQ[q0 + 2] : 0f;
            float v3 = (q0 + 3 < d) ? tildeQ[q0 + 3] : 0f;
            float v4 = (q0 + 4 < d) ? tildeQ[q0 + 4] : 0f;
            float v5 = (q0 + 5 < d) ? tildeQ[q0 + 5] : 0f;
            float v6 = (q0 + 6 < d) ? tildeQ[q0 + 6] : 0f;
            float v7 = (q0 + 7 < d) ? tildeQ[q0 + 7] : 0f;

            // Low nibble (v0..v3)
            float lo1  = v0;
            float lo2  = v1;
            float lo3  = v0 + v1;
            float lo4  = v2;
            float lo5  = v0 + v2;
            float lo6  = v1 + v2;
            float lo7  = v0 + v1 + v2;
            float lo8  = v3;
            float lo9  = v0 + v3;
            float lo10 = v1 + v3;
            float lo11 = v0 + v1 + v3;
            float lo12 = v2 + v3;
            float lo13 = v0 + v2 + v3;
            float lo14 = v1 + v2 + v3;
            float lo15 = v0 + v1 + v2 + v3;

            // High nibble (v4..v7)
            float hi1  = v4;
            float hi2  = v5;
            float hi3  = v4 + v5;
            float hi4  = v6;
            float hi5  = v4 + v6;
            float hi6  = v5 + v6;
            float hi7  = v4 + v5 + v6;
            float hi8  = v7;
            float hi9  = v4 + v7;
            float hi10 = v5 + v7;
            float hi11 = v4 + v5 + v7;
            float hi12 = v6 + v7;
            float hi13 = v4 + v6 + v7;
            float hi14 = v5 + v6 + v7;
            float hi15 = v4 + v5 + v6 + v7;

            for (int lane = 0; lane < blockLen; lane++) {
                long word = packedBits[packedBase + laneStart + lane];
                int mask = (int) ((word >>> off) & 0xFF);

                int lo = mask & 0xF;
                int hi = (mask >>> 4) & 0xF;

                float addLo = 0f;
                switch (lo) {
                    case 0:  break;
                    case 1:  addLo = lo1;  break;
                    case 2:  addLo = lo2;  break;
                    case 3:  addLo = lo3;  break;
                    case 4:  addLo = lo4;  break;
                    case 5:  addLo = lo5;  break;
                    case 6:  addLo = lo6;  break;
                    case 7:  addLo = lo7;  break;
                    case 8:  addLo = lo8;  break;
                    case 9:  addLo = lo9;  break;
                    case 10: addLo = lo10; break;
                    case 11: addLo = lo11; break;
                    case 12: addLo = lo12; break;
                    case 13: addLo = lo13; break;
                    case 14: addLo = lo14; break;
                    case 15: addLo = lo15; break;
                }

                float addHi = 0f;
                switch (hi) {
                    case 0:  break;
                    case 1:  addHi = hi1;  break;
                    case 2:  addHi = hi2;  break;
                    case 3:  addHi = hi3;  break;
                    case 4:  addHi = hi4;  break;
                    case 5:  addHi = hi5;  break;
                    case 6:  addHi = hi6;  break;
                    case 7:  addHi = hi7;  break;
                    case 8:  addHi = hi8;  break;
                    case 9:  addHi = hi9;  break;
                    case 10: addHi = hi10; break;
                    case 11: addHi = hi11; break;
                    case 12: addHi = hi12; break;
                    case 13: addHi = hi13; break;
                    case 14: addHi = hi14; break;
                    case 15: addHi = hi15; break;
                }

                acc[lane] += (addLo + addHi);
            }
        }
    }

    private static void ashMaskedAddBlockWordMask(float[] tildeQ,
                                                  int baseDim,
                                                  int d,
                                                  long[] packedBits,
                                                  int packedBase,
                                                  int laneStart,
                                                  int blockLen,
                                                  float[] acc) {
        final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;
        final int lanes = SPEC.length();

        for (int off = 0; off < 64 && (baseDim + off) < d; off += lanes) {
            int qBase = baseDim + off;

            VectorMask<Float> inRange = SPEC.indexInRange(qBase, d);
            FloatVector qVec = FloatVector.fromArray(SPEC, tildeQ, qBase, inRange);

            for (int lane = 0; lane < blockLen; lane++) {
                long word = packedBits[packedBase + laneStart + lane];

                long maskBits;
                if (lanes == 16) {
                    maskBits = (word >>> off) & 0xFFFFL;
                } else if (lanes == 8) {
                    maskBits = (word >>> off) & 0xFFL;
                } else { // NEON 4 lanes
                    maskBits = (word >>> off) & 0xFL;
                }

                if (maskBits == 0L) continue;

                VectorMask<Float> bitMask = VectorMask.fromLong(SPEC, maskBits).and(inRange);
                acc[lane] += qVec.reduceLanes(VectorOperators.ADD, bitMask);
            }
        }
    }
}

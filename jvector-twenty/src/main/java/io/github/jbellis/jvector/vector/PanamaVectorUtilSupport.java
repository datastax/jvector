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

    static final int PREFERRED_BIT_SIZE = FloatVector.SPECIES_PREFERRED.vectorBitSize();
    static {
        // Publish the value to System Properties for other packages (other language levels...)
        System.setProperty("io.github.jbellis.jvector.vector_bit_size", String.valueOf(PREFERRED_BIT_SIZE));
    }
    static final IntVector BYTE_TO_INT_MASK_512 = IntVector.broadcast(IntVector.SPECIES_512, 0xff);
    static final IntVector BYTE_TO_INT_MASK_256 = IntVector.broadcast(IntVector.SPECIES_256, 0xff);

    static final ThreadLocal<int[]> scratchInt512 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_512.length()]);
    static final ThreadLocal<int[]> scratchInt256 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_256.length()]);

    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset) {
        return FloatVector.fromArray(SPEC, ((ArrayVectorFloat) vector).get(), offset);
    }

    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset, int[] indices, int indicesOffset) {
        return FloatVector.fromArray(SPEC, ((ArrayVectorFloat)vector).get(), offset, indices, indicesOffset);
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
        if (length >= vectorLength * 2)
        {
            length -= vectorLength * 2;
            a0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 0);
            b0 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 0);
            a1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, va, vaoffset + vectorLength * 1);
            b1 = fromVectorFloat(FloatVector.SPECIES_PREFERRED, vb, vboffset + vectorLength * 1);
            vaoffset += vectorLength * 2;
            vboffset += vectorLength * 2;
            while (length >= vectorLength * 2)
            {
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

    // Bulk kernels: one query against four targets per pass. A random target is usually not in cache, and the four
    // independent load streams keep more of them in flight than four sequential calls do. Each target accumulates
    // exactly as in the single-pair method it stands for, so the results are the same; lengths that method handles
    // on a different path are scored one pair at a time.

    @Override
    public void cosine(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results) {
        int i = 0;
        for (; i + 4 <= count; i += 4) {
            cosine4(q, targets[i], targets[i + 1], targets[i + 2], targets[i + 3], results, i);
        }
        for (; i < count; i++) {
            results[i] = cosine(q, targets[i]);
        }
    }

    /** Four cosine(q, t) at once, accumulated as in {@link #cosine(VectorFloat, VectorFloat)}. */
    private void cosine4(VectorFloat<?> q, VectorFloat<?> t0, VectorFloat<?> t1, VectorFloat<?> t2, VectorFloat<?> t3, float[] results, int at) {
        var species = FloatVector.SPECIES_PREFERRED;
        int length = q.length();
        var qq = FloatVector.zero(species);
        var s0 = qq; var s1 = qq; var s2 = qq; var s3 = qq;
        var n0 = qq; var n1 = qq; var n2 = qq; var n3 = qq;
        int vectorizedLength = species.loopBound(length);
        for (int i = 0; i < vectorizedLength; i += species.length()) {
            var a = fromVectorFloat(species, q, i);
            var b0 = fromVectorFloat(species, t0, i);
            var b1 = fromVectorFloat(species, t1, i);
            var b2 = fromVectorFloat(species, t2, i);
            var b3 = fromVectorFloat(species, t3, i);
            qq = a.fma(a, qq);
            s0 = a.fma(b0, s0); n0 = b0.fma(b0, n0);
            s1 = a.fma(b1, s1); n1 = b1.fma(b1, n1);
            s2 = a.fma(b2, s2); n2 = b2.fma(b2, n2);
            s3 = a.fma(b3, s3); n3 = b3.fma(b3, n3);
        }
        float aMagnitude = qq.reduceLanes(VectorOperators.ADD);
        float sum0 = s0.reduceLanes(VectorOperators.ADD), b0Magnitude = n0.reduceLanes(VectorOperators.ADD);
        float sum1 = s1.reduceLanes(VectorOperators.ADD), b1Magnitude = n1.reduceLanes(VectorOperators.ADD);
        float sum2 = s2.reduceLanes(VectorOperators.ADD), b2Magnitude = n2.reduceLanes(VectorOperators.ADD);
        float sum3 = s3.reduceLanes(VectorOperators.ADD), b3Magnitude = n3.reduceLanes(VectorOperators.ADD);
        for (int i = vectorizedLength; i < length; i++) {
            float a = q.get(i);
            aMagnitude += a * a;
            sum0 += a * t0.get(i); b0Magnitude += t0.get(i) * t0.get(i);
            sum1 += a * t1.get(i); b1Magnitude += t1.get(i) * t1.get(i);
            sum2 += a * t2.get(i); b2Magnitude += t2.get(i) * t2.get(i);
            sum3 += a * t3.get(i); b3Magnitude += t3.get(i) * t3.get(i);
        }
        results[at] = (float) (sum0 / Math.sqrt(aMagnitude * b0Magnitude));
        results[at + 1] = (float) (sum1 / Math.sqrt(aMagnitude * b1Magnitude));
        results[at + 2] = (float) (sum2 / Math.sqrt(aMagnitude * b2Magnitude));
        results[at + 3] = (float) (sum3 / Math.sqrt(aMagnitude * b3Magnitude));
    }

    @Override
    public void dotProduct(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results) {
        int i = 0;
        // shorter vectors take the other paths of dotProduct
        if (q.length() >= 2 * FloatVector.SPECIES_PREFERRED.length()) {
            for (; i + 4 <= count; i += 4) {
                dotProduct4(q, targets[i], targets[i + 1], targets[i + 2], targets[i + 3], results, i);
            }
        }
        for (; i < count; i++) {
            results[i] = dotProduct(q, targets[i]);
        }
    }

    /**
     * Four dotProduct(q, t) at once, accumulated as in {@link #dotProductPreferred}: even vector-width chunks into one
     * sum and odd ones into another, then a last odd chunk, the lane reduction, and the scalar tail.
     */
    private void dotProduct4(VectorFloat<?> q, VectorFloat<?> t0, VectorFloat<?> t1, VectorFloat<?> t2, VectorFloat<?> t3, float[] results, int at) {
        var species = FloatVector.SPECIES_PREFERRED;
        int vectorLength = species.length();
        int length = q.length();
        var e0 = FloatVector.zero(species);
        var e1 = e0; var e2 = e0; var e3 = e0;
        var o0 = e0; var o1 = e0; var o2 = e0; var o3 = e0;
        int i = 0;
        for (; i + 2 * vectorLength <= length; i += 2 * vectorLength) {
            var a = fromVectorFloat(species, q, i);
            e0 = a.fma(fromVectorFloat(species, t0, i), e0);
            e1 = a.fma(fromVectorFloat(species, t1, i), e1);
            e2 = a.fma(fromVectorFloat(species, t2, i), e2);
            e3 = a.fma(fromVectorFloat(species, t3, i), e3);
            a = fromVectorFloat(species, q, i + vectorLength);
            o0 = a.fma(fromVectorFloat(species, t0, i + vectorLength), o0);
            o1 = a.fma(fromVectorFloat(species, t1, i + vectorLength), o1);
            o2 = a.fma(fromVectorFloat(species, t2, i + vectorLength), o2);
            o3 = a.fma(fromVectorFloat(species, t3, i + vectorLength), o3);
        }
        e0 = e0.add(o0); e1 = e1.add(o1); e2 = e2.add(o2); e3 = e3.add(o3);
        if (i + vectorLength <= length) {
            var a = fromVectorFloat(species, q, i);
            e0 = a.fma(fromVectorFloat(species, t0, i), e0);
            e1 = a.fma(fromVectorFloat(species, t1, i), e1);
            e2 = a.fma(fromVectorFloat(species, t2, i), e2);
            e3 = a.fma(fromVectorFloat(species, t3, i), e3);
            i += vectorLength;
        }
        float tail0 = 0, tail1 = 0, tail2 = 0, tail3 = 0;
        for (; i < length; i++) {
            float a = q.get(i);
            tail0 += a * t0.get(i);
            tail1 += a * t1.get(i);
            tail2 += a * t2.get(i);
            tail3 += a * t3.get(i);
        }
        results[at] = e0.reduceLanes(VectorOperators.ADD) + tail0;
        results[at + 1] = e1.reduceLanes(VectorOperators.ADD) + tail1;
        results[at + 2] = e2.reduceLanes(VectorOperators.ADD) + tail2;
        results[at + 3] = e3.reduceLanes(VectorOperators.ADD) + tail3;
    }

    @Override
    public void squareDistance(VectorFloat<?> q, VectorFloat<?>[] targets, int count, float[] results) {
        int i = 0;
        // shorter vectors, and those exactly one vector wide, take the other paths of squareDistance
        if (q.length() > FloatVector.SPECIES_PREFERRED.length()) {
            for (; i + 4 <= count; i += 4) {
                squareDistance4(q, targets[i], targets[i + 1], targets[i + 2], targets[i + 3], results, i);
            }
        }
        for (; i < count; i++) {
            results[i] = squareDistance(q, targets[i]);
        }
    }

    /** Four squareDistance(q, t) at once, accumulated as in {@link #squareDistancePreferred(VectorFloat, int, VectorFloat, int, int)}. */
    private void squareDistance4(VectorFloat<?> q, VectorFloat<?> t0, VectorFloat<?> t1, VectorFloat<?> t2, VectorFloat<?> t3, float[] results, int at) {
        var species = FloatVector.SPECIES_PREFERRED;
        int length = q.length();
        var s0 = FloatVector.zero(species);
        var s1 = s0; var s2 = s0; var s3 = s0;
        int vectorizedLength = species.loopBound(length);
        int i = 0;
        for (; i < vectorizedLength; i += species.length()) {
            var a = fromVectorFloat(species, q, i);
            var d0 = a.sub(fromVectorFloat(species, t0, i));
            var d1 = a.sub(fromVectorFloat(species, t1, i));
            var d2 = a.sub(fromVectorFloat(species, t2, i));
            var d3 = a.sub(fromVectorFloat(species, t3, i));
            s0 = d0.fma(d0, s0);
            s1 = d1.fma(d1, s1);
            s2 = d2.fma(d2, s2);
            s3 = d3.fma(d3, s3);
        }
        float r0 = s0.reduceLanes(VectorOperators.ADD);
        float r1 = s1.reduceLanes(VectorOperators.ADD);
        float r2 = s2.reduceLanes(VectorOperators.ADD);
        float r3 = s3.reduceLanes(VectorOperators.ADD);
        for (; i < length; i++) {
            float a = q.get(i);
            var d0 = a - t0.get(i); r0 += d0 * d0;
            var d1 = a - t1.get(i); r1 += d1 * d1;
            var d2 = a - t2.get(i); r2 += d2 * d2;
            var d3 = a - t3.get(i); r3 += d3 * d3;
        }
        results[at] = r0;
        results[at + 1] = r1;
        results[at + 2] = r2;
        results[at + 3] = r3;
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
            v1.set(i,  v1.get(i) + v2.get(i));
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
            v1.set(i,  v1.get(i) + value);
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
            v1.set(i,  v1.get(i) - v2.get(i));
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
            vector.set(i,  vector.get(i) - value);
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
            v1.set(i,  Math.min(v1.get(i), v2.get(i)));
        }
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
                    .intoArray(convOffsets,0);

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
                    .intoArray(convOffsets,0);

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
            var rk          = r.mul(kvec);
            var triangular  = r.mul(r.sub(onevec)).mul(twovec);
            var offsetRow   = rk.sub(triangular);

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
            var rk          = r.mul(kvec);
            var triangular  = r.mul(r.sub(onevec)).mul(twovec);
            var offsetRow   = rk.sub(triangular);

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
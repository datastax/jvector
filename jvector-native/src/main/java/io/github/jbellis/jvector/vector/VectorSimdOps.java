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

import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import java.nio.ByteOrder;
import java.util.List;

/**
 * Support class for vector operations using a mix of native and Panama SIMD.
 */
final class VectorSimdOps {
    static final boolean HAS_AVX512 = IntVector.SPECIES_PREFERRED == IntVector.SPECIES_512;

    static float sum(MemorySegmentVectorFloat vector) {
        var sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, vector.get(), vector.offset(i), ByteOrder.LITTLE_ENDIAN);
            sum = sum.add(a);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            res += vector.get(i);
        }

        return res;
    }

    static VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Input list cannot be null or empty");
        }

        int dimension = vectors.get(0).length();
        MemorySegmentVectorFloat sum = new MemorySegmentVectorFloat(dimension);

        // Process each vector from the list
        for (VectorFloat<?> vector : vectors) {
            addInPlace(sum, (MemorySegmentVectorFloat) vector);
        }

        return sum;
    }

    static void scale(MemorySegmentVectorFloat vector, float multiplier) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, vector.get(), vector.offset(i), ByteOrder.LITTLE_ENDIAN);
            var divResult = a.mul(multiplier);
            divResult.intoMemorySegment(vector.get(), vector.offset(i), ByteOrder.LITTLE_ENDIAN);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, vector.get(i) * multiplier);
        }
    }

    static float dot64(MemorySegmentVectorFloat v1, int offset1, MemorySegmentVectorFloat v2, int offset2) {
        var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v1.get(), v1.offset(offset1), ByteOrder.LITTLE_ENDIAN);
        var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v2.get(), v1.offset(offset2), ByteOrder.LITTLE_ENDIAN);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    static float dot128(MemorySegmentVectorFloat v1, int offset1, MemorySegmentVectorFloat v2, int offset2) {
        var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_128, v1.get(), v1.offset(offset1), ByteOrder.LITTLE_ENDIAN);
        var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_128, v2.get(), v2.offset(offset2), ByteOrder.LITTLE_ENDIAN);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    static float dot256(MemorySegmentVectorFloat v1, int offset1, MemorySegmentVectorFloat v2, int offset2) {
        var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, v1.get(), v1.offset(offset1), ByteOrder.LITTLE_ENDIAN);
        var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, v2.get(), v2.offset(offset2), ByteOrder.LITTLE_ENDIAN);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    static float dotPreferred(MemorySegmentVectorFloat v1, int offset1, MemorySegmentVectorFloat v2, int offset2) {
        var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v1.get(), v1.offset(offset1), ByteOrder.LITTLE_ENDIAN);
        var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v2.get(), v2.offset(offset2), ByteOrder.LITTLE_ENDIAN);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    static float dotProduct(MemorySegmentVectorFloat v1, MemorySegmentVectorFloat v2) {
        return dotProduct(v1, 0, v2, 0, v1.length());
    }

    static float dotProduct(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, final int length)
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

    static float dotProduct64(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_64.length())
            return dot64(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v1.get(), v1.offset(v1offset + i), ByteOrder.LITTLE_ENDIAN);
            FloatVector b = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v2.get(), v2.offset(v2offset + i), ByteOrder.LITTLE_ENDIAN);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    static float dotProduct128(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_128.length())
            return dot128(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_128, v1.get(), v1.offset(v1offset + i), ByteOrder.LITTLE_ENDIAN);
            FloatVector b = FloatVector.fromMemorySegment(FloatVector.SPECIES_128, v2.get(), v2.offset(v2offset + i), ByteOrder.LITTLE_ENDIAN);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }


    static float dotProduct256(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_256.length())
            return dot256(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, v1.get(), v1.offset(v1offset + i), ByteOrder.LITTLE_ENDIAN);
            FloatVector b = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, v2.get(), v1.offset(v2offset + i), ByteOrder.LITTLE_ENDIAN);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    static float dotProductPreferred(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_PREFERRED.length())
            return dotPreferred(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v1.get(), v1.offset(v1offset + i), ByteOrder.LITTLE_ENDIAN);
            FloatVector b = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v2.get(), v2.offset(v2offset + i), ByteOrder.LITTLE_ENDIAN);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    static float cosineSimilarity(MemorySegmentVectorFloat v1, MemorySegmentVectorFloat v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());
        // Process the vectorized part, convert from 8 bytes to 8 ints
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v1.get(), v1.offset(i), ByteOrder.LITTLE_ENDIAN);
            var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v2.get(), v2.offset(i), ByteOrder.LITTLE_ENDIAN);
            vsum = vsum.add(a.mul(b));
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

    static float cosineSimilarity(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, int length) {
        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v1.get(), v1.offset(v1offset + i), ByteOrder.LITTLE_ENDIAN);
            var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v2.get(), v2.offset(v2offset + i), ByteOrder.LITTLE_ENDIAN);
            vsum = vsum.add(a.mul(b));
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

    static float squareDistance64(MemorySegmentVectorFloat v1, int offset1, MemorySegmentVectorFloat v2, int offset2) {
        var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v1.get(), v1.offset(offset1), ByteOrder.LITTLE_ENDIAN);
        var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v2.get(), v2.offset(offset2), ByteOrder.LITTLE_ENDIAN);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    static float squareDistance128(MemorySegmentVectorFloat v1, int offset1, MemorySegmentVectorFloat v2, int offset2) {
        var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_128, v1.get(), v1.offset(offset1), ByteOrder.LITTLE_ENDIAN);
        var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_128, v2.get(), v2.offset(offset2), ByteOrder.LITTLE_ENDIAN);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    static float squareDistance256(MemorySegmentVectorFloat v1, int offset1, MemorySegmentVectorFloat v2, int offset2) {
        var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, v1.get(), v1.offset(offset1), ByteOrder.LITTLE_ENDIAN);
        var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, v2.get(), v2.offset(offset2), ByteOrder.LITTLE_ENDIAN);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    static float squareDistancePreferred(MemorySegmentVectorFloat v1, int offset1, MemorySegmentVectorFloat v2, int offset2) {
        var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v1.get(), v1.offset(offset1), ByteOrder.LITTLE_ENDIAN);
        var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v2.get(), v2.offset(offset2), ByteOrder.LITTLE_ENDIAN);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    static float squareDistance(MemorySegmentVectorFloat v1, MemorySegmentVectorFloat v2) {
        return squareDistance(v1, 0, v2, 0, v1.length());
    }

    static float squareDistance(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, final int length)
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

    static float squareDistance64(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_64.length())
            return squareDistance64(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v1.get(), v1.offset(v1offset + i), ByteOrder.LITTLE_ENDIAN);
            FloatVector b = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v2.get(), v2.offset(v2offset + i), ByteOrder.LITTLE_ENDIAN);
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

    static float squareDistance128(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_128.length())
            return squareDistance128(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_128, v1.get(), v1.offset(v1offset + i), ByteOrder.LITTLE_ENDIAN);
            FloatVector b = FloatVector.fromMemorySegment(FloatVector.SPECIES_128, v2.get(), v2.offset(v2offset + i), ByteOrder.LITTLE_ENDIAN);
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


    static float squareDistance256(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_256.length())
            return squareDistance256(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, v1.get(), v1.offset(v1offset + i), ByteOrder.LITTLE_ENDIAN);
            FloatVector b = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, v2.get(), v2.offset(v2offset + i), ByteOrder.LITTLE_ENDIAN);
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

    static float squareDistancePreferred(MemorySegmentVectorFloat v1, int v1offset, MemorySegmentVectorFloat v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_PREFERRED.length())
            return squareDistancePreferred(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v1.get(), v1.offset(v1offset + i), ByteOrder.LITTLE_ENDIAN);
            FloatVector b = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v2.get(), v2.offset(v2offset + i), ByteOrder.LITTLE_ENDIAN);
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

    static void addInPlace64(MemorySegmentVectorFloat v1, MemorySegmentVectorFloat v2) {
        var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v1.get(), 0, ByteOrder.LITTLE_ENDIAN);
        var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_64, v2.get(), 0, ByteOrder.LITTLE_ENDIAN);
        a.add(b).intoMemorySegment(v1.get(), v1.offset(0), ByteOrder.LITTLE_ENDIAN);
    }

    static void addInPlace(MemorySegmentVectorFloat v1, MemorySegmentVectorFloat v2) {
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
            var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v1.get(), v1.offset(i), ByteOrder.LITTLE_ENDIAN);
            var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v2.get(), v2.offset(i), ByteOrder.LITTLE_ENDIAN);
            a.add(b).intoMemorySegment(v1.get(), v1.offset(i), ByteOrder.LITTLE_ENDIAN);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i,  v1.get(i) + v2.get(i));
        }
    }

    static VectorFloat<?> sub(MemorySegmentVectorFloat a, int aOffset, MemorySegmentVectorFloat b, int bOffset, int length) {
        MemorySegmentVectorFloat result = new MemorySegmentVectorFloat(length);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var lhs = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, a.get(), a.offset(aOffset + i), ByteOrder.LITTLE_ENDIAN);
            var rhs = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, b.get(), b.offset(bOffset + i), ByteOrder.LITTLE_ENDIAN);
            var subResult = lhs.sub(rhs);
            subResult.intoMemorySegment(result.get(), result.offset(i), ByteOrder.LITTLE_ENDIAN);
        }

        // Process the tail
        for (int i = vectorizedLength; i < length; i++) {
            result.set(i, a.get(aOffset + i) - b.get(bOffset + i));
        }

        return result;
    }

    static void subInPlace(MemorySegmentVectorFloat v1, MemorySegmentVectorFloat v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v1.get(), v1.offset(i), ByteOrder.LITTLE_ENDIAN);
            var b = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, v2.get(), v2.offset(i), ByteOrder.LITTLE_ENDIAN);
            a.sub(b).intoMemorySegment(v1.get(), v1.offset(i), ByteOrder.LITTLE_ENDIAN);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i,  v1.get(i) - v2.get(i));
        }
    }

    public static int hammingDistance(long[] a, long[] b) {
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

    public static float max(MemorySegmentVectorFloat vector) {
        var accum = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, -Float.MAX_VALUE);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, vector.get(), vector.offset(i), ByteOrder.LITTLE_ENDIAN);
            accum = accum.max(a);
        }
        float max = accum.reduceLanes(VectorOperators.MAX);
        for (int i = vectorizedLength; i < vector.length(); i++) {
            max = Math.max(max, vector.get(i));
        }
        return max;
    }

    public static float min(MemorySegmentVectorFloat vector) {
        var accum = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, Float.MAX_VALUE);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, vector.get(), vector.offset(i), ByteOrder.LITTLE_ENDIAN);
            accum = accum.min(a);
        }
        float min = accum.reduceLanes(VectorOperators.MIN);
        for (int i = vectorizedLength; i < vector.length(); i++) {
            min = Math.min(min, vector.get(i));
        }
        return min;
    }

    public static void quantizePartials(float delta, MemorySegmentVectorFloat partials, MemorySegmentVectorFloat partialBases, MemorySegmentByteSequence quantizedPartials) {
        var codebookSize = partials.length() / partialBases.length();
        var codebookCount = partialBases.length();

        for (int i = 0; i < codebookCount; i++) {
            var vectorizedLength = FloatVector.SPECIES_512.loopBound(codebookSize);
            var codebookBase = partialBases.get(i);
            var codebookBaseVector = FloatVector.broadcast(FloatVector.SPECIES_512, codebookBase);
            int j = 0;
            for (; j < vectorizedLength; j += FloatVector.SPECIES_512.length()) {
                var partialVector = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, partials.get(), partials.offset(i * codebookSize + j), ByteOrder.LITTLE_ENDIAN);
                var quantized = (partialVector.sub(codebookBaseVector)).div(delta);
                quantized = quantized.max(FloatVector.zero(FloatVector.SPECIES_512)).min(FloatVector.broadcast(FloatVector.SPECIES_512, 65535));
                var quantizedBytes = (ShortVector) quantized.convertShape(VectorOperators.F2S, ShortVector.SPECIES_256, 0);
                quantizedBytes.intoMemorySegment(quantizedPartials.get(), 2 * (i * codebookSize + j), ByteOrder.LITTLE_ENDIAN);
            }
            for (; j < codebookSize; j++) {
                var val = partials.get(i * codebookSize + j);
                var quantized = (short) Math.min((val - codebookBase) / delta, 65535);
                quantizedPartials.setLittleEndianShort(i * codebookSize + j, quantized);
            }
        }
    }
}

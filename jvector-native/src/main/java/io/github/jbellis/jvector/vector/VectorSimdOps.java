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
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.nio.ByteOrder;
import java.util.List;

/**
 * Support class for vector operations using a mix of native and Panama SIMD.
 */
class VectorSimdOps extends SimdOps {

    @Override
    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset)
    {
        return FloatVector.fromMemorySegment(SPEC, ((MemorySegmentVectorFloat) vector).get(), vector.offset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected FloatVector fromVectorFloat(VectorSpecies<Float> SPEC, VectorFloat<?> vector, int offset, int[] indices, int indicesOffset)
    {
        throw new UnsupportedOperationException("Assembly not supported with memory segments.");
    }

    @Override
    protected void intoVectorFloat(FloatVector vector, VectorFloat<?> v, int offset)
    {
        vector.intoMemorySegment(((MemorySegmentVectorFloat) v).get(), v.offset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected ByteVector fromByteSequence(VectorSpecies<Byte> SPEC, ByteSequence<?> vector, int offset)
    {
        return ByteVector.fromMemorySegment(SPEC, ((MemorySegmentByteSequence) vector).get(), offset, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    protected void intoByteSequence(ByteVector vector, ByteSequence<?> v, int offset)
    {
        vector.intoMemorySegment(((MemorySegmentByteSequence) v).get(), offset, ByteOrder.LITTLE_ENDIAN);
    }
}

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

package io.github.jbellis.jvector.fastervector;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * {@link Vecs} view of jvector's {@link RandomAccessVectorValues}, so jvector vectors can be
 * indexed without first being copied into another container. Ordinal {@code i} is the
 * jvector node ordinal {@code i}, so search-result ordinals are jvector node ordinals.
 * <p>
 * Each vector is copied out of {@link RandomAccessVectorValues#getVector} before the next
 * call, so a {@code ravv} that returns a shared buffer is fine. A cursor reads {@code ravv}
 * directly, so cursors must not be used concurrently unless {@code ravv} is thread-safe;
 * the index builder reads its passes one after another on one thread.
 */
public final class RandomAccessVectorValuesVecs implements Vecs {
    private final RandomAccessVectorValues ravv;

    public RandomAccessVectorValuesVecs(RandomAccessVectorValues ravv) {
        this.ravv = Objects.requireNonNull(ravv, "ravv must not be null");
    }

    @Override
    public long nvecs() {
        return ravv.size();
    }

    @Override
    public int dim() {
        return ravv.dimension();
    }

    @Override
    public VecsCursor getCursor() {
        return new VecsCursor() {
            private long ord;

            @Override
            public long ord() {
                return ord;
            }

            @Override
            public void seek(long n) {
                if (n < 0 || n > ravv.size()) {
                    throw new IllegalArgumentException("Seek out of range: " + n);
                }
                ord = n;
            }

            @Override
            public boolean nextVectorInto(MemorySegment out) {
                if (ord >= ravv.size()) {
                    return false;
                }
                VectorFloat<?> v = ravv.getVector((int) ord);
                if (v == null) {
                    throw new IllegalStateException("No vector for ordinal " + ord);
                }
                if (v.length() != ravv.dimension()) {
                    throw new IllegalStateException(
                        "Vector " + ord + " has length " + v.length() + ", expected " + ravv.dimension());
                }
                // TODO: bulk copy for array- and MemorySegment-backed VectorFloats
                for (int i = 0; i < v.length(); i++) {
                    out.setAtIndex(ValueLayout.JAVA_FLOAT_UNALIGNED, i, v.get(i));
                }
                ord++;
                return true;
            }
        };
    }
}

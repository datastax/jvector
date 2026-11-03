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

package io.github.jbellis.jvector.fastervector.utils;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * In-memory {@link io.github.jbellis.jvector.fastervector.Vecs} over a {@code float[][]}.
 * The dimension is taken from the first row; every row must have that length.
 * Rows are not copied, so they must not be mutated while this object is in use.
 */
public final class Array2DVecs extends HeapVecs {

    private final float[][] rows;

    /** @throws IllegalArgumentException if empty, or row lengths differ or are zero */
    public Array2DVecs(float[][] rows) {
        super(rows.length, dimOf(rows));
        this.rows = rows;
        for (int i = 0; i < rows.length; i++) {
            if (rows[i].length != dim()) {
                throw new IllegalArgumentException(
                    "Vector " + i + " has length " + rows[i].length + ", expected " + dim());
            }
        }
    }

    private static int dimOf(float[][] rows) {
        if (rows.length == 0) {
            throw new IllegalArgumentException("Cannot infer dimension from an empty vector set");
        }
        return rows[0].length;
    }

    @Override
    void copyInto(long ord, MemorySegment out) {
        MemorySegment.copy(rows[(int) ord], 0, out, ValueLayout.JAVA_FLOAT_UNALIGNED, 0L, dim());
    }
}

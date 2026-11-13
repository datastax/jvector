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
package io.github.jbellis.jvector.graph.disk.feature;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import io.github.jbellis.jvector.util.MaxVectorLength;

import java.io.IOException;

/**
 * The largest length of the full-precision vectors a feature stores, kept in its header from format version
 * {@value #VERSION}. Dot-product searches that stop adaptively need it; without it they stop by the standard rule.
 *
 * <p>A feature records each vector as it writes it. A header written before any vector, as a sequential writer writes
 * it, stores NaN (unknown); a writer that rewrites the header after the vectors stores their largest length.
 */
final class MaxVectorNorm {
    /** The first on-disk format version whose full-precision vector headers store the largest length. */
    static final int VERSION = 7;

    private final MaxVectorLength written = new MaxVectorLength();

    void record(VectorFloat<?> vector) {
        written.record(vector);
    }

    /** The largest length recorded, or NaN if none was. */
    float get() {
        return written.get();
    }

    static int headerSize(int version) {
        return version >= VERSION ? Float.BYTES : 0;
    }

    void write(IndexWriter out, int version) throws IOException {
        if (version >= VERSION) out.writeFloat(get());
    }

    /** The stored largest length, NaN when unknown or when {@code version} predates it. */
    static float read(RandomAccessReader in, int version) throws IOException {
        return version >= VERSION ? in.readFloat() : Float.NaN;
    }
}

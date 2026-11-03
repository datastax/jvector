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

/**
 * A read-only dataset of float vectors with random access via cursors.
 * <p>
 * Mirrors the Rust {@code Vecs} trait in {@code api/mod.rs}.
 */
public interface Vecs {

    /** Number of vectors in this dataset. */
    long nvecs();

    /** Dimension (number of floats) per vector. */
    int dim();

    /**
     * Create a new cursor positioned at ordinal 0.
     * The caller is responsible for closing the cursor if it implements {@link AutoCloseable}.
     */
    VecsCursor getCursor();
}

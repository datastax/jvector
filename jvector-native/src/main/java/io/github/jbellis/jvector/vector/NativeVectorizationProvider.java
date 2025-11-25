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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import io.github.jbellis.jvector.vector.cnative.NativeSimdOps;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Experimental!
 * VectorizationProvider implementation that uses MemorySegment vectors and prefers native/Panama SIMD.
 */
@Experimental
public class NativeVectorizationProvider extends VectorizationProvider {
    private final VectorUtilSupport vectorUtilSupport;
    private final VectorTypeSupport vectorTypeSupport;

    public NativeVectorizationProvider() {
        var libraryLoaded = LibraryLoader.loadJvector();
        if (!libraryLoaded) {
            throw new UnsupportedOperationException("Failed to load supporting native library.");
        }
        if (!NativeSimdOps.check_compatibility()) {
            throw new UnsupportedOperationException("Native SIMD operations are not supported on this platform due to missing CPU support.");
        }
        this.vectorUtilSupport = new NativeVectorUtilSupport();
        this.vectorTypeSupport = new MemorySegmentVectorProvider();
    }

    @Override
    public VectorUtilSupport getVectorUtilSupport() {
        return vectorUtilSupport;
    }

    @Override
    public VectorTypeSupport getVectorTypeSupport() {
        return vectorTypeSupport;
    }
}
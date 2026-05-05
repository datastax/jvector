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

import io.github.jbellis.jvector.quantization.ash.AshDriverFactory;
import io.github.jbellis.jvector.quantization.ash.PanamaAshDriverFactory;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import jdk.incubator.vector.FloatVector;

/**
 * Vectorization provider that uses on-heap arrays and SIMD operations through Panama SIMD API.
 */
public class PanamaVectorizationProvider extends VectorizationProvider
{
    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
    }

    private final VectorUtilSupport vectorUtilSupport;
    private final VectorTypeSupport vectorTypeSupport;
    private final AshDriverFactory ashDriverFactory;

    public PanamaVectorizationProvider() {
        this.vectorUtilSupport = new PanamaVectorUtilSupport();
        LOG.info("Preferred f32 species is " + FloatVector.SPECIES_PREFERRED.vectorBitSize());
        this.vectorTypeSupport = new ArrayVectorProvider();
        this.ashDriverFactory = new PanamaAshDriverFactory();
    }

    @Override
    public VectorUtilSupport getVectorUtilSupport() {
        return vectorUtilSupport;
    }

    @Override
    public VectorTypeSupport getVectorTypeSupport() {
        return vectorTypeSupport;
    }

    @Override
    public AshDriverFactory getAshDriverFactory() {
        return ashDriverFactory;
    }
}

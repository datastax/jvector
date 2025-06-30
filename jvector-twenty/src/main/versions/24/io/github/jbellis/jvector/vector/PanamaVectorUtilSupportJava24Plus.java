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

import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.concurrent.atomic.AtomicReference;

final class PanamaVectorUtilSupportJava24Plus
        extends PanamaVectorUtilSupport implements VectorUtilSupport
{
    @Override
    public void quantizePartials(float delta, VectorFloat<?> partials, VectorFloat<?> partialBases, AtomicReference<Object> quantizedPartialsShorts) {
        SimdOpsJava24Plus.quantizePartials(delta, (ArrayVectorFloat) partials, (ArrayVectorFloat) partialBases, quantizedPartialsShorts);
    }

    @Override
    public void bulkShuffleQuantizedSimilarity(ByteSequence<?> shuffles,
            int codebookCount,
            AtomicReference<Object> quantizedPartialSumsShorts,
            float delta,
            float minDistance,
            VectorSimilarityFunction vsf, VectorFloat<?> results) {
        SimdOpsJava24Plus.bulkShuffleQuantizedSimilarity((ArrayByteSequence) shuffles,
                codebookCount,
                (short []) quantizedPartialSumsShorts.get(),
                delta,
                minDistance,
                vsf, (ArrayVectorFloat) results);
    }

    @Override
    public void bulkShuffleQuantizedSimilarityCosine(ByteSequence<?> shuffles,
            int codebookCount,
            AtomicReference<Object> quantizedPartialSums,  float sumDelta, float minDistance,
            AtomicReference<Object> quantizedPartialMagnitudes,  float magnitudeDelta, float minMagnitude,
            float queryMagnitudeSquared, VectorFloat<?> results) {
        SimdOpsJava24Plus.bulkShuffleQuantizedSimilarityCosine((ArrayByteSequence) shuffles,
                codebookCount,
                (short []) quantizedPartialSums.get(),  sumDelta, minDistance,
                (short []) quantizedPartialMagnitudes.get(), magnitudeDelta, minMagnitude,
                queryMagnitudeSquared, (ArrayVectorFloat) results);
    }
}

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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class TestPQEncodeAndBuild {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Test
    public void testSubsetEncoding() {
        // Create 1000 vectors
        int dimension = 16;
        int totalVectors = 1000;
        Random R = new Random(42);
        List<VectorFloat<?>> vectors = IntStream.range(0, totalVectors)
                .mapToObj(i -> {
                    float[] v = new float[dimension];
                    for (int j = 0; j < dimension; j++) v[j] = R.nextFloat();
                    return vectorTypeSupport.createFloatVector(v);
                })
                .collect(Collectors.toList());
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);

        // Compute PQ on all vectors (simplification, usually computed on sample)
        var pq = ProductQuantization.compute(ravv, 4, 256, false);
        // Try to encode a subset of 100 vectors, mapped to the *end* of the RAVV.
        // We want to encode vectors 900..999.
        // We will pretend we are building a graph of 100 nodes.
        int subsetSize = 100;

        // This call should succeed if the bounds check respects ravv.size() instead of vectorCount
        try {
            PQVectors.encodeAndBuild(
                    pq,
                    subsetSize,
                    i -> i + 900, // Mapping: 0->900, 1->901, ..., 99->999. All < totalVectors (1000)
                    ravv,
                    ForkJoinPool.commonPool()
            );
            System.out.println("SUCCESS: encodeAndBuild completed without error.");
        } catch (IllegalArgumentException e) {
            System.out.println("FAILURE: " + e.getMessage());
            if (e.getMessage().contains("out-of-bounds ordinal")) {
                System.out.println("This confirms the bug: check is against vectorCount (" + subsetSize + ") instead of ravv.size (" + totalVectors + ")");
            }
            throw e;
        }
    }
}

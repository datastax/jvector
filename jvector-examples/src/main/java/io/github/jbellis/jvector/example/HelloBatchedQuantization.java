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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.FvecFileVectorValues;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.ProductQuantization;

import java.util.concurrent.ForkJoinPool;

/** Train and encode directly from a file through the ordinary PQ and NVQ APIs. */
public class HelloBatchedQuantization {
    public static void main(String[] args) throws Exception {
        var executor = new ForkJoinPool(Math.min(8, Runtime.getRuntime().availableProcessors()));
        try (var source = FvecFileVectorValues.open(HelloVectorFile.path(args))) {
            if (source.size() < 256)
                throw new IllegalArgumentException("PQ example requires at least 256 vectors");
            System.out.printf("%,d vectors, %d dimensions%n", source.size(), source.dimension());

            // PQ chooses its training sample; VectorAccess handles efficient selection reads.
            int subspaces = Math.max(1, source.dimension() / 8);
            var pq = ProductQuantization.compute(source, subspaces, 256, false);
            var pqVectors = pq.encodeAll(source, executor);
            System.out.printf("PQ encoded %,d vectors (%d subspaces)%n", pqVectors.count(), subspaces);

            // NVQ scans the same file; the source owns batching and read-ahead.
            var nvq = NVQuantization.compute(source, 1);
            var nvqVectors = nvq.encodeAll(source, executor);
            System.out.printf("NVQ encoded %,d vectors%n", nvqVectors.count());
        } finally {
            executor.shutdown();
        }
    }
}

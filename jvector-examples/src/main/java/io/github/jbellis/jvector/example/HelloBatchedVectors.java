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
import io.github.jbellis.jvector.graph.VectorAccess;

/** Runnable example of range reads, selected reads, and borrowed-vector ownership. */
public class HelloBatchedVectors {
    public static void main(String[] args) throws Exception {
        try (var source = FvecFileVectorValues.open(HelloVectorFile.path(args))) {
            System.out.printf("%,d vectors, %d dimensions%n", source.size(), source.dimension());

            // Exclusive end: read ordinals 0 through 4 (or the entire smaller file).
            try (var cursor = VectorAccess.openRange(source, 0, Math.min(5, source.size()))) {
                while (cursor.next()) {
                    System.out.printf("Scan %d: first coordinate = %.6f%n",
                            cursor.ordinal(), cursor.vector().get(0));
                }
            }

            // A complete ordered scan consumes each vector without retaining it.
            int scanned = 0;
            double firstCoordinateSum = 0;
            try (var cursor = VectorAccess.openRange(source, 0, source.size())) {
                while (cursor.next()) {
                    firstCoordinateSum += cursor.vector().get(0);
                    scanned++;
                }
            }
            System.out.printf("Scanned %,d vectors; first-coordinate sum = %.6f%n",
                    scanned, firstCoordinateSum);

            // Arbitrary ordinals need not be sorted; duplicate requests are preserved.
            int[] selected = {source.size() - 1, 0, source.size() / 2, source.size() - 1};
            try (var cursor = VectorAccess.openSelection(source, selected, 0, selected.length)) {
                cursor.next();
                // The file cursor reuses its vector; make a copy before retaining it.
                var retained = cursor.vector().copy();
                System.out.printf("Selected %d%n", cursor.ordinal());
                while (cursor.next()) System.out.printf("Selected %d%n", cursor.ordinal());
                System.out.printf("Retained first coordinate = %.6f%n", retained.get(0));
            }

            System.out.printf("I/O payload: %,d bytes allocated, %,d bytes limit%n",
                    source.statistics().bufferBytes, source.statistics().bufferLimitBytes);
        }
    }
}

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

import io.github.jbellis.jvector.quantization.ASHNeighborhoodScorer;
import io.github.jbellis.jvector.quantization.ASHScorer;
import io.github.jbellis.jvector.quantization.FusedASHLayout;
import java.util.Locale;

/**
 * Untimed decoded-component oracle, independent of the scoring LUT and block-scoring implementation.
 * It uses the model's selected query precision and checks the serialized graph headers.
 * Projection arithmetic is covered separately by the provider's FP32 and FP16 oracle tests.
 * The tolerance permits FP32 reduction-order differences, without introducing LUT quantization.
 */
final class ASHNeighborhoodValidator {
    private ASHNeighborhoodValidator() {}

    static void validate(ASHNeighborhoodFixture fixture) {
        double maximum = 0;
        int neighborhoods = Math.min(fixture.neighborhoods.length,
                ASHNeighborhoodBenchmark.positive("jvector.bench.validationNeighborhoods", 1024));
        for (var query : fixture.queries) {
            var scorer = new ASHNeighborhoodScorer(fixture.ash, query, fixture.capacity, fixture.blockSize);
            // Share the query precision, then independently decode and accumulate
            // components in double precision instead of using the nibble LUT kernel.
            float[] input = new float[query.length()];
            double dotMu = 0;
            for (int d = 0; d < input.length; d++) {
                input[d] = query.get(d);
                dotMu += (double) input[d] * fixture.ash.landmarks[0].get(d);
            }
            float[] projected = new float[fixture.ash.quantizedDim];
            fixture.ash.stiefelTransform.projectQuery(input, projected);
            float[] scores = new float[fixture.capacity];
            for (int n = 0; n < neighborhoods; n++) {
                int index = (int) ((long) n * fixture.neighborhoods.length / neighborhoods);
                byte[] packed = fixture.neighborhoods[index];
                scorer.score(packed, scores);
                for (int i = 0; i < fixture.capacity; i++) {
                    int offset = FusedASHLayout.blockOffset(
                            i / fixture.blockSize, projected.length, fixture.ash.bitsPerDimension, fixture.blockSize);
                    int lane = i % fixture.blockSize, bits = fixture.ash.bitsPerDimension;
                    double dot = 0;
                    for (int d = 0; d < projected.length; d++) {
                        int perNibble = 4 / bits;
                        int nibble =
                                FusedASHLayout.getPackedNibble(packed, offset, lane, d / perNibble, fixture.blockSize);
                        int field = nibble >>> ((d % perNibble) * bits) & ((1 << bits) - 1);
                        dot += (double) projected[d] * FusedASHLayout.decodeProjectionComponent(field, bits);
                    }
                    float expected = ASHScorer.toSimilarity((float) (FusedASHLayout.readScale(packed, offset, lane,
                                                                             projected.length, bits, fixture.blockSize)
                                    * dot
                            + dotMu
                            + FusedASHLayout.readOffset(
                                    packed, offset, lane, projected.length, bits, fixture.blockSize)));
                    double error = Math.abs(scores[i] - expected);
                    if (!Float.isFinite(expected) || !Float.isFinite(scores[i])
                            || error > 3e-5 + Math.abs(expected) * 3e-5)
                        throw new AssertionError("Neighborhood score mismatch at " + index + "/" + i + ": " + scores[i]
                                + " vs " + expected);
                    maximum = Math.max(maximum, error);
                }
            }
        }
        System.out.printf(Locale.ROOT,
                "Validated %d queries x %d neighborhoods against decoded-component oracle; maxAbsError=%.9g%n",
                fixture.queries.size(), neighborhoods, maximum);
    }
}

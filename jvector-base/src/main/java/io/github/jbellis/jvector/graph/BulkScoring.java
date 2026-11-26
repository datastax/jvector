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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Arrays;

/**
 * The default bulk scoring of {@link RandomAccessVectorValues}: four nodes per {@link VectorSimilarityFunction} bulk
 * call, which overlaps their memory loads. A shared source's vectors are copied into per-thread vectors first.
 */
final class BulkScoring {
    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(Scratch::new);

    private static final class Scratch {
        final VectorFloat<?>[] targets = new VectorFloat<?>[4];
        final float[] scores = new float[4];
        VectorFloat<?>[] copies = new VectorFloat<?>[0];
    }

    private BulkScoring() {}

    static void score(RandomAccessVectorValues values, VectorFloat<?> query, VectorSimilarityFunction vsf,
                      int[] nodes, int count, float[] results) {
        Scratch s = SCRATCH.get();
        boolean shared = values.isValueShared();
        if (shared && (s.copies.length == 0 || s.copies[0].length() != values.dimension())) {
            var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
            s.copies = new VectorFloat<?>[4];
            for (int j = 0; j < 4; j++) {
                s.copies[j] = vts.createFloatVector(values.dimension());
            }
        }
        VectorFloat<?>[] targets = shared ? s.copies : s.targets;
        for (int i = 0; i < count; i += 4) {
            int n = Math.min(4, count - i);
            for (int j = 0; j < n; j++) {
                if (shared) {
                    values.getVectorInto(nodes[i + j], targets[j], 0);
                } else {
                    targets[j] = values.getVector(nodes[i + j]);
                }
            }
            vsf.compare(query, targets, n, s.scores);
            System.arraycopy(s.scores, 0, results, i, n);
        }
        if (!shared) {
            Arrays.fill(targets, null);   // keep no references to the source's vectors
        }
    }
}

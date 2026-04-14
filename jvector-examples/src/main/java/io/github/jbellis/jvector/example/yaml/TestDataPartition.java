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

package io.github.jbellis.jvector.example.yaml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Specifically for defining how data is partitioned for testing compaction.
 */
public class TestDataPartition {
    public List<Integer> numSplits;
    public List<Distribution> splitDistribution;

    public TestDataPartition() {
        this.numSplits = Collections.singletonList(1);
        this.splitDistribution = Collections.singletonList(Distribution.UNIFORM);
    }

    public TestDataPartition(int numSplits) {
        this.numSplits = Collections.singletonList(numSplits);
        this.splitDistribution = Collections.singletonList(Distribution.UNIFORM);
    }

    public enum Distribution {
        UNIFORM,
        FIBONACCI,
        LOG2N;

        public List<Integer> computeSplitSizes(int total, int numSplits) {
            int[] weights = new int[numSplits];
            switch (this) {
                case UNIFORM:
                    for (int i = 0; i < numSplits; i++) weights[i] = 1;
                    break;
                case FIBONACCI:
                    int a = 1, b = 2;
                    weights[0] = 1;
                    for (int i = 1; i < numSplits; i++) {
                        weights[i] = b;
                        int next = a + b;
                        a = b;
                        b = next;
                    }
                    break;
                case LOG2N:
                    for (int i = 0; i < numSplits; i++) weights[i] = 1 << i;
                    break;
            }

            long weightSum = 0;
            for (int w : weights) weightSum += w;

            List<Integer> sizes = new ArrayList<>(numSplits);
            int assigned = 0;
            for (int i = 0; i < numSplits; i++) {
                int size;
                if (i == numSplits - 1) {
                    size = total - assigned;
                } else {
                    size = (int) (((long) weights[i] * total) / weightSum);
                }
                sizes.add(size);
                assigned += size;
            }
            return sizes;
        }
    }
}

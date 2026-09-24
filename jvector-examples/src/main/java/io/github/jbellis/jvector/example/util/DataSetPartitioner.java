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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.yaml.TestDataPartition;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility for partitioning a DataSet into multiple contiguous segments based on a distribution.
 * <p>
 * Partitions are {@link RandomAccessVectorValues#range(int, int)} views over the base vectors, so no
 * vectors are copied. Partition {@code i} covers global ordinals {@code [sum(sizes[0..i)), sum(sizes[0..i]))},
 * which is the ordering compaction relies on to map partition-local ordinals back to global ones.
 */
public final class DataSetPartitioner {
    private DataSetPartitioner() {}

    public static final class PartitionedData {
        public final List<RandomAccessVectorValues> vectors;
        public final List<Integer> sizes;

        public PartitionedData(List<RandomAccessVectorValues> vectors, List<Integer> sizes) {
            this.vectors = vectors;
            this.sizes = sizes;
        }
    }

    public static PartitionedData partition(DataSet ds, int numParts, TestDataPartition.Distribution distribution) {
        return partition(ds.getBaseRavv(), numParts, distribution);
    }

    /**
     * Splits {@code baseVectors} into {@code numParts} contiguous ranged views sized by {@code distribution}.
     *
     * @param baseVectors  the vectors to partition
     * @param numParts     the number of partitions
     * @param distribution how to size the partitions
     * @return the partition views and their sizes, in partition order
     */
    public static PartitionedData partition(RandomAccessVectorValues baseVectors, int numParts, TestDataPartition.Distribution distribution) {
        List<Integer> sizes = distribution.computeSplitSizes(baseVectors.size(), numParts);
        List<RandomAccessVectorValues> parts = new ArrayList<>(numParts);

        int runningStart = 0;
        for (int size : sizes) {
            int start = runningStart;
            int end = start + size;
            runningStart = end;
            parts.add(baseVectors.range(start, end));
        }

        return new PartitionedData(parts, sizes);
    }
}

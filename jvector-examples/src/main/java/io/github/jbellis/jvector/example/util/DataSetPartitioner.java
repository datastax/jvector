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
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility for partitioning a DataSet into multiple segments based on a distribution.
 */
public final class DataSetPartitioner {
    private DataSetPartitioner() {}

    public static final class PartitionedData {
        public final List<List<VectorFloat<?>>> vectors;
        public final List<Integer> sizes;

        public PartitionedData(List<List<VectorFloat<?>>> vectors, List<Integer> sizes) {
            this.vectors = vectors;
            this.sizes = sizes;
        }
    }

    public static PartitionedData partition(DataSet ds, int numParts, TestDataPartition.Distribution distribution) {
        return partition(ds.getBaseVectors(), numParts, distribution);
    }

    public static PartitionedData partition(List<VectorFloat<?>> baseVectors, int numParts, TestDataPartition.Distribution distribution) {
        List<Integer> sizes = distribution.computeSplitSizes(baseVectors.size(), numParts);
        List<List<VectorFloat<?>>> parts = new ArrayList<>(numParts);

        int runningStart = 0;
        for (int size : sizes) {
            int start = runningStart;
            int end = start + size;
            runningStart = end;
            parts.add(new ArrayList<>(baseVectors.subList(start, end)));
        }

        return new PartitionedData(parts, sizes);
    }
}

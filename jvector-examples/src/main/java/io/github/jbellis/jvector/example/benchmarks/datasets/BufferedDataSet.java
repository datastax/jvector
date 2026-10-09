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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import io.github.jbellis.jvector.disk.FvecFileVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.nio.file.Path;
import java.util.AbstractList;
import java.util.List;

/** Resident queries/ground truth with bounded, file-backed base vectors. */
final class BufferedDataSet implements DataSet {
    private final String name;
    private final VectorSimilarityFunction similarity;
    private final FvecFileVectorValues base;
    private final List<VectorFloat<?>> queries;
    private final List<? extends List<Integer>> groundTruth;
    private final List<VectorFloat<?>> baseList;

    BufferedDataSet(String name, VectorSimilarityFunction similarity, Path file,
                    List<VectorFloat<?>> queries, List<? extends List<Integer>> groundTruth) throws IOException {
        this.name = name;
        this.similarity = similarity;
        this.queries = queries;
        this.groundTruth = groundTruth;
        this.base = FvecFileVectorValues.open(file);
        try {
            if (base.size() == 0 || queries.isEmpty() || queries.size() != groundTruth.size())
                throw new IllegalArgumentException("Empty or mismatched dataset facets for " + name);
            for (var query : queries)
                if (query.length() != base.dimension())
                    throw new IllegalArgumentException("Base/query dimensions differ for " + name);
            var views = base.threadLocalSupplier();
            // Compatibility list is lazy; retained entries must not alias a view's scratch vector.
            baseList = new AbstractList<>() {
                @Override public VectorFloat<?> get(int ordinal) { return views.get().getVector(ordinal).copy(); }
                @Override public int size() { return base.size(); }
            };
        } catch (RuntimeException | Error failure) {
            try { base.close(); } catch (IOException closeFailure) { failure.addSuppressed(closeFailure); }
            throw failure;
        }
        System.out.printf("%n%s: %d base and %d query vectors loaded, dimensions %d (buffered)%n",
                name, base.size(), queries.size(), base.dimension());
    }

    @Override public int getDimension() { return base.dimension(); }
    @Override public RandomAccessVectorValues getBaseRavv() { return base; }
    @Override public String getName() { return name; }
    @Override public VectorSimilarityFunction getSimilarityFunction() { return similarity; }
    @Override public List<VectorFloat<?>> getBaseVectors() { return baseList; }
    @Override public List<VectorFloat<?>> getQueryVectors() { return queries; }
    @Override public List<? extends List<Integer>> getGroundTruth() { return groundTruth; }
    @Override public void close() throws IOException { base.close(); }
}

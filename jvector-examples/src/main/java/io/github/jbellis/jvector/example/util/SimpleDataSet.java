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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

public class SimpleDataSet implements DataSet {
    private final String name;
    private final VectorSimilarityFunction similarityFunction;
    private final List<VectorFloat<?>> baseVectors;
    private final List<VectorFloat<?>> queryVectors;
    private final List<? extends List<Integer>> groundTruth;
    private RandomAccessVectorValues baseRavv;

    public SimpleDataSet(String name,
                         VectorSimilarityFunction similarityFunction,
                         List<VectorFloat<?>> baseVectors,
                         List<VectorFloat<?>> queryVectors,
                         List<? extends List<Integer>> groundTruth)
    {
        if (baseVectors.isEmpty()) {
            throw new IllegalArgumentException("Base vectors must not be empty");
        }
        if (queryVectors.isEmpty()) {
            throw new IllegalArgumentException("Query vectors must not be empty");
        }
        if (groundTruth.isEmpty()) {
            throw new IllegalArgumentException("Ground truth vectors must not be empty");
        }

        if (baseVectors.get(0).length() != queryVectors.get(0).length()) {
            throw new IllegalArgumentException("Base and query vectors must have the same dimensionality");
        }
        if (queryVectors.size() != groundTruth.size()) {
            throw new IllegalArgumentException("Query and ground truth lists must be the same size");
        }

        this.name = name;
        this.similarityFunction = similarityFunction;
        this.baseVectors = baseVectors;
        this.queryVectors = queryVectors;
        this.groundTruth = groundTruth;

        System.out.format("%n%s: %d base and %d query vectors created, dimensions %d%n",
                name, baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());
    }

    @Override
    public int getDimension() {
        return getBaseVectors().get(0).length();
    }

    @Override
    public RandomAccessVectorValues getBaseRavv() {
        if (baseRavv == null) {
            baseRavv = new ListRandomAccessVectorValues(getBaseVectors(), getDimension());
        }
        return baseRavv;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public VectorSimilarityFunction getSimilarityFunction() {
        return similarityFunction;
    }

    @Override
    public List<VectorFloat<?>> getBaseVectors() {
        return baseVectors;
    }

    @Override
    public List<VectorFloat<?>> getQueryVectors() {
        return queryVectors;
    }

    @Override
    public List<? extends List<Integer>> getGroundTruth() {
        return groundTruth;
    }
}

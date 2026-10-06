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

import io.github.jbellis.jvector.graph.ListRandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessByteVectorValues;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.ByteSequence;

import java.util.List;

/**
 * An int8 dataset backed by in-memory {@link ByteSequence} lists.
 *
 * <p>Type-specific accessors ({@link #getBaseByteRavv()}, {@link #getByteSimilarityFunction()})
 * live here rather than on the {@link DataSet} interface.
 */
public class ByteDataSet implements DataSet<ByteSequence<?>> {
    private final String name;
    private final ByteVectorSimilarityFunction similarityFunction;
    private final List<ByteSequence<?>> baseVectors;
    private final List<ByteSequence<?>> queryVectors;
    private final List<? extends List<Integer>> groundTruth;
    private RandomAccessByteVectorValues baseRavv;

    public ByteDataSet(String name,
                       ByteVectorSimilarityFunction similarityFunction,
                       List<ByteSequence<?>> baseVectors,
                       List<ByteSequence<?>> queryVectors,
                       List<? extends List<Integer>> groundTruth)
    {
        if (baseVectors.isEmpty()) {
            throw new IllegalArgumentException("Base vectors must not be empty");
        }
        if (queryVectors.isEmpty()) {
            throw new IllegalArgumentException("Query vectors must not be empty");
        }
        if (groundTruth.isEmpty()) {
            throw new IllegalArgumentException("Ground truth must not be empty");
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
    public String getName() {
        return name;
    }

    @Override
    public int getDimension() {
        return baseVectors.get(0).length();
    }

    @Override
    public List<ByteSequence<?>> getBaseVectors() {
        return baseVectors;
    }

    @Override
    public List<ByteSequence<?>> getQueryVectors() {
        return queryVectors;
    }

    @Override
    public List<? extends List<Integer>> getGroundTruth() {
        return groundTruth;
    }

    public RandomAccessByteVectorValues getBaseByteRavv() {
        if (baseRavv == null) {
            baseRavv = new ListRandomAccessByteVectorValues(baseVectors, getDimension());
        }
        return baseRavv;
    }

    public ByteVectorSimilarityFunction getByteSimilarityFunction() {
        return similarityFunction;
    }
}

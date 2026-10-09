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

package io.github.jbellis.jvector.management;

/**
 * The compression a graph builder applies to the vectors before building, so that the graph is built
 * with compressed (approximate) scores. See {@code HnswIndexBuilder.withCompressionType}.
 */
public enum CompressionType {
    /** No compression: the graph is built with exact scores. */
    NONE("None"),
    /** Product quantization ({@code ProductQuantization}). */
    PQ("PQ"),
    /** Binary quantization ({@code BinaryQuantization}), one bit per dimension. */
    BQ("BQ"),
    /**
     * Asymmetric hashing ({@code AsymmetricHashing}). Supports
     * {@code VectorSimilarityFunction.DOT_PRODUCT} only, and trains a single landmark, which ASH
     * construction scoring requires; see {@code AsymmetricHashing} for its other limitations.
     */
    ASH("ASH");

    private final String type;

    CompressionType(String type) {
        this.type = type;
    }

    public String getType() {
        return type;
    }
}

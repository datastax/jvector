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

import io.github.jbellis.jvector.quantization.BinaryQuantization;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;

/**
 * Base class for compressor parameters.
 */
public abstract class CompressorParameters {
    /**
     * Constructs a CompressorParameters.
     */
    public CompressorParameters() {}

    /** No compression constant. */
    public static final CompressorParameters NONE = new NoCompressionParameters();

    /**
     * Checks if this compressor supports caching.
     * @return true if caching is supported
     */
    public boolean supportsCaching() {
        return false;
    }

    /**
     * Gets the ID string for the specified dataset.
     * @param ds the dataset
     * @return the ID string
     */
    public String idStringFor(DataSet ds) {
        // only required when supportsCaching() is true
        throw new UnsupportedOperationException();
    }

    /**
     * Computes the compressor for the specified dataset.
     * @param ds the dataset
     * @return the vector compressor
     */
    public abstract VectorCompressor<?> computeCompressor(DataSet ds);

    /**
     * Product quantization parameters.
     */
    public static class PQParameters extends CompressorParameters {
        private final int m;
        private final int k;
        private final boolean isCentered;
        private final float anisotropicThreshold;

        /**
         * Constructs PQParameters.
         * @param m the m parameter
         * @param k the k parameter
         * @param isCentered the isCentered parameter
         * @param anisotropicThreshold the anisotropicThreshold parameter
         */
        public PQParameters(int m, int k, boolean isCentered, float anisotropicThreshold) {
            this.m = m;
            this.k = k;
            this.isCentered = isCentered;
            this.anisotropicThreshold = anisotropicThreshold;
        }

        @Override
        public VectorCompressor<?> computeCompressor(DataSet ds) {
            return ProductQuantization.compute(ds.getBaseRavv(), m, k, isCentered, anisotropicThreshold);
        }

        @Override
        public String idStringFor(DataSet ds) {
            return String.format("PQ_%s_%d_%d_%s_%s", ds.name, m, k, isCentered, anisotropicThreshold);
        }

        @Override
        public boolean supportsCaching() {
            return true;
        }
    }

    /**
     * Binary quantization parameters.
     */
    public static class BQParameters extends CompressorParameters {
        /**
         * Constructs BQParameters.
         */
        public BQParameters() {}
        @Override
        public VectorCompressor<?> computeCompressor(DataSet ds) {
            return new BinaryQuantization(ds.getDimension());
        }
    }

    /**
     * NVQ parameters.
     */
    public static class NVQParameters extends CompressorParameters {
        private final int nSubVectors;

        /**
         * Constructs NVQParameters.
         * @param nSubVectors the number of sub-vectors
         */
        public NVQParameters(int nSubVectors) {
            this.nSubVectors = nSubVectors;
        }

        @Override
        public VectorCompressor<?> computeCompressor(DataSet ds) {
            return NVQuantization.compute(ds.getBaseRavv(), nSubVectors);
        }

        @Override
        public String idStringFor(DataSet ds) {
            return String.format("NVQ_%s_%d_%s", ds.name, nSubVectors);
        }

        @Override
        public boolean supportsCaching() {
            return true;
        }
    }

    private static class NoCompressionParameters extends CompressorParameters {
        @Override
        public VectorCompressor<?> computeCompressor(DataSet ds) {
            return null;
        }
    }
}


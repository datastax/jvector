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

public abstract class CompressorParameters {
    public static final CompressorParameters NONE = new NoCompressionParameters();

    public boolean supportsCaching() {
        return false;
    }

    public String idStringFor(DataSet ds) {
        // only required when supportsCaching() is true
        throw new UnsupportedOperationException();
    }

    public abstract VectorCompressor<?> computeCompressor(DataSet ds);

    public static class PQParameters extends CompressorParameters {
        private final int m;
        private final int k;
        private final boolean isCentered;
        private final float anisotropicThreshold;

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
            return String.format("PQ_%s_%d_%d_%s_%s", ds.getName(), m, k, isCentered, anisotropicThreshold);
        }

        @Override
        public boolean supportsCaching() {
            return true;
        }
    }

    public static class BQParameters extends CompressorParameters {
        @Override
        public VectorCompressor<?> computeCompressor(DataSet ds) {
            return new BinaryQuantization(ds.getDimension());
        }
    }

    public static class NVQParameters extends CompressorParameters {
        private final int nSubVectors;

        public NVQParameters(int nSubVectors) {
            this.nSubVectors = nSubVectors;
        }

        @Override
        public VectorCompressor<?> computeCompressor(DataSet ds) {
            return NVQuantization.compute(ds.getBaseRavv(), nSubVectors);
        }

        @Override
        public String idStringFor(DataSet ds) {
            return String.format("NVQ_%s_%d_%s", ds.getName(), nSubVectors);
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


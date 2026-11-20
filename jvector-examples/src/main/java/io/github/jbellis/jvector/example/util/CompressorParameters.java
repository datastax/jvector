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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RemappedRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.BinaryQuantization;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.quantization.VectorCompressor;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.util.concurrent.ForkJoinTask;

public abstract class CompressorParameters {
    public static final CompressorParameters NONE = new NoCompressionParameters();

    public boolean supportsCaching() {
        return false;
    }

    public String idStringFor(DataSet ds) {
        // only required when supportsCaching() is true
        throw new UnsupportedOperationException();
    }

    /**
     * Reads back a compressor that {@link #computeCompressor} produced and that was saved with
     * {@link VectorCompressor#write}. Only required when {@link #supportsCaching()} is true.
     */
    public VectorCompressor<?> loadCompressor(RandomAccessReader in) throws IOException {
        throw new UnsupportedOperationException();
    }

    public abstract VectorCompressor<?> computeCompressor(DataSet ds);

    /** Validate dataset-dependent constraints before allocating codes or training a model. */
    public void validateFor(DataSet ds, boolean fused, String ashConstructionMode) {}

    /**
     * Whether {@code compressor} was made with these parameters, so codes it encoded can stand in for codes these
     * parameters would produce. False unless a subclass can tell from the compressor itself.
     */
    public boolean describes(VectorCompressor<?> compressor) {
        return false;
    }

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
        public void validateFor(DataSet ds, boolean fused, String ashConstructionMode) {
            if (m < 1 || m > ds.getDimension()) {
                throw new IllegalArgumentException("PQ m must be between 1 and the dataset dimension");
            }
            if (k < 1 || k > 256 || k > ds.getBaseRavv().size()) {
                throw new IllegalArgumentException("PQ k must be between 1 and 256 and fit the training set");
            }
            if (fused && k != 256) throw new IllegalArgumentException("Fused PQ requires k=256");
            if (!Float.isFinite(anisotropicThreshold) || anisotropicThreshold < -1 || anisotropicThreshold >= 1) {
                throw new IllegalArgumentException("PQ anisotropicThreshold must be finite and in [-1, 1)");
            }
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
        public VectorCompressor<?> loadCompressor(RandomAccessReader in) throws IOException {
            return ProductQuantization.load(in);
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
            return String.format("NVQ_%s_%d", ds.getName(), nSubVectors);
        }

        @Override
        public VectorCompressor<?> loadCompressor(RandomAccessReader in) throws IOException {
            return NVQuantization.load(in);
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
    public static class ASHParameters extends CompressorParameters {
        private final int optimizer;
        private final int encodedBits;
        private final int landmarkCount;
        private final int bitsPerDimension;
        // Training window over the base vectors; trainingCount < 0 trains on the whole set.
        private final int trainingStart;
        private final int trainingCount;

        public ASHParameters(int optimizer, int encodedBits, int landmarkCount) {
            this(optimizer, encodedBits, landmarkCount, 1);
        }

        public ASHParameters(int optimizer, int encodedBits, int landmarkCount, int bitsPerDimension) {
            this(optimizer, encodedBits, landmarkCount, bitsPerDimension, 0, -1);
        }

        /**
         * @param trainingStart first base ordinal of the training window
         * @param trainingCount number of base vectors in the training window, or a negative value
         *                      to train on the whole set
         */
        public ASHParameters(int optimizer, int encodedBits, int landmarkCount, int bitsPerDimension,
                             int trainingStart, int trainingCount) {
            this.optimizer = optimizer;
            this.encodedBits = encodedBits;
            this.landmarkCount = landmarkCount;
            this.bitsPerDimension = bitsPerDimension;
            this.trainingStart = trainingStart;
            this.trainingCount = trainingCount;
        }

        /** ASH code size, including the header. */
        public int encodedBits() {
            return encodedBits;
        }

        public int bitsPerDimension() {
            return bitsPerDimension;
        }

        private boolean hasTrainingWindow() {
            return trainingCount >= 0;
        }

        /** The base vectors the model trains on: the whole set, or the training window as a view over it. */
        private RandomAccessVectorValues trainingVectors(DataSet ds) {
            var base = ds.getBaseRavv();
            if (!hasTrainingWindow()) return base;
            int[] ordinals = new int[trainingCount];
            for (int i = 0; i < trainingCount; i++) ordinals[i] = trainingStart + i;
            return new RemappedRandomAccessVectorValues(base, ordinals);
        }

        @Override
        public VectorCompressor<?> loadCompressor(RandomAccessReader in) throws IOException {
            return AsymmetricHashing.load(in);
        }

        @Override
        public boolean describes(VectorCompressor<?> compressor) {
            if (!(compressor instanceof AsymmetricHashing)) return false;
            // A model does not record the window it was trained on, so a windowed model is never matched.
            if (hasTrainingWindow()) return false;
            var ash = (AsymmetricHashing) compressor;
            return ash.optimizer == optimizer && ash.encodedBits == encodedBits
                    && ash.landmarkCount == landmarkCount && ash.bitsPerDimension == bitsPerDimension;
        }

        @Override
        public void validateFor(DataSet ds, boolean fused, String ashConstructionMode) {
            if (ds.getSimilarityFunction() == VectorSimilarityFunction.COSINE) {
                throw new IllegalArgumentException("ASH requires dot-product or Euclidean similarity");
            }
            int payload = encodedBits - AsymmetricHashing.HEADER_BITS;
            if (encodedBits <= AsymmetricHashing.HEADER_BITS || payload % bitsPerDimension != 0
                    || payload / bitsPerDimension > ds.getDimension()) {
                throw new IllegalArgumentException("ASH encodedBits must yield a whole projected dimension count from 1 to the dataset dimension");
            }
            if (optimizer != AsymmetricHashing.ITQ && optimizer != AsymmetricHashing.RANDOM) {
                throw new IllegalArgumentException("ASH optimizer must be ITQ or RANDOM");
            }
            if (landmarkCount < 1 || landmarkCount > 256 || landmarkCount > ds.getBaseRavv().size()) {
                throw new IllegalArgumentException("ASH landmarkCount must be between 1 and 256 and fit the training set");
            }
            if (hasTrainingWindow()) {
                long end = (long) trainingStart + trainingCount;
                if (trainingStart < 0 || trainingCount < 1 || end > ds.getBaseRavv().size()) {
                    throw new IllegalArgumentException("ASH trainingStart and trainingCount must select 1 or more base vectors inside the dataset");
                }
                if (landmarkCount > trainingCount) {
                    throw new IllegalArgumentException("ASH landmarkCount must not exceed trainingCount");
                }
            }
            if (ashConstructionMode != null) {
                switch (ashConstructionMode) {
                    case "symmetric":
                    case "symmetric-all":
                    case "symmetric-scalar":
                        if (landmarkCount != 1) {
                            throw new IllegalArgumentException("Symmetric ASH construction scoring requires one landmark");
                        }
                        break;
                    case "asymmetric":
                    case "asymmetric-cached":
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown ASH construction mode: " + ashConstructionMode);
                }
            }
        }

        @Override
        public VectorCompressor<?> computeCompressor(DataSet ds) {
            try {
                // Native BLAS uses the construction pool's size when training runs in it,
                // otherwise the library's physical-core pool.
                var executor = ForkJoinTask.inForkJoinPool() ? ForkJoinTask.getPool() : PhysicalCoreExecutor.pool();
                return AsymmetricHashing.initialize(
                        trainingVectors(ds),
                        ds.getSimilarityFunction(),
                        optimizer,
                        encodedBits,
                        landmarkCount,
                        bitsPerDimension,
                        executor);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public String idStringFor(DataSet ds) {
            return String.format(
                    "ASH_%s_%s_v%d_opt%d_bits%d_bpd%d_C%d",
                    ds.getName(),
                    OnDiskGraphIndexCache.similarityTag(ds.getSimilarityFunction()),
                    io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex.CURRENT_VERSION,
                    optimizer,
                    encodedBits,
                    bitsPerDimension,
                    landmarkCount
            ) + (hasTrainingWindow() ? String.format("_train%d+%d", trainingStart, trainingCount) : "");
        }

        @Override
        public boolean supportsCaching() {
            return true;
        }
    }
}


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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;


/**
 * Non-uniform Vector Quantization for float vectors.
 * It divides each vector in subvectors and then quantizes each one individually using a non-uniform quantizer.
 */
public class NVQuantization implements VectorCompressor<NVQuantization.QuantizedVector>, Accountable {
    public enum BitsPerDimension {
        EIGHT {
            @Override
            public int getInt() {
                return 8;
            }

            @Override
            public ByteSequence<?> createByteSequence(int nDimensions) {
                return vectorTypeSupport.createByteSequence(nDimensions);
            }
        },
        FOUR {
            @Override
            public int getInt() {
                return 4;
            }

            @Override
            public ByteSequence<?> createByteSequence(int nDimensions) {
                return vectorTypeSupport.createByteSequence((int) Math.ceil(nDimensions / 2.));
            }
        };

        /**
         * Writes the BitsPerDimension to DataOutput.
         * @param out the DataOutput into which to write the object
         * @throws IOException if there is a problem writing to out.
         */
        public void write(DataOutput out) throws IOException {
            out.writeInt(getInt());
        }

        /**
         * Returns the integer 4 for FOUR and 8 for EIGHT
         */
        public abstract int getInt();

        /**
         * Creates a ByteSequence of the proper length to store the quantized vector.
         * @param nDimensions the number of dimensions of the original vector
         * @return the byte sequence
         */
        public abstract ByteSequence<?> createByteSequence(int nDimensions);

        /**
         * Loads the BitsPerDimension from a RandomAccessReader.
         * @param in the RandomAccessReader to read from.
         * @throws IOException if there is a problem reading from in.
         */
        public static BitsPerDimension load(RandomAccessReader in) throws IOException {
            int nBitsPerDimension = in.readInt();
            switch (nBitsPerDimension) {
                case 8:
                    return BitsPerDimension.EIGHT;
                default:
                    throw new IllegalArgumentException("Unsupported BitsPerDimension " + nBitsPerDimension);
            }
        }
    }

    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    // How many bits to use for each dimension when quantizing the vector:
    public final BitsPerDimension bitsPerDimension;

    // We subtract the global mean vector to make it robust against center datasets with a large mean:
    public final VectorFloat<?> globalMean;

    // The number of dimensions of the original (uncompressed) vectors:
    public final int originalDimension;

    // A matrix that stores the size and starting point of each subvector:
    public final int[][] subvectorSizesAndOffsets;

    // Whether we want to skip the optimization of the NVQ parameters. Here for debug purposes only.
    @VisibleForTesting
    public boolean learn = true;

    /**
     * Class constructor.
     * @param subvectorSizesAndOffsets a matrix that stores the size and starting point of each subvector
     * @param globalMean the mean of the database (its average vector)
     */
    private NVQuantization(int[][] subvectorSizesAndOffsets, VectorFloat<?> globalMean) {
        this.bitsPerDimension = BitsPerDimension.EIGHT;
        this.globalMean = globalMean;
        this.subvectorSizesAndOffsets = subvectorSizesAndOffsets;
        this.originalDimension = Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum();

        if (globalMean.length() != originalDimension) {
            var msg = String.format("Global mean length %d does not match vector dimensionality %d", globalMean.length(), originalDimension);
            throw new IllegalArgumentException(msg);
        }
    }

    /**
     * Computes the global mean vector and the data structures used to divide each vector into subvectors.
     *
     * @param ravv the vectors to quantize
     * @param nSubVectors number of subvectors
     */
    public static NVQuantization compute(RandomAccessVectorValues ravv, int nSubVectors) {
        var ravvCopy = ravv.threadLocalSupplier().get();
        var dim = ravvCopy.getVector(0).length();
        var globalMean = vectorTypeSupport.createFloatVector(dim);
        for (int i = 0; i < ravvCopy.size(); i++) {
            VectorUtil.addInPlace(globalMean, ravvCopy.getVector(i));
        }
        VectorUtil.scale(globalMean, 1.0f / ravvCopy.size());
        return create(globalMean, nSubVectors);
    }

    /**
     * Creates a NVQuantization instance by using the global mean and computing the data structures used to divide each
     * vector into subvectors.
     *
     * @param scaledGlobalMean the mean of the graph (its average vector)
     * @param nSubVectors number of subvectors
     */
    public static NVQuantization create(VectorFloat<?> scaledGlobalMean, int nSubVectors) {
        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(scaledGlobalMean.length(), nSubVectors);
        return new NVQuantization(subvectorSizesAndOffsets, scaledGlobalMean);
    }

    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        return new NVQVectors(this, (QuantizedVector[]) compressedVectors);
    }

    /**
     * Encodes the given vectors in parallel using NVQ.
     */
    @Override
    public NVQVectors encodeAll(RandomAccessVectorValues ravv, ForkJoinPool parallelExecutor) {
        var ravvCopy = ravv.threadLocalSupplier();
        return new NVQVectors(this,
                parallelExecutor.submit(() -> IntStream.range(0, ravv.size())
                                .parallel()
                                .mapToObj(i -> {
                                    var localRavv = ravvCopy.get();
                                    VectorFloat<?> v = localRavv.getVector(i);
                                    return encode(v);
                                })
                                .toArray(QuantizedVector[]::new))
                        .join());
    }

    /**
     * Encodes the input vector using NVQ.
     * @return one subvector per subspace
     */
    @Override
    public QuantizedVector encode(VectorFloat<?> vector) {
        var qv = QuantizedVector.createEmpty(subvectorSizesAndOffsets, bitsPerDimension);
        encodeTo(vector, qv);
        return qv;
    }

    /**
     * Encodes the input vector using NVQ into dest
     */
    @Override
    public void encodeTo(VectorFloat<?> v, NVQuantization.QuantizedVector dest) {
        var tempVector = VectorUtil.sub(v, globalMean);
        QuantizedVector.quantizeTo(getSubVectors(tempVector), bitsPerDimension, learn, dest);
    }

    /**
     * Creates an array of subvectors from a given vector
     */
    public VectorFloat<?>[] getSubVectors(VectorFloat<?> vector) {
        VectorFloat<?>[] subvectors = new VectorFloat<?>[subvectorSizesAndOffsets.length];

        // Iterate through the subvectorSizesAndOffsets to create each subvector and copy slices into them
        for (int i = 0; i < subvectorSizesAndOffsets.length; i++) {
            int size = subvectorSizesAndOffsets[i][0];   // Size of the subvector
            int offset = subvectorSizesAndOffsets[i][1]; // Offset from the start of the input vector
            VectorFloat<?> subvector = vectorTypeSupport.createFloatVector(size);
            subvector.copyFrom(vector, offset, 0, size);
            subvectors[i] = subvector;
        }
        return subvectors;
    }

    /**
     * Splits the vector dimension into M subvectors of roughly equal size.
     */
    static int[][] getSubvectorSizesAndOffsets(int dimensions, int M) {
        if (M > dimensions) {
            throw new IllegalArgumentException("Number of subspaces must be less than or equal to the vector dimension");
        }
        int[][] sizes = new int[M][2];
        int baseSize = dimensions / M;
        int remainder = dimensions % M;
        // distribute the remainder among the subvectors
        int offset = 0;
        for (int i = 0; i < M; i++) {
            int size = baseSize + (i < remainder ? 1 : 0);
            sizes[i] = new int[]{size, offset};
            offset += size;
        }
        return sizes;
    }

    /**
     * Writes the instance to a DataOutput.
     * @param out DataOutput to write to
     * @param version serialization version.
     * @throws IOException fails if we cannot write to the DataOutput
     */
    public void write(DataOutput out, int version) throws IOException
    {
        if (version > OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("Unsupported serialization version " + version);
        }

        out.writeInt(version);

        out.writeInt(globalMean.length());
        vectorTypeSupport.writeFloatVector(out, globalMean);

        bitsPerDimension.write(out);

        out.writeInt(subvectorSizesAndOffsets.length);
        assert Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum() == originalDimension;
        for (var a : subvectorSizesAndOffsets) {
            out.writeInt(a[0]);
        }
    }

    /**
     * Returns the size in bytes of this class when writing it using the write method.
     * @return the size in bytes
     */
    @Override
    public int compressorSize() {
        int size = 0;
        size += Integer.BYTES; // STORAGE_VERSION
        size += Integer.BYTES; // globalCentroidLength
        size += Float.BYTES * globalMean.length();
        size += Integer.BYTES; // bitsPerDimension
        size += Integer.BYTES; // nSubVectors
        size += subvectorSizesAndOffsets.length * Integer.BYTES;
        return size;
    }

    /**
     * Loads an instance from a RandomAccessReader
     * @param in the RandomAccessReader
     * @return the instance
     * @throws IOException fails if we cannot read from the RandomAccessReader
     */
    public static NVQuantization load(RandomAccessReader in) throws IOException {
        int version = in.readInt();
        int globalMeanLength = in.readInt();

        VectorFloat<?> globalMean = null;
        if (globalMeanLength > 0) {
            globalMean = vectorTypeSupport.readFloatVector(in, globalMeanLength);
        }

        BitsPerDimension bitsPerDimension = BitsPerDimension.load(in);

        int nSubVectors = in.readInt();
        int[][] subvectorSizes = new int[nSubVectors][];
        int offset = 0;
        for (int i = 0; i < nSubVectors; i++) {
            subvectorSizes[i] = new int[2];
            int size = in.readInt();
            subvectorSizes[i][0] = size;
            subvectorSizes[i][1] = offset;
            offset += size;
        }

        return new NVQuantization(subvectorSizes, globalMean);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        NVQuantization that = (NVQuantization) o;
        return originalDimension == that.originalDimension
                && Objects.equals(globalMean, that.globalMean)
                && Arrays.deepEquals(subvectorSizesAndOffsets, that.subvectorSizesAndOffsets);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(originalDimension);
        result = 31 * result + Objects.hashCode(globalMean);
        result = 31 * result + Arrays.deepHashCode(subvectorSizesAndOffsets);
        return result;
    }

    @Override
    public int compressedVectorSize() {
        int size = Integer.BYTES; // number of subvectors
        for (int[] subvectorSizesAndOffset : subvectorSizesAndOffsets) {
            size += QuantizedSubVector.compressedVectorSize(subvectorSizesAndOffset[0], bitsPerDimension);
        }
        return size;
    }

    @Override
    public long ramBytesUsed() {
        return globalMean.ramBytesUsed();
    }

    @Override
    public String toString() {
        return String.format("NVQuantization(sub-vectors=%d)", subvectorSizesAndOffsets.length);
    }

    /**
     * A NuVeQ vector.
     */
    public static class QuantizedVector {
        public final QuantizedSubVector[] subVectors;

        /**
         * Class constructor.
         * @param subVectors receives the subvectors to quantize
         * @param bitsPerDimension the number of bits per dimension
         * @param learn whether to use optimization to find the parameters of the nonlinearity
         */
        public static void quantizeTo(VectorFloat<?>[] subVectors, BitsPerDimension bitsPerDimension, boolean learn, QuantizedVector dest) {
            for (int i = 0; i < subVectors.length; i++) {
                QuantizedSubVector.quantizeTo(subVectors[i], bitsPerDimension, learn, dest.subVectors[i]);
            }
        }

        /**
         * Constructs an instance from existing subvectors. Used when loading from a RandomAccessReader.
         * @param subVectors the subvectors
         */
        private QuantizedVector(QuantizedSubVector[] subVectors) {
            this.subVectors = subVectors;
        }

        /**
         * Create an empty instance. Meant to be used as scratch space in conjunction with loadInto
         * @param subvectorSizesAndOffsets the array containing the sizes for the subvectors
         * @param bitsPerDimension the number of bits per dimension
         */
        public static QuantizedVector createEmpty(int[][] subvectorSizesAndOffsets, BitsPerDimension bitsPerDimension) {
            var subVectors = new QuantizedSubVector[subvectorSizesAndOffsets.length];
            for (int i = 0; i < subvectorSizesAndOffsets.length; i++) {
                subVectors[i] = QuantizedSubVector.createEmpty(bitsPerDimension, subvectorSizesAndOffsets[i][0]);
            }
            return new QuantizedVector(subVectors);
        }


        /**
         * Write the instance to a DataOutput
         * @param out the DataOutput
         * @throws IOException fails if we cannot write to the DataOutput
         */
        public void write(DataOutput out) throws IOException {
            out.writeInt(subVectors.length);

            for (var sv : subVectors) {
                sv.write(out);
            }
        }

        /**
         * Read the instance from a RandomAccessReader
         * @param in the RandomAccessReader
         * @throws IOException fails if we cannot read from the RandomAccessReader
         */
        public static QuantizedVector load(RandomAccessReader in) throws IOException {
            int length = in.readInt();
            var subVectors = new QuantizedSubVector[length];
            for (int i = 0; i < length; i++) {
                subVectors[i] = QuantizedSubVector.load(in);
            }

            return new QuantizedVector(subVectors);
        }

        /**
         * Read the instance from a RandomAccessReader
         * @param in the RandomAccessReader
         * @throws IOException fails if we cannot read from the RandomAccessReader
         */
        public static void loadInto(RandomAccessReader in, QuantizedVector qvector) throws IOException {
            in.readInt();
            for (int i = 0; i < qvector.subVectors.length; i++) {
                QuantizedSubVector.loadInto(in, qvector.subVectors[i]);
            }
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QuantizedVector that = (QuantizedVector) o;
            return Arrays.deepEquals(subVectors, that.subVectors);
        }
    }

    /**
     * A NuVeQ sub-vector.
     */
    public static class QuantizedSubVector {
        // The byte sequence that stores the quantized subvector
        public ByteSequence<?> bytes;

        // The number of bits for each dimension of the input uncompressed subvector
        public BitsPerDimension bitsPerDimension;

        // The NVQ parameters
        public float growthRate;
        public float midpoint;
        public float maxValue;
        public float minValue;

        // The number of dimensions of the input uncompressed subvector
        public int originalDimensions;

        /**
         * Return the number of bytes occupied by the serialization of a QuantizedSubVector
         * @param nDims the number fof dimensions of the subvector
         * @param bitsPerDimension the number of bits per dimensions
         * @return the size in bytes of the quantized subvector
         */
        public static int compressedVectorSize(int nDims, BitsPerDimension bitsPerDimension) {
            // Here we assume that an enum takes 4 bytes
            switch (bitsPerDimension) {
                case EIGHT: return nDims + 4 * Float.BYTES + 3 * Integer.BYTES;
                default: throw new IllegalArgumentException("Unsupported bits per dimension: " + bitsPerDimension);
            }
        }

        /**
         * Quantize the vector using NVQ into dest
         * @param vector the subvector to quantize
         * @param bitsPerDimension the number of bits per dimension
         * @param learn whether to use optimization to find the parameters of the nonlinearity
         * @param dest the destination subvector
         */
        public static void quantizeTo(VectorFloat<?> vector, BitsPerDimension bitsPerDimension, boolean learn, QuantizedSubVector dest) {
            var minValue = VectorUtil.min(vector);
            var maxValue = VectorUtil.max(vector);

            //-----------------------------------------------------------------
            // Optimization to find the hyperparameters of the quantization
            float growthRate = 1e-2f;
            float midpoint = 0;

            if (learn) {
                NonuniformQuantizationLossFunction lossFunction = new NonuniformQuantizationLossFunction(bitsPerDimension);
                lossFunction.setVector(vector, minValue, maxValue);

                float growthRateCoarse = 1e-2f;
                float bestLossValue = Float.MIN_VALUE;
                float[] tempSolution = {growthRateCoarse, 0.f};
                for (float gr = 1e-6f; gr < 20.f; gr += 1f) {
                    tempSolution[0] = gr;
                    float lossValue = lossFunction.compute(tempSolution);
                    if (lossValue > bestLossValue) {
                        bestLossValue = lossValue;
                        growthRateCoarse = gr;
                    }
                }
                float growthRateFineTuned = growthRateCoarse;
                for (float gr = growthRateCoarse - 1; gr < growthRateCoarse + 1; gr += 0.1f) {
                    tempSolution[0] = gr;
                    float lossValue = lossFunction.compute(tempSolution);
                    if (lossValue > bestLossValue) {
                        bestLossValue = lossValue;
                        growthRateFineTuned = gr;
                    }
                }

                growthRate = growthRateFineTuned;
            }
            //---------------------------------------------------------------------------

            var quantized = bitsPerDimension.createByteSequence(vector.length());
            switch (bitsPerDimension) {
                case EIGHT:
                    VectorUtil.nvqQuantize8bit(vector, growthRate, midpoint, minValue, maxValue, quantized);
                    break;
                default:
                    throw new IllegalArgumentException("Unsupported bits per dimension: " + bitsPerDimension);
            }

            dest.bitsPerDimension = bitsPerDimension;
            dest.minValue = minValue;
            dest.maxValue = maxValue;
            dest.growthRate = growthRate;
            dest.midpoint = midpoint;
            dest.bytes = quantized;
            dest.originalDimensions = vector.length();
        }

        /**
         * Constructor used when loading from a RandomAccessReader. It takes its member fields.
         */
        private QuantizedSubVector(ByteSequence<?> bytes, int originalDimensions, BitsPerDimension bitsPerDimension,
                                   float minValue, float maxValue,
                                   float growthRate, float midpoint) {
            this.bitsPerDimension = bitsPerDimension;
            this.bytes = bytes;
            this.minValue = minValue;
            this.maxValue = maxValue;
            this.growthRate = growthRate;
            this.midpoint = midpoint;
            this.originalDimensions = originalDimensions;
        }

        /**
         * Write the instance to a DataOutput
         * @param out the DataOutput
         * @throws IOException fails if we cannot write to the DataOutput
         */
        public void write(DataOutput out) throws IOException {
            bitsPerDimension.write(out);
            out.writeFloat(minValue);
            out.writeFloat(maxValue);
            out.writeFloat(growthRate);
            out.writeFloat(midpoint);
            out.writeInt(originalDimensions);
            out.writeInt(bytes.length());

            vectorTypeSupport.writeByteSequence(out, bytes);
        }

        /**
         * Create an empty instance. Meant to be used as scratch space in conjunction with loadInto
         * @param bitsPerDimension the number of bits per dimension
         * @param length the number of dimensions
         */
        public static QuantizedSubVector createEmpty(BitsPerDimension bitsPerDimension, int length) {
            ByteSequence<?> bytes = bitsPerDimension.createByteSequence(length);
            return new QuantizedSubVector(bytes, length, bitsPerDimension, 0.f, 0.f, 0.f, 0.f);
        }

        /**
         * Read the instance from a RandomAccessReader
         * @param in the RandomAccessReader
         * @throws IOException fails if we cannot read from the RandomAccessReader
         */
        public static QuantizedSubVector load(RandomAccessReader in) throws IOException {
            BitsPerDimension bitsPerDimension = BitsPerDimension.load(in);
            float minValue = in.readFloat();
            float maxValue = in.readFloat();
            float logisticAlpha = in.readFloat();
            float logisticX0 = in.readFloat();
            int originalDimensions = in.readInt();
            int compressedDimension = in.readInt();

            ByteSequence<?> bytes = vectorTypeSupport.readByteSequence(in, compressedDimension);

            return new QuantizedSubVector(bytes, originalDimensions, bitsPerDimension, minValue, maxValue, logisticAlpha, logisticX0);
        }

        /**
         * Read the instance from a RandomAccessReader
         * @param in the RandomAccessReader
         * @throws IOException fails if we cannot read from the RandomAccessReader
         */
        public static void loadInto(RandomAccessReader in, QuantizedSubVector quantizedSubVector) throws IOException {
            quantizedSubVector.bitsPerDimension = BitsPerDimension.load(in);
            quantizedSubVector.minValue = in.readFloat();
            quantizedSubVector.maxValue = in.readFloat();
            quantizedSubVector.growthRate = in.readFloat();
            quantizedSubVector.midpoint = in.readFloat();
            quantizedSubVector.originalDimensions = in.readInt();
            in.readInt();

            vectorTypeSupport.readByteSequence(in, quantizedSubVector.bytes);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QuantizedSubVector that = (QuantizedSubVector) o;
            return (maxValue == that.maxValue)
                    && (minValue == that.minValue)
                    && (growthRate == that.growthRate)
                    && (midpoint == that.midpoint)
                    && (bitsPerDimension == that.bitsPerDimension)
                    && bytes.equals(that.bytes);
        }
    }

    /**
     * The loss used to optimize for the NVQ hyperparameters
     * We use the ratio between the loss given by the uniform quantization and the NVQ loss.
     */
    private static class NonuniformQuantizationLossFunction {
        final private BitsPerDimension bitsPerDimension;
        private VectorFloat<?> vector;
        private float minValue;
        private float maxValue;
        private float baseline;

        public NonuniformQuantizationLossFunction(BitsPerDimension bitsPerDimension) {
            this.bitsPerDimension = bitsPerDimension;
        }

        public void setVector(VectorFloat<?> vector, float minValue, float maxValue) {
            this.vector = vector;
            this.minValue = minValue;
            this.maxValue = maxValue;
            baseline = VectorUtil.nvqUniformLoss(vector, minValue, maxValue, bitsPerDimension.getInt());
        }

        public float computeRaw(float[] x) {
            return VectorUtil.nvqLoss(vector, x[0], x[1], minValue, maxValue, bitsPerDimension.getInt());
        }

        public float compute(float[] x) {
            return baseline / computeRaw(x);
        }
    }
}

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
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static io.github.jbellis.jvector.util.MathUtil.square;
import static io.github.jbellis.jvector.vector.VectorUtil.dotProduct;
import static io.github.jbellis.jvector.vector.VectorUtil.sub;
import static java.lang.Math.min;
import static java.lang.Math.sqrt;

/**
 * Product Quantization for float vectors.  Supports arbitrary source and target dimensionality;
 * in particular, the source does not need to be evenly divisible by the target.
 */
public class ProductQuantization implements VectorCompressor<ByteSequence<?>>, Accountable {
    private static final int MAGIC = 0x75EC4012; // JVECTOR, with some imagination

    protected static final Logger LOG = Logger.getLogger(ProductQuantization.class.getName());

    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    static final int DEFAULT_CLUSTERS = 256; // number of clusters per subspace = one byte's worth
    static final int K_MEANS_ITERATIONS = 6;
    public static final int MAX_PQ_TRAINING_SET_SIZE = 128000;

    final VectorFloat<?>[] codebooks; // array of codebooks, where each codebook is a VectorFloat consisting of k contiguous subvectors each of length M
    final int M; // codebooks.length, redundantly reproduced for convenience
    private final int clusterCount; // codebooks[0].length, redundantly reproduced for convenience
    final int originalDimension;
    final VectorFloat<?> globalCentroid;
    final int[][] subvectorSizesAndOffsets;
    final float anisotropicThreshold; // parallel cost multiplier
    private final float[][] centroidNormsSquared; // precomputed norms of the centroids, for encoding
    private final ThreadLocal<VectorFloat<?>> partialSums; // for dot product, euclidean, and cosine partials
    private final ThreadLocal<VectorFloat<?>> partialBestDistances; // for partial best distances during fused ADC
    private final ThreadLocal<ByteSequence<?>> partialQuantizedSums; // for quantized sums during fused ADC
    private final AtomicReference<VectorFloat<?>> partialSquaredMagnitudes; // for cosine partials
    private final AtomicReference<ByteSequence<?>> partialQuantizedSquaredMagnitudes; // for quantized squared magnitude partials during cosine fused ADC
    protected volatile float squaredMagnitudeDelta = 0; // for cosine fused ADC squared magnitude quantization delta (since this is invariant for a given PQ)
    protected volatile float minSquaredMagnitude = 0; // for cosine fused ADC minimum squared magnitude (invariant for a given PQ)

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param ravv the vectors to quantize
     * @param M number of subspaces
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     */
    public static ProductQuantization compute(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter) {
        return compute(ravv, M, clusterCount, globallyCenter, UNWEIGHTED, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    public static ProductQuantization compute(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter, float anisotropicThreshold) {
        return compute(ravv, M, clusterCount, globallyCenter, anisotropicThreshold, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param ravv the vectors to quantize
     * @param M number of subspaces
     * @param clusterCount number of clusters per subspace
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     * @param anisotropicThreshold the threshold of relevance for anisotropic angular distance shaping, giving
     *        higher priority to parallel error.  Anisotropic shaping requires that your dataset be normalized
     *        to unit length.  Use a threshold of UNWEIGHTED for isotropic distance
     *        (i.e. normal, unweighted L2 distance).
     * @param simdExecutor     ForkJoinPool instance for SIMD operations, best is to use a pool with the size of
     *                         the number of physical cores.
     * @param parallelExecutor ForkJoinPool instance for parallel stream operations
     */
    public static ProductQuantization compute(RandomAccessVectorValues ravv,
                                              int M,
                                              int clusterCount,
                                              boolean globallyCenter,
                                              float anisotropicThreshold,
                                              ForkJoinPool simdExecutor,
                                              ForkJoinPool parallelExecutor)
    {
        checkClusterCount(clusterCount);

        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), M);
        var vectors = extractTrainingVectors(ravv, parallelExecutor);

        // subtract the centroid from each training vector
        VectorFloat<?> globalCentroid;
        if (globallyCenter) {
            globalCentroid = KMeansPlusPlusClusterer.centroidOf(vectors);
            // subtract the centroid from each vector
            List<VectorFloat<?>> finalVectors = vectors;
            vectors = simdExecutor.submit(() -> finalVectors.stream().parallel().map(v -> VectorUtil.sub(v, globalCentroid)).collect(Collectors.<VectorFloat<?>>toList())).join();
        } else {
            globalCentroid = null;
        }

        // derive the codebooks
        var codebooks = createCodebooks(vectors, subvectorSizesAndOffsets, clusterCount, anisotropicThreshold, simdExecutor);
        return new ProductQuantization(codebooks, clusterCount, subvectorSizesAndOffsets, globalCentroid, anisotropicThreshold);
    }

    static List<VectorFloat<?>> extractTrainingVectors(RandomAccessVectorValues ravv, ForkJoinPool parallelExecutor) {
        // limit the number of vectors we train on
        var P = min(1.0f, MAX_PQ_TRAINING_SET_SIZE / (float) ravv.size());
        var ravvCopy = ravv.threadLocalSupplier();
        return parallelExecutor.submit(() -> IntStream.range(0, ravv.size()).parallel()
                        .filter(i -> ThreadLocalRandom.current().nextFloat() < P)
                        .mapToObj(targetOrd -> {
                            var localRavv = ravvCopy.get();
                            VectorFloat<?> v = localRavv.getVector(targetOrd);
                            return localRavv.isValueShared() ? v.copy() : v;
                        })
                        .collect(Collectors.toList()))
                .join();
    }

    /**
     * Create a new PQ by fine-tuning this one with the data in `ravv`
     */
    public ProductQuantization refine(RandomAccessVectorValues ravv) {
        return refine(ravv, 1, UNWEIGHTED, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * Create a new PQ by fine-tuning this one with the data in `ravv`
     *
     * @param lloydsRounds number of Lloyd's iterations to run against
     *                     the new data.  Suggested values are 1 or 2.
     */
    public ProductQuantization refine(RandomAccessVectorValues ravv,
                                      int lloydsRounds,
                                      float anisotropicThreshold,
                                      ForkJoinPool simdExecutor,
                                      ForkJoinPool parallelExecutor)
    {
        if (lloydsRounds < 0) {
            throw new IllegalArgumentException("lloydsRounds must be non-negative");
        }

        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), M);
        var vectorsMutable = extractTrainingVectors(ravv, parallelExecutor);
        if (globalCentroid != null) {
            var vectors = vectorsMutable;
            vectorsMutable = simdExecutor.submit(() -> vectors.stream().parallel().map(v -> VectorUtil.sub(v, globalCentroid)).collect(Collectors.<VectorFloat<?>>toList())).join();
        }
        var vectors = vectorsMutable; // "effectively final" to make the closure happy

        var refinedCodebooks = simdExecutor.submit(() -> IntStream.range(0, M).parallel().mapToObj(m -> {
            VectorFloat<?>[] subvectors = extractSubvectors(vectors, m, subvectorSizesAndOffsets);
            var clusterer = new KMeansPlusPlusClusterer(subvectors, codebooks[m], anisotropicThreshold);
            return clusterer.cluster(anisotropicThreshold == UNWEIGHTED ? lloydsRounds : 0,
                                     anisotropicThreshold == UNWEIGHTED ? 0 : lloydsRounds);
        }).toArray(VectorFloat<?>[]::new)).join();

        return new ProductQuantization(refinedCodebooks, clusterCount, subvectorSizesAndOffsets, globalCentroid, anisotropicThreshold);
    }

    ProductQuantization(VectorFloat<?>[] codebooks, int clusterCount, int[][] subvectorSizesAndOffsets, VectorFloat<?> globalCentroid, float anisotropicThreshold) {
        checkClusterCount(clusterCount);

        this.codebooks = codebooks;
        this.globalCentroid = globalCentroid;
        this.M = codebooks.length;
        this.clusterCount = clusterCount;
        this.subvectorSizesAndOffsets = subvectorSizesAndOffsets;
        this.originalDimension = Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum();
        if (globalCentroid != null && globalCentroid.length() != originalDimension) {
            var msg = String.format("Global centroid length %d does not match vector dimensionality %d", globalCentroid.length(), originalDimension);
            throw new IllegalArgumentException(msg);
        }
        this.anisotropicThreshold = anisotropicThreshold;
        this.partialSums = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(getSubspaceCount() * getClusterCount()));
        this.partialBestDistances = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(getSubspaceCount()));
        this.partialQuantizedSums = ThreadLocal.withInitial(() -> vectorTypeSupport.createByteSequence(getSubspaceCount() * getClusterCount() * 2));
        this.partialSquaredMagnitudes = new AtomicReference<>(null);
        this.partialQuantizedSquaredMagnitudes= new AtomicReference<>(null);


        centroidNormsSquared = new float[M][clusterCount];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < clusterCount; j++) {
                centroidNormsSquared[i][j] = dotProduct(codebooks[i], j * subvectorSizesAndOffsets[i][0],
                                                        codebooks[i], j * subvectorSizesAndOffsets[i][0],
                                                        subvectorSizesAndOffsets[i][0]);
            }
        }
    }

    @Override
    public ImmutablePQVectors createCompressedVectors(Object[] compressedVectors) {
        return new ImmutablePQVectors(this, (ByteSequence<?>[]) compressedVectors, compressedVectors.length, 1);
    }

    /**
     * Encodes the given vectors in parallel using the PQ codebooks. If a vector is missing (null), it will be encoded
     * as a zero vector.
     */
    @Override
    public PQVectors encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        return PQVectors.encodeAndBuild(this, ravv.size(), ravv, simdExecutor);
    }

    /**
     * Encodes the input vector using the PQ codebooks, weighing parallel loss more than orthogonal loss, into
     * the given ByteSequence.
     */
    private void encodeAnisotropic(VectorFloat<?> vector, ByteSequence<?> result) {
        // compute the residuals from each subvector to each corresponding codebook centroid
        Residual[][] residuals = computeResiduals(vector);
        assert residuals.length == M : "Residuals length mismatch " + residuals.length + " != " + M;
        // start with centroids that minimize the residual norms
        initializeToMinResidualNorms(residuals, result);
        // sum the initial parallel residual component
        float parallelResidualComponentSum = 0;
        for (int i = 0; i < result.length(); i++) {
            int centroidIdx = Byte.toUnsignedInt(result.get(i));
            parallelResidualComponentSum += residuals[i][centroidIdx].parallelResidualComponent;
        }

        // SCANN sorts the subspaces by residual norm here (and adds a sorted->original subspace index map),
        // presumably with the intent to help this converge faster, but profiling shows that almost 90% of the
        // cost of this method is computeResiduals + initializeToMinResidualNorms, so we're not going to bother.

        // Optimize until convergence
        int MAX_ITERATIONS = 10; // borrowed from SCANN code without experimenting w/ other values
        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            // loop over each subspace
            boolean changed = false;
            for (int i = 0; i < residuals.length; i++) {
                int oldIdx = Byte.toUnsignedInt(result.get(i));
                CoordinateDescentResult cdr = optimizeSingleSubspace(residuals[i], oldIdx, parallelResidualComponentSum);
                if (cdr.newCenterIdx != oldIdx) {
                    parallelResidualComponentSum = cdr.newParallelResidualComponent;
                    result.set(i, (byte) cdr.newCenterIdx);
                    changed = true;
                }
            }
            // Done if nothing changed this iteration
            if (!changed) {
                break;
            }
        }
    }

    private CoordinateDescentResult optimizeSingleSubspace(Residual[] residuals, int oldIdx, float oldParallelResidualSum) {
        // (this is global to all subspaces but it's not worth stashing in a field)
        float pcm = KMeansPlusPlusClusterer.computeParallelCostMultiplier(anisotropicThreshold, originalDimension);

        float oldResidualNormSquared = residuals[oldIdx].residualNormSquared;
        float oldParallelComponent = residuals[oldIdx].parallelResidualComponent;

        float bestCostDelta = 0;
        int bestIndex = oldIdx;
        float bestParallelResidualSum = oldParallelResidualSum;

        // loop over potential new centers
        for (int thisIdx = 0; thisIdx < residuals.length; thisIdx++) {
            if (thisIdx == oldIdx) {
                continue;
            }

            // compute the new parallel residual sum and parallel norm delta
            Residual rs = residuals[thisIdx];
            float thisParallelResidualSum = oldParallelResidualSum - oldParallelComponent + rs.parallelResidualComponent;
            float parallelNormDelta = square(thisParallelResidualSum) - square(oldParallelResidualSum);
            // quit early if new parallel norm is worse than the old
            if (parallelNormDelta > 0) {
                continue;
            }

            // compute the total cost delta
            float residualNormDelta = rs.residualNormSquared - oldResidualNormSquared;
            float perpendicularNormDelta = residualNormDelta - parallelNormDelta;
            float costDelta = pcm * parallelNormDelta + perpendicularNormDelta;

            // save the new center if it's the best so far
            if (costDelta < bestCostDelta) {
                bestCostDelta = costDelta;
                bestIndex = thisIdx;
                bestParallelResidualSum = thisParallelResidualSum;
            }
        }

        return new CoordinateDescentResult(bestIndex, bestParallelResidualSum);
    }

    /**
     * Wraps the two values we want to return from optimizeSingleSubspace
     */
    private static class CoordinateDescentResult {
        final int newCenterIdx;
        final float newParallelResidualComponent;

        CoordinateDescentResult(int newCenterIdx, float newParallelResidualComponent) {
            this.newCenterIdx = newCenterIdx;
            this.newParallelResidualComponent = newParallelResidualComponent;
        }
    }

    /**
     * @return codebook ordinals representing the cluster centroids for each subspace that minimize the residual norm
     */
    private void initializeToMinResidualNorms(Residual[][] residualStats, ByteSequence<?> dest) {
        // for each subspace
        for (int i = 0; i < residualStats.length; i++) {
            int minIndex = -1;
            double minNormSquared = Double.MAX_VALUE;
            // find the centroid with the smallest residual norm in this subspace
            for (int j = 0; j < residualStats[i].length; j++) {
                if (residualStats[i][j].residualNormSquared < minNormSquared) {
                    minNormSquared = residualStats[i][j].residualNormSquared;
                    minIndex = j;
                }
            }
            dest.set(i, (byte) minIndex);
        }
    }

    /**
     * @return the parallel-cost residuals for each subspace and cluster
     */
    private Residual[][] computeResiduals(VectorFloat<?> vector) {
        Residual[][] residuals = new Residual[codebooks.length][];

        float inverseNorm = (float) (1.0 / sqrt(dotProduct(vector, vector)));
        for (int i = 0; i < codebooks.length; i++) {
            var x = getSubVector(vector, i, subvectorSizesAndOffsets);
            float xNormSquared = dotProduct(x, x);
            residuals[i] = new Residual[clusterCount];

            for (int j = 0; j < clusterCount; j++) {
                residuals[i][j] = computeResidual(x, codebooks[i], j, centroidNormsSquared[i][j], xNormSquared, inverseNorm);
            }
        }

        return residuals;
    }

    /**
     * Represents the residual after subtracting a cluster centroid from a [sub]vector.
     */
    private static class Residual {
        final float residualNormSquared;
        final float parallelResidualComponent;

        Residual(float residualNormSquared, float parallelResidualComponent) {
            this.residualNormSquared = residualNormSquared;
            this.parallelResidualComponent = parallelResidualComponent;
        }
    }

    private Residual computeResidual(VectorFloat<?> x, VectorFloat<?> centroids, int centroid, float cNormSquared, float xNormSquared, float inverseNorm) {
        float cDotX = VectorUtil.dotProduct(centroids, centroid * x.length(), x, 0, x.length());
        float residualNormSquared = cNormSquared - 2 * cDotX + xNormSquared;
        float parallelErrorSubtotal = cDotX - xNormSquared;
        float parallelResidualComponent = square(parallelErrorSubtotal) * inverseNorm;
        return new Residual(residualNormSquared, parallelResidualComponent);
    }

    private void encodeUnweighted(VectorFloat<?> vector, ByteSequence<?> dest) {
        for (int m = 0; m < M; m++) {
            dest.set(m, (byte) closestCentroidIndex(vector, m, codebooks[m]));
        }
    }

    /**
     * Encodes the input vector using the PQ codebooks.
     * @return one byte per subspace
     */
    @Override
    public ByteSequence<?> encode(VectorFloat<?> vector) {
        var result = vectorTypeSupport.createByteSequence(M);
        encodeTo(vector, result);
        return result;
    }

    @Override
    public void encodeTo(VectorFloat<?> vector, ByteSequence<?> dest) {
        if (globalCentroid != null) {
            vector = sub(vector, globalCentroid);
        }

        if (anisotropicThreshold > UNWEIGHTED)
            encodeAnisotropic(vector, dest);
        else
            encodeUnweighted(vector, dest);
    }

    /**
     * Decodes the quantized representation (ByteSequence) to its approximate original vector.
     */
    public void decode(ByteSequence<?> encoded, VectorFloat<?> target) {
        decodeCentered(encoded, target);

        if (globalCentroid != null) {
            // Add back the global centroid to get the approximate original vector.
            VectorUtil.addInPlace(target, globalCentroid);
        }
    }

    /**
     * Decodes the quantized representation (ByteSequence) to its approximate original vector, relative to the global centroid.
     */
    void decodeCentered(ByteSequence<?> encoded, VectorFloat<?> target) {
        for (int m = 0; m < M; m++) {
            int centroidIndex = Byte.toUnsignedInt(encoded.get(m));
            target.copyFrom(codebooks[m], centroidIndex * subvectorSizesAndOffsets[m][0], subvectorSizesAndOffsets[m][1], subvectorSizesAndOffsets[m][0]);
        }
    }

    /**
     * @return how many bytes we are compressing to
     */
    public int getSubspaceCount() {
        return M;
    }

    /**
     * @return number of clusters per subspace
     */
    public int getClusterCount() {
        return clusterCount;
    }

    static VectorFloat<?>[] createCodebooks(List<VectorFloat<?>> vectors, int[][] subvectorSizeAndOffset, int clusters, float anisotropicThreshold, ForkJoinPool simdExecutor) {
        int M = subvectorSizeAndOffset.length;
        return simdExecutor.submit(() -> IntStream.range(0, M).parallel().mapToObj(m -> {
            VectorFloat<?>[] subvectors = extractSubvectors(vectors, m, subvectorSizeAndOffset);
            var clusterer = new KMeansPlusPlusClusterer(subvectors, clusters, anisotropicThreshold);
            return clusterer.cluster(K_MEANS_ITERATIONS, anisotropicThreshold == UNWEIGHTED ? 0 : K_MEANS_ITERATIONS);
        }).toArray(VectorFloat<?>[]::new)).join();
    }

    /**
     * Extract VectorFloat subvectors corresponding to the m'th subspace.
     * This is NOT done in parallel (since the callers are themselves running in parallel).
     */
    private static VectorFloat<?>[] extractSubvectors(List<VectorFloat<?>> vectors, int m, int[][] subvectorSizeAndOffset) {
        return vectors.stream()
                .map(vector -> getSubVector(vector, m, subvectorSizeAndOffset))
                .toArray(VectorFloat<?>[]::new);
    }

    int closestCentroidIndex(VectorFloat<?> subvector, int m, VectorFloat<?> codebook) {
        int index = 0;
        float minDist = Float.MAX_VALUE;
        int subvectorSize = subvectorSizesAndOffsets[m][0];
        int subvectorOffset = subvectorSizesAndOffsets[m][1];
        for (int i = 0; i < clusterCount; i++) {
            float dist = VectorUtil.squareL2Distance(subvector, subvectorOffset, codebook, i * subvectorSize, subvectorSize);
            if (dist < minDist) {
                minDist = dist;
                index = i;
            }
        }
        return index;
    }

    /**
     * Extracts the m-th subvector from a single vector.
     */
    static VectorFloat<?> getSubVector(VectorFloat<?> vector, int m, int[][] subvectorSizeAndOffset) {
        VectorFloat<?> subvector = vectorTypeSupport.createFloatVector(subvectorSizeAndOffset[m][0]);
        subvector.copyFrom(vector, subvectorSizeAndOffset[m][1], 0, subvectorSizeAndOffset[m][0]);
        return subvector;
    }

    /**
     * Splits the vector dimension into M subvectors of roughly equal size.
     */
    @VisibleForTesting
    static int[][] getSubvectorSizesAndOffsets(int dimensions, int M) {
        if (M > dimensions) {
            throw new IllegalArgumentException("Number of subspaces must be less than or equal to the vector dimension");
        }
        int[][] sizes = new int[M][];
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

    VectorFloat<?> reusablePartialSums() {
        return partialSums.get();
    }

    ByteSequence<?> reusablePartialQuantizedSums() {
        return partialQuantizedSums.get();
    }

    VectorFloat<?> reusablePartialBestDistances() {
        return partialBestDistances.get();
    }

    AtomicReference<VectorFloat<?>> partialSquaredMagnitudes() {
        return partialSquaredMagnitudes;
    }

    AtomicReference<ByteSequence<?>> partialQuantizedSquaredMagnitudes() {
        return partialQuantizedSquaredMagnitudes;
    }

    public void write(DataOutput out, int version) throws IOException
    {
        if (version > OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("Unsupported serialization version " + version);
        }
        if (version < 3 && anisotropicThreshold != UNWEIGHTED) {
            throw new IllegalArgumentException("Anisotropic threshold is only supported in serialization version 3 and above");
        }

        if (version >= 3) {
            out.writeInt(MAGIC);
            out.writeInt(version);
        }

        if (globalCentroid == null) {
            out.writeInt(0);
        } else {
            out.writeInt(globalCentroid.length());
            vectorTypeSupport.writeFloatVector(out, globalCentroid);
        }

        out.writeInt(M);
        assert Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum() == originalDimension;
        assert M == subvectorSizesAndOffsets.length;
        for (var a : subvectorSizesAndOffsets) {
            out.writeInt(a[0]);
        }

        if (version >= 3) {
            out.writeFloat(anisotropicThreshold);
        }

        assert codebooks.length == M;
        out.writeInt(clusterCount);
        for (int i = 0; i < M; i++) {
            var codebook = codebooks[i];
            assert codebook.length() == clusterCount * subvectorSizesAndOffsets[i][0];
            vectorTypeSupport.writeFloatVector(out, codebook);
        }
    }

    @Override
    public int compressorSize() {
        int size = 0;
        size += Integer.BYTES; // MAGIC
        size += Integer.BYTES; // STORAGE_VERSION
        size += Integer.BYTES; // globalCentroidLength
        if (globalCentroid != null) {
            size += Float.BYTES * globalCentroid.length();
        }
        size += Integer.BYTES; // M
        size += Integer.BYTES * M; // subvectorSizesAndOffsets (only the sizes are written)
        size += Float.BYTES; // anisotropicThreshold
        size += Integer.BYTES; // clusterCount
        for (int i = 0; i < M; i++) {
            size += Float.BYTES * codebooks[i].length();
        }
        return size;
    }

    public static ProductQuantization load(RandomAccessReader in) throws IOException {
        int maybeMagic = in.readInt();
        int version;
        int globalCentroidLength;
        if (maybeMagic != MAGIC) {
            // JVector 1+2 format, no magic or version, starts straight off with the centroid length
            version = 0;
            globalCentroidLength = maybeMagic;
        } else {
            version = in.readInt();
            globalCentroidLength = in.readInt();
        }

        VectorFloat<?> globalCentroid = null;
        if (globalCentroidLength > 0) {
            globalCentroid = vectorTypeSupport.readFloatVector(in, globalCentroidLength);
        }

        int M = in.readInt();
        int[][] subvectorSizes = new int[M][];
        int offset = 0;
        for (int i = 0; i < M; i++) {
            subvectorSizes[i] = new int[2];
            int size = in.readInt();
            subvectorSizes[i][0] = size;
            subvectorSizes[i][1] = offset;
            offset += size;
        }

        float anisotropicThreshold;
        if (version < 3) {
            anisotropicThreshold = UNWEIGHTED;
        } else {
            anisotropicThreshold = in.readFloat();
        }

        int clusters = in.readInt();
        VectorFloat<?>[] codebooks = new VectorFloat<?>[M];
        for (int m = 0; m < M; m++) {
            VectorFloat<?> codebook = vectorTypeSupport.readFloatVector(in, clusters * subvectorSizes[m][0]);
            codebooks[m] = codebook;
        }

        return new ProductQuantization(codebooks, clusters, subvectorSizes, globalCentroid, anisotropicThreshold);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ProductQuantization that = (ProductQuantization) o;
        return M == that.M
                && originalDimension == that.originalDimension
                && Objects.equals(globalCentroid, that.globalCentroid)
                && Arrays.deepEquals(subvectorSizesAndOffsets, that.subvectorSizesAndOffsets)
                && Arrays.deepEquals(codebooks, that.codebooks)
                && anisotropicThreshold == that.anisotropicThreshold;
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(M, originalDimension);
        result = 31 * result + Arrays.deepHashCode(codebooks);
        result = 31 * result + Objects.hashCode(globalCentroid);
        result = 31 * result + Arrays.deepHashCode(subvectorSizesAndOffsets);
        return result;
    }

    /**
     * @return the centroid of the codebooks
     */
    public VectorFloat<?> getOrComputeCentroid() {
        if (globalCentroid != null) {
            return globalCentroid;
        }

        // typically we only precompute the centroid for Euclidean similarity
        var centroid = vectorTypeSupport.createFloatVector(originalDimension);
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < clusterCount; i++) {
                var subspaceSize = subvectorSizesAndOffsets[m][0];
                var subCentroid = vectorTypeSupport.createFloatVector(subspaceSize);
                subCentroid.copyFrom(codebooks[m], i * subspaceSize, 0, subspaceSize);
                // we don't have addInPlace for vectors of different length, so do it by hand
                for (int j = 0; j < subspaceSize; j++) {
                    var k = subvectorSizesAndOffsets[m][1] + j;
                    centroid.set(k, centroid.get(k) + subCentroid.get(j));
                }
            }
        }
        VectorUtil.scale(centroid, 1.0f / M);
        return centroid;
    }

    @Override
    public int compressedVectorSize() {
        return codebooks.length;
    }

    @Override
    public long ramBytesUsed() {
        long size = 0;
        for (VectorFloat<?> codebook : codebooks) {
            size += codebook.ramBytesUsed();
        }

        return size;
    }

    @Override
    public String toString() {
        if (anisotropicThreshold == UNWEIGHTED) {
            return String.format("ProductQuantization(M=%d, clusters=%d)", M, clusterCount);
        }
        return String.format("ProductQuantization(M=%d, clusters=%d, T=%.3f, eta=%.1f)",
                             M,
                             clusterCount,
                             anisotropicThreshold,
                             KMeansPlusPlusClusterer.computeParallelCostMultiplier(anisotropicThreshold, originalDimension));
    }

    private static void checkClusterCount(int clusterCount) {
        if (clusterCount > 256) {
            throw new IllegalArgumentException("Too many PQ clusters: " + clusterCount + " > 256");
        }
        if (clusterCount < 256) {
            LOG.warning("Using less than 256 PQ clusters will not reduce the memory footprint.");
        }
    }
}

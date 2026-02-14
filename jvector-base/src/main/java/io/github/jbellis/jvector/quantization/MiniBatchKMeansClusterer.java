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

import io.github.jbellis.jvector.vector.Matrix;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static io.github.jbellis.jvector.util.MathUtil.square;
import static io.github.jbellis.jvector.vector.VectorUtil.*;
import static java.lang.Math.max;

/**
 * A MiniBatch KMeans implementation for float vectors.
 * Drop-in replacement for KMeansPlusPlusClusterer, optimized for large datasets (10M+).
 * * Includes:
 * 1. Mini-Batch Stochastic Updates (fast convergence).
 * 2. K-Means++ Subsampling Initialization (fast startup).
 */
public class MiniBatchKMeansClusterer {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static final boolean VERBOSE_KMEANS = Boolean.getBoolean("jvector.kmeans.verbose");
    private static final boolean VERBOSE_KMEANS_TIMING = Boolean.getBoolean("jvector.kmeans.timing");

    public static final float UNWEIGHTED = -1.0f;

    private final int k;
    private final VectorFloat<?>[] points;
    private final int[] assignments;
    private final VectorFloat<?> centroids;
    private final float anisotropicThreshold;

    // MiniBatch specific
    private final int batchSize;
    private final int[] centroidCounts;

    /**
     * @param batchSize Recommended: 1024 or 2048 for large datasets
     */
    public MiniBatchKMeansClusterer(VectorFloat<?>[] points, int k, int batchSize) {
        this(points, chooseInitialCentroids(points, k), UNWEIGHTED, batchSize);
    }

    public MiniBatchKMeansClusterer(VectorFloat<?>[] points, int k, float anisotropicThreshold, int batchSize) {
        this(points, chooseInitialCentroids(points, k), anisotropicThreshold, batchSize);
    }

    public MiniBatchKMeansClusterer(VectorFloat<?>[] points, VectorFloat<?> centroids, float anisotropicThreshold, int batchSize) {
        if (Float.isNaN(anisotropicThreshold) || anisotropicThreshold < -1.0 || anisotropicThreshold >= 1.0) {
            throw new IllegalArgumentException("Valid range for anisotropic threshold T is -1.0 <= t < 1.0");
        }

        this.points = points;
        this.k = centroids.length() / points[0].length();
        this.centroids = centroids.copy();
        this.anisotropicThreshold = anisotropicThreshold;
        this.batchSize = Math.min(batchSize, points.length);

        this.centroidCounts = new int[k];
        this.assignments = new int[points.length];
        Arrays.fill(this.assignments, -1);
    }

    static float computeParallelCostMultiplier(double threshold, int dimensions) {
        assert Double.isFinite(threshold) : "threshold=" + threshold;
        double parallelCost = threshold * threshold;
        double perpendicularCost = (1 - parallelCost) / (dimensions - 1);
        return (float) max(1.0, (parallelCost / perpendicularCost));
    }

    /**
     * Performs clustering.
     * @param unweightedIterations If small (< 100), interpreted as EPOCHS (full passes).
     * If large, interpreted as TOTAL BATCH UPDATES.
     * @param anisotropicIterations Number of full-batch anisotropic refinement steps.
     */
    public VectorFloat<?> cluster(int unweightedIterations, int anisotropicIterations) {

        // ------------------------------------------------------------
        // Mini-Batch Unweighted (isotropic) k-means phase
        // ------------------------------------------------------------
        int totalBatches = unweightedIterations;
        // Interpret small iteration counts as Epochs to ensure sufficient data coverage
        if (unweightedIterations < 100) {
            int batchesPerEpoch = Math.max(1, points.length / batchSize);
            totalBatches = unweightedIterations * batchesPerEpoch;
        }

        long tTotal0 = System.nanoTime();

        for (int i = 0; i < totalBatches; i++) {
            long t0 = VERBOSE_KMEANS_TIMING ? System.nanoTime() : 0L;

            clusterOnceMiniBatch();

            long t1 = VERBOSE_KMEANS_TIMING ? System.nanoTime() : 0L;
            if (VERBOSE_KMEANS_TIMING && i % 100 == 0) {
                System.out.printf("MiniBatch iter %d time = %.3f ms%n", i, (t1 - t0) * 1e-6);
            }
        }

        if (VERBOSE_KMEANS) {
            System.out.printf("MiniBatch unweighted phase finished. Total batches: %d. Total time: %.3f s%n",
                    totalBatches, (System.nanoTime() - tTotal0) * 1e-9);
        }

        // ------------------------------------------------------------
        // Anisotropic refinement phase (Full Batch)
        // ------------------------------------------------------------
        if (anisotropicIterations > 0) {
            // MiniBatch leaves assignments in flux; we must perform a full assignment pass
            // so the anisotropic phase has valid global clusters.
            assignAllPoints();

            final int threshold = Math.max(1, (int) (0.01 * points.length));

            for (int i = 0; i < anisotropicIterations; i++) {
                long t0 = VERBOSE_KMEANS_TIMING ? System.nanoTime() : 0L;
                int changedCount = clusterOnceAnisotropic();
                long t1 = VERBOSE_KMEANS_TIMING ? System.nanoTime() : 0L;

                if (VERBOSE_KMEANS) {
                    System.out.printf("KMeans anisotropic iter %d: %d/%d reassigned (%.3f%%)%n",
                            i, changedCount, points.length, 100.0 * changedCount / points.length);
                }

                if (VERBOSE_KMEANS_TIMING) {
                    System.out.printf("  time = %.3f ms%n", (t1 - t0) * 1e-6);
                }

                if (changedCount <= threshold) {
                    if (VERBOSE_KMEANS) System.out.println("KMeans anisotropic converged early");
                    break;
                }
            }
        }

        return centroids;
    }

    /**
     * Performs a single MiniBatch update step:
     * 1. Sample batch
     * 2. Find nearest centroids
     * 3. Update centroids immediately using learning rate
     */
    public void clusterOnceMiniBatch() {
        var random = ThreadLocalRandom.current();

        // 1. Sample indices
        int[] batchIndices = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            batchIndices[i] = random.nextInt(points.length);
        }

        // 2. Compute assignments for the batch
        int[] batchAssignments = new int[batchSize];
        for(int i = 0; i < batchSize; i++) {
            batchAssignments[i] = getNearestCluster(points[batchIndices[i]]);
        }

        // 3. Update Centroids (Online)
        int dim = centroids.length() / k;
        VectorFloat<?> tempDiff = vectorTypeSupport.createFloatVector(dim);

        for (int i = 0; i < batchSize; i++) {
            int pointIdx = batchIndices[i];
            int clusterIdx = batchAssignments[i];
            VectorFloat<?> point = points[pointIdx];

            centroidCounts[clusterIdx]++;
            float learningRate = 1.0f / centroidCounts[clusterIdx];

            // tempDiff = X
            tempDiff.copyFrom(point, 0, 0, dim);
            // tempDiff = X - C_old
            subFromOffset(tempDiff, centroids, clusterIdx * dim);

            // tempDiff = lr * (X - C_old)
            scale(tempDiff, learningRate);

            // C_new = C_old + tempDiff
            // Pass the offset explicitly as the second argument
            addToOffset(centroids, clusterIdx * dim, tempDiff);
        }
    }

    // --- Helpers ---

    private void assignAllPoints() {
        for (int i = 0; i < points.length; i++) {
            assignments[i] = getNearestCluster(points[i]);
        }
    }

    public int clusterOnceAnisotropic() {
        updateCentroidsAnisotropic();
        return updateAssignedPointsAnisotropic();
    }

    // addToOffset: Adds a SMALL vector (source) into a LARGE vector (dest) at a specific offset.
    private static void addToOffset(VectorFloat<?> dest, int destOffset, VectorFloat<?> source) {
        // Fix: Iterate over source.length() (Small Temp Vector)
        for (int i = 0; i < source.length(); i++) {
            dest.set(destOffset + i, dest.get(destOffset + i) + source.get(i));
        }
    }

    // subFromOffset: Subtracts a slice of a LARGE vector (source) from a SMALL vector (dest).
    // (This logic was actually correct, but verifying strictly ensures safety)
    private static void subFromOffset(VectorFloat<?> dest, VectorFloat<?> source, int sourceOffset) {
        for (int i = 0; i < dest.length(); i++) {
            dest.set(i, dest.get(i) - source.get(sourceOffset + i));
        }
    }

    // --- Optimized Initialization Logic ---

    /**
     * Chooses the initial centroids for clustering using K-Means++.
     * If the dataset is large, uses a random subsample to accelerate initialization.
     */
    private static VectorFloat<?> chooseInitialCentroids(VectorFloat<?>[] points, int k) {
        if (k <= 0) throw new IllegalArgumentException("Number of clusters must be positive.");
        if (k > points.length) throw new IllegalArgumentException("K cannot exceed N");

        // Heuristic: If N is massive, run KMeans++ on a subset only.
        // Cap at 50,000 points or 10*k to ensure statistical representation without performance penalty.
        int maxInitSamples = Math.max(k * 10, 50_000);

        if (points.length <= maxInitSamples) {
            return runStandardKMeansPlusPlus(points, k);
        } else {
            if (VERBOSE_KMEANS) {
                System.out.printf("Subsampling %d points for K-Means++ initialization...%n", maxInitSamples);
            }
            var random = ThreadLocalRandom.current();
            VectorFloat<?>[] subset = new VectorFloat<?>[maxInitSamples];
            for (int i = 0; i < maxInitSamples; i++) {
                subset[i] = points[random.nextInt(points.length)];
            }
            return runStandardKMeansPlusPlus(subset, k);
        }
    }

    /**
     * Standard KMeans++ implementation.
     * When called by chooseInitialCentroids, 'points' may be the full set or a subset.
     */
    private static VectorFloat<?> runStandardKMeansPlusPlus(VectorFloat<?>[] points, int k) {
        var random = ThreadLocalRandom.current();
        VectorFloat<?> centroids = vectorTypeSupport.createFloatVector(k * points[0].length());

        float[] distances = new float[points.length];
        Arrays.fill(distances, Float.MAX_VALUE);
        VectorFloat<?> distancesVector = vectorTypeSupport.createFloatVector(distances);

        VectorFloat<?> firstCentroid = points[random.nextInt(points.length)];
        centroids.copyFrom(firstCentroid, 0, 0, firstCentroid.length());

        VectorFloat<?> newDistancesVector = vectorTypeSupport.createFloatVector(points.length);
        for (int i = 0; i < points.length; i++) {
            newDistancesVector.set(i, squareL2Distance(points[i], firstCentroid));
        }
        VectorUtil.minInPlace(distancesVector, newDistancesVector);

        for (int i = 1; i < k; i++) {
            float totalDistance = VectorUtil.sum(distancesVector);
            float r = random.nextFloat() * totalDistance;
            int selectedIdx = -1;

            for (int j = 0; j < points.length; j++) {
                r -= distancesVector.get(j);
                if (r < 1e-6) {
                    selectedIdx = j;
                    break;
                }
            }
            if (selectedIdx == -1) selectedIdx = random.nextInt(points.length);

            VectorFloat<?> nextCentroid = points[selectedIdx];
            centroids.copyFrom(nextCentroid, 0, i * nextCentroid.length(), nextCentroid.length());

            for (int j = 0; j < points.length; j++) {
                newDistancesVector.set(j, squareL2Distance(points[j], nextCentroid));
            }
            VectorUtil.minInPlace(distancesVector, newDistancesVector);
        }
        return centroids;
    }

    // --- Standard Distance and Update Logic ---

    private int getNearestCluster(VectorFloat<?> point) {
        float minDistance = Float.MAX_VALUE;
        int nearestCluster = 0;
        int dim = point.length();

        for (int i = 0; i < k; i++) {
            float distance = squareL2Distance(point, 0, centroids, i * dim, dim);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }
        return nearestCluster;
    }

    private int updateAssignedPointsAnisotropic() {
        float pcm = computeParallelCostMultiplier(anisotropicThreshold, points[0].length());
        int dim = points[0].length();

        float[] cNormSquared = new float[k];
        for (int i = 0; i < k; i++) {
            cNormSquared[i] = dotProduct(centroids, i * dim, centroids, i * dim, dim);
        }

        int changedCount = 0;
        for (int i = 0; i < points.length; i++) {
            var x = points[i];
            var xNormSquared = dotProduct(x, x);

            int index = assignments[i];
            float minDist = Float.MAX_VALUE;
            for (int j = 0; j < k; j++) {
                float dist = weightedDistance(x, j, pcm, cNormSquared[j], xNormSquared);
                if (dist < minDist) {
                    minDist = dist;
                    index = j;
                }
            }

            if (index != assignments[i]) {
                changedCount++;
                assignments[i] = index;
            }
        }
        return changedCount;
    }

    private float weightedDistance(VectorFloat<?> x, int centroid, float parallelCostMultiplier, float cNormSquared, float xNormSquared) {
        float cDotX = VectorUtil.dotProduct(centroids, centroid * x.length(), x, 0, x.length());
        float parallelErrorSubtotal = cDotX - xNormSquared;
        float residualSquaredNorm = cNormSquared - 2 * cDotX + xNormSquared;
        float parallelError = square(parallelErrorSubtotal);
        float perpendicularError = residualSquaredNorm - parallelError;
        return parallelCostMultiplier * parallelError + perpendicularError;
    }

    private void updateCentroidsAnisotropic() {
        int dimensions = points[0].length();
        float pcm = computeParallelCostMultiplier(anisotropicThreshold, dimensions);
        float orthogonalCostMultiplier = 1.0f / pcm;

        var pointsByCluster = new HashMap<Integer, List<Integer>>();
        for (int i = 0; i < assignments.length; i++) {
            pointsByCluster.computeIfAbsent(assignments[i], __ -> new ArrayList<>()).add(i);
        }

        for (int i = 0; i < k; i++) {
            var L = pointsByCluster.getOrDefault(i, List.of());
            if (L.isEmpty()) {
                initializeCentroidToRandomPoint(i);
                continue;
            }

            var mean = vectorTypeSupport.createFloatVector(dimensions);
            var outerProdSums = new Matrix(dimensions, dimensions);
            for (int j : L) {
                var point = points[j];
                // Offset is 0 because 'mean' is a fresh vector of the same dimension as 'point'
                addToOffset(mean, 0, point);
                float denom = dotProduct(point, point);
                if (denom > 0) {
                    var op = Matrix.outerProduct(point, point);
                    op.scale(1.0f / denom);
                    outerProdSums.addInPlace(op);
                }
            }
            outerProdSums.scale((1 - orthogonalCostMultiplier) / L.size());
            scale(mean, 1.0f / L.size());

            for (int j = 0; j < dimensions; j++) {
                outerProdSums.addTo(j, j, orthogonalCostMultiplier);
            }

            var invertedMatrix = outerProdSums.invert();
            centroids.copyFrom(invertedMatrix.multiply(mean), 0, i * dimensions, dimensions);
        }
    }

    private void initializeCentroidToRandomPoint(int i) {
        var random = ThreadLocalRandom.current();
        centroids.copyFrom(points[random.nextInt(points.length)], 0, i * points[0].length(), points[0].length());
    }
}

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
 * A KMeans++ implementation for float vectors.  Optimizes to use SIMD vector instructions if available.
 */
public class KMeansPlusPlusClusterer {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static final float UNWEIGHTED = -1.0f;

    // number of centroids to compute
    private final int k;

    // the points to train on
    private final VectorFloat<?>[] points;
    // the cluster each point is assigned to
    private final int[] assignments;
    // the centroids of each cluster, as one big k*M-dimension vector
    private final VectorFloat<?> centroids;

    // the threshold of relevance for anisotropic angular distance shaping, -1.0 < anisotropicThreshold <= 1.0
    private final float anisotropicThreshold;

    // used to accelerate updating clusters by unweighted L2 distance.  (not used for anisotropic clustering)
    private final int[] centroidDenoms; // the number of points assigned to each cluster
    private final VectorFloat<?>[] centroidNums; // the sum of all points assigned to each cluster

    /**
     * Constructs a KMeansPlusPlusFloatClusterer with the specified points and number of clusters.
     *
     * @param points the points to cluster (points[n][i] is the ith component of the nth point)
     * @param k number of clusters.
     */
    public KMeansPlusPlusClusterer(VectorFloat<?>[] points, int k) {
        this(points, chooseInitialCentroids(points, k), UNWEIGHTED);
    }

    public KMeansPlusPlusClusterer(VectorFloat<?>[] points, int k, float anisotropicThreshold) {
        this(points, chooseInitialCentroids(points, k), anisotropicThreshold);
    }

    /**
     * Constructs a KMeansPlusPlusFloatClusterer with the specified points and initial centroids.
     * <p>
     * The initial centroids provided as a parameter are copied before modification.
     *
     * @param points the points to cluster (points[n][i] is the ith component of the nth point)
     * @param centroids the initial centroids.
     * @param anisotropicThreshold the threshold of relevance for anisotropic angular distance shaping, giving
     *        higher priority to parallel error.  Anisotropic shaping requires that your dataset be normalized
     *        to unit length.  Use a threshold of `UNWEIGHTED` for normal isotropic L2 distance.
     *        anisotropicThreshold is only valid when the supplied points are normalized to unit length.
     */
    public KMeansPlusPlusClusterer(VectorFloat<?>[] points, VectorFloat<?> centroids, float anisotropicThreshold) {
        if (Float.isNaN(anisotropicThreshold) || anisotropicThreshold < -1.0 || anisotropicThreshold >= 1.0) {
            // We use the weight function I(t >= T) from section 3.2 of the AVQ paper, which only considers
            // quantization loss when the dot product is above a threshold T.  For unit vectors, the dot product
            // is between -1 and 1, so the valid range for T is -1 <= t < 1.
            throw new IllegalArgumentException("Valid range for anisotropic threshold T is -1.0 <= t < 1.0");
        }

        this.points = points;
        this.k = centroids.length() / points[0].length();
        this.centroids = centroids.copy();
        this.anisotropicThreshold = anisotropicThreshold;

        centroidDenoms = new int[k];
        // initialize with empty vectors
        centroidNums = new VectorFloat<?>[k];
        for (int i = 0; i < centroidNums.length; i++) {
            centroidNums[i] = vectorTypeSupport.createFloatVector(points[0].length());
        }
        assignments = new int[points.length];

        initializeAssignedPoints();
    }

    /**
     * Compute the parallel cost multiplier for a given threshold and squared norm.
     * <p>
     * This uses the approximation derived in Theorem 3.4 of
     * "Accelerating Large-Scale Inference with Anisotropic Vector Quantization".
     */
    static float computeParallelCostMultiplier(double threshold, int dimensions) {
        assert Double.isFinite(threshold) : "threshold=" + threshold;
        // It's not completely clear that computing the PCM from a dot product threshold is better
        // than just allowing the user to specify the PCM directly, but it works well enough,
        // and it's arguably easier to reason about.
        double parallelCost = threshold * threshold; // we would divide by norm squared if we supported non-unit vectors
        double perpendicularCost = (1 - parallelCost) / (dimensions - 1);
        return (float) max(1.0, (parallelCost / perpendicularCost));
    }

    /**
     * Performs clustering on the provided set of points.
     *
     * @return a VectorFloat of cluster centroids.
     */
    public VectorFloat<?> cluster(int unweightedIterations, int anisotropicIterations) {
        // Always cluster unweighted first, it is significantly faster
        for (int i = 0; i < unweightedIterations; i++) {
            int changedCount = clusterOnceUnweighted();
            if (changedCount <= 0.01 * points.length) {
                break;
            }
        }

        // Optionally, refine using anisotropic clustering
        for (int i = 0; i < anisotropicIterations; i++) {
            int changedCount = clusterOnceAnisotropic();
            if (changedCount <= 0.01 * points.length) {
                break;
            }
        }

        return centroids;
    }

    // This is broken out as a separate public method to allow implementing OPQ efficiently
    public int clusterOnceUnweighted() {
        updateCentroidsUnweighted();
        return updateAssignedPointsUnweighted();
    }
    public int clusterOnceAnisotropic() {
        updateCentroidsAnisotropic();
        return updateAssignedPointsAnisotropic();
    }

    /**
     * Chooses the initial centroids for clustering.
     * The first centroid is chosen randomly from the data points. Subsequent centroids
     * are selected with a probability proportional to the square of their distance
     * to the nearest existing centroid. This ensures that the centroids are spread out
     * across the data and not initialized too closely to each other, leading to better
     * convergence and potentially improved final clusterings.
     *
     * @return an array of initial centroids.
     */
    private static VectorFloat<?> chooseInitialCentroids(VectorFloat<?>[] points, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive.");
        }
        if (k > points.length) {
            throw new IllegalArgumentException(String.format("Number of clusters %d cannot exceed number of points %d", k, points.length));
        }

        var random = ThreadLocalRandom.current();
        VectorFloat<?> centroids = vectorTypeSupport.createFloatVector(k * points[0].length());

        float[] distances = new float[points.length];
        Arrays.fill(distances, Float.MAX_VALUE);
        VectorFloat<?> distancesVector = vectorTypeSupport.createFloatVector(distances);
        int distancesLength = points.length;

        // Choose the first centroid randomly
        VectorFloat<?> firstCentroid = points[random.nextInt(points.length)];
        centroids.copyFrom(firstCentroid, 0, 0, firstCentroid.length());
        VectorFloat<?> newDistancesVector = vectorTypeSupport.createFloatVector(points.length);
        for (int i = 0; i < points.length; i++) {
            newDistancesVector.set(i, squareL2Distance(points[i], firstCentroid));
        }
        VectorUtil.minInPlace(distancesVector, newDistancesVector);

        // For each subsequent centroid
        for (int i = 1; i < k; i++) {
            float totalDistance = VectorUtil.sum(distancesVector);

            float r = random.nextFloat() * totalDistance;
            int selectedIdx = -1;
            for (int j = 0; j < distancesLength; j++) {
                r -= distancesVector.get(j);
                if (r < 1e-6) {
                    selectedIdx = j;
                    break;
                }
            }

            if (selectedIdx == -1) {
                selectedIdx = random.nextInt(points.length);
            }

            VectorFloat<?> nextCentroid = points[selectedIdx];
            centroids.copyFrom(nextCentroid, 0, i * nextCentroid.length(), nextCentroid.length());

            // Update distances, but only if the new centroid provides a closer distance
            // All entries of newDistancesVector is overwritten with the updated squareL2Distance value
            for (int j = 0; j < distancesLength; j++) {
                newDistancesVector.set(j, squareL2Distance(points[j], nextCentroid));
            }
            VectorUtil.minInPlace(distancesVector, newDistancesVector);
        }
        assertFinite(centroids);
        return centroids;
    }

    /**
     * Assigns points to the nearest cluster.  The results are stored as ordinals in `assignments`.
     * This method should only be called once after initial centroids are chosen.
     */
    private void initializeAssignedPoints() {
        for (int i = 0; i < points.length; i++) {
            VectorFloat<?> point = points[i];
            var newAssignment = getNearestCluster(point);
            centroidDenoms[newAssignment] = centroidDenoms[newAssignment] + 1;
            addInPlace(centroidNums[newAssignment], point);
            assignments[i] = newAssignment;
        }
    }

    /**
     * Assigns points to the nearest cluster.  The results are stored as ordinals in `assignments`,
     * and `centroidNums` and `centroidDenoms` are updated to accelerate centroid recomputation.
     * <p>
     * This method relies on valid assignments existing from either initializeAssignedPoints or
     * a previous invocation of this method.
     *
     * @return the number of points that changed clusters
     */
    private int updateAssignedPointsUnweighted() {
        int changedCount = 0;

        for (int i = 0; i < points.length; i++) {
            VectorFloat<?> point = points[i];
            var oldAssignment = assignments[i];
            var newAssignment = getNearestCluster(point);

            if (newAssignment != oldAssignment) {
                centroidDenoms[oldAssignment] = centroidDenoms[oldAssignment] - 1;
                subInPlace(centroidNums[oldAssignment], point);
                centroidDenoms[newAssignment] = centroidDenoms[newAssignment] + 1;
                addInPlace(centroidNums[newAssignment], point);
                assignments[i] = newAssignment;
                changedCount++;
            }
        }

        return changedCount;
    }

    /**
     * Assigns points to the nearest cluster.  Only `assignments` are updated, there is no precomputation
     * done for the centroid updates.
     *
     * @return the number of points that changed clusters
     */
    private int updateAssignedPointsAnisotropic() {
        float pcm = computeParallelCostMultiplier(anisotropicThreshold, points[0].length());

        // precompute norms for each centroid
        float[] cNormSquared = new float[k];
        for (int i = 0; i < k; i++) {
            cNormSquared[i] = dotProduct(centroids, i * points[0].length(),
                                         centroids, i * points[0].length(),
                                         points[0].length());
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

    /**
     * Calculates the weighted distance between two data points.
     */
    private float weightedDistance(VectorFloat<?> x, int centroid, float parallelCostMultiplier, float cNormSquared, float xNormSquared) {
        float cDotX = VectorUtil.dotProduct(centroids, centroid * x.length(), x, 0, x.length());
        float parallelErrorSubtotal = cDotX - xNormSquared;
        float residualSquaredNorm = cNormSquared - 2 * cDotX + xNormSquared;
        float parallelError = square(parallelErrorSubtotal);
        float perpendicularError = residualSquaredNorm - parallelError;

        return parallelCostMultiplier * parallelError + perpendicularError;
    }

    /**
     * Return the index of the closest centroid to the given point
     */
    private int getNearestCluster(VectorFloat<?> point) {
        float minDistance = Float.MAX_VALUE;
        int nearestCluster = 0;

        for (int i = 0; i < k; i++) {
            float distance = squareL2Distance(point, 0, centroids, i * point.length(), point.length());
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }

        return nearestCluster;
    }

    @SuppressWarnings({"AssertWithSideEffects", "ConstantConditions"})
    private static void assertFinite(VectorFloat<?> vector) {
        boolean assertsEnabled = false;
        assert assertsEnabled = true;

        if (assertsEnabled) {
            for (int i = 0; i < vector.length(); i++) {
                assert Float.isFinite(vector.get(i)) : "vector " + vector + " contains non-finite value";
            }
        }
    }

    /**
     * Calculates centroids from centroidNums/centroidDenoms updated during point assignment
     */
    private void updateCentroidsUnweighted() {
        for (int i = 0; i < k; i++) {
            var denom = centroidDenoms[i];
            if (denom == 0) {
                // no points assigned to this cluster
                initializeCentroidToRandomPoint(i);
            } else {
                var centroid = centroidNums[i].copy();
                scale(centroid, 1.0f / centroidDenoms[i]);
                centroids.copyFrom(centroid, 0, i * centroid.length(), centroid.length());
            }
        }
    }

    private void initializeCentroidToRandomPoint(int i) {
        var random = ThreadLocalRandom.current();
        centroids.copyFrom(points[random.nextInt(points.length)], 0, i * points[0].length(), points[0].length());
    }

    // Uses the algorithm given in appendix 7.5 of "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"
    private void updateCentroidsAnisotropic() {
        int dimensions = points[0].length();
        float pcm = computeParallelCostMultiplier(anisotropicThreshold, dimensions);
        // in most places we simplify
        //   loss = pcm' * parallel + ocm' * orthogonal
        // with pcm = pcm' / ocm', so
        //   loss ~ pcm * parallel + orthogonal
        // here we invert that as ocm = 1 / pcm
        float orthogonalCostMultiplier = 1.0f / pcm;

        // turn the point -> cluster assignments into cluster -> list of points so we don't
        // have to make `k` passes over `assignments`
        var pointsByCluster = new HashMap<Integer, List<Integer>>();
        for (int i = 0; i < assignments.length; i++) {
            pointsByCluster.computeIfAbsent(assignments[i], __ -> new ArrayList<>()).add(i);
        }

        for (int i = 0; i < k; i++) {
            var L = pointsByCluster.getOrDefault(i, List.of());
            if (L.isEmpty()) {
                // no points assigned to this cluster
                initializeCentroidToRandomPoint(i);
                continue;
            }

            // Calculate the mean and outer product sums for all points in the cluster k
            var mean = vectorTypeSupport.createFloatVector(dimensions);
            var outerProdSums = new Matrix(dimensions, dimensions);
            for (int j : L) {
                var point = points[j];
                // update mean
                addInPlace(mean, point);
                // update outer product sum
                float denom = dotProduct(point, point);
                if (denom > 0) {
                    var op = Matrix.outerProduct(point, point);
                    op.scale(1.0f / denom);
                    outerProdSums.addInPlace(op);
                }
            }
            outerProdSums.scale((1 - orthogonalCostMultiplier) / L.size());
            scale(mean, 1.0f / L.size());

            // Add (orthogonalCostMultiplier * I) factor
            for (int j = 0; j < dimensions; j++) {
                outerProdSums.addTo(j, j, orthogonalCostMultiplier);
            }

            // Invert the matrix and multiply with the mean to find the new centroid
            var invertedMatrix = outerProdSums.invert();
            centroids.copyFrom(invertedMatrix.multiply(mean), 0, i * dimensions, dimensions);
        }
    }

    /**
     * Computes the centroid of a list of points.
     */
    public static VectorFloat<?> centroidOf(List<VectorFloat<?>> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Can't compute centroid of empty points list");
        }

        VectorFloat<?> centroid = sum(points);
        scale(centroid, 1.0f / points.size());

        return centroid;
    }

    public VectorFloat<?> getCentroids() {
        return centroids;
    }
}

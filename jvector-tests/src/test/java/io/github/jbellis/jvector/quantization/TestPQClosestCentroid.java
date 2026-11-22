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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;
import java.util.Random;
import static org.junit.Assert.assertEquals;

public class TestPQClosestCentroid extends RandomizedTest {
    @Test
    public void testEncodingMatchesPreviousScanWithAndWithoutCentering() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var support = VectorizationProvider.getInstance().getVectorUtilSupport();
        Random random = new Random(1702);
        int[][] sizes = {{3, 0}, {8, 3}, {17, 11}};
        VectorFloat<?>[] codebooks = new VectorFloat<?>[sizes.length];
        for (int m = 0; m < sizes.length; m++) {
            codebooks[m] = vts.createFloatVector(256 * sizes[m][0]);
            for (int i = 0; i < codebooks[m].length(); i++) codebooks[m].set(i, random.nextFloat() * 4 - 2);
        }
        for (boolean centered : new boolean[]{false, true}) {
            var mean = centered ? vts.createFloatVector(28) : null;
            if (mean != null) for (int i = 0; i < mean.length(); i++) mean.set(i, random.nextFloat());
            var pq = new ProductQuantization(codebooks, 256, sizes, mean, KMeansPlusPlusClusterer.UNWEIGHTED);
            for (int query = 0; query < 200; query++) {
                var point = vts.createFloatVector(28);
                for (int i = 0; i < point.length(); i++) point.set(i, random.nextFloat() * 4 - 2);
                var actual = pq.encode(point);
                var referencePoint = mean == null ? point : VectorUtil.sub(point, mean);
                for (int m = 0; m < sizes.length; m++) {
                    float best = Float.MAX_VALUE;
                    int winner = 0;
                    for (int c = 0; c < 256; c++) {
                        float distance = support.squareDistance(referencePoint, sizes[m][1], codebooks[m], c * sizes[m][0], sizes[m][0]);
                        if (distance < best) { best = distance; winner = c; }
                    }
                    assertEquals(winner, Byte.toUnsignedInt(actual.get(m)));
                }
            }
        }
    }

    @Test
    public void testUnsignedCentroidCodesAndFirstTie() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var codebook = vts.createFloatVector(256 * 8);
        for (int c = 0; c < 256; c++) for (int d = 0; d < 8; d++) codebook.set(c * 8 + d, c);
        var pq = new ProductQuantization(new VectorFloat<?>[]{codebook}, 256, new int[][]{{8, 0}}, null, KMeansPlusPlusClusterer.UNWEIGHTED);
        for (int c : new int[]{0, 127, 128, 255}) {
            var point = vts.createFloatVector(8);
            for (int d = 0; d < 8; d++) point.set(d, c);
            assertEquals(c, Byte.toUnsignedInt(pq.encode(point).get(0)));
        }
        var midpoint = vts.createFloatVector(8);
        for (int d = 0; d < 8; d++) midpoint.set(d, 127.5f);
        assertEquals(127, Byte.toUnsignedInt(pq.encode(midpoint).get(0)));
    }
    @Test
    public void testFiniteDistanceAssertionIsPreserved() {
        org.junit.Assume.assumeTrue(VectorUtil.class.desiredAssertionStatus());
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var point = vts.createFloatVector(new float[]{0});
        var centroids = vts.createFloatVector(new float[]{0, 1});
        var pq = new ProductQuantization(new VectorFloat<?>[]{centroids}, 2,
                new int[][]{{1, 0}}, null, KMeansPlusPlusClusterer.UNWEIGHTED);
        centroids.set(1, Float.POSITIVE_INFINITY);
        org.junit.Assert.assertThrows(AssertionError.class, () -> pq.encode(point));
    }

}

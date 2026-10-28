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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.quantization.ASHScorer;
import io.github.jbellis.jvector.quantization.FusedASHLayout;
import java.util.Arrays;
import java.util.Random;
import org.junit.Test;

import static org.junit.Assert.*;

/** Native array contracts, layout tails, IEEE headers, and critical/ordinary FFM agreement. */
public class TestASHNativeScoring {
    @Test
    public void nativeNeighborhoodMatchesDecodedLutForWidthsTailsAndHeaders() {
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        org.junit.Assume.assumeTrue(backend.supportsAshNeighborhoodScoring());
        var random = new Random(527399);
        for (int groups : new int[] {1, 2, 7, 8, 9, 16, 17, 33, 65, 128, 256, 512, 1536}) {
            float[] lut = new float[groups * 16];
            for (int i = 0; i < lut.length; i++) lut[i] = random.nextFloat() - .5f;
            for (int blockSize : new int[] {8, 16, 32}) {
                for (int capacity : new int[] {1, 7, 23, 32, 37, 64, 128, 129}) {
                    // Four bits/dimension makes quantizedDim equal to the group count.
                    byte[] codes = new byte[FusedASHLayout.featureSize(capacity, groups, 4, blockSize)];
                    random.nextBytes(codes);
                    for (int lane = 0; lane < capacity; lane++) {
                        int offset = FusedASHLayout.blockOffset(lane / blockSize, groups, 4, blockSize);
                        FusedASHLayout.writeLaneHeader(codes, offset, lane % blockSize, groups, 4, blockSize,
                                random.nextFloat(), random.nextFloat() - .5f, (byte) 0);
                    }
                    float landmarkDot = random.nextFloat() - .5f;
                    float[] actual = new float[capacity + 1];
                    actual[capacity] = Float.NaN;
                    backend.ashScoreNeighborhood(codes, groups, blockSize, capacity, lut, landmarkDot, actual);
                    for (int lane = 0; lane < capacity; lane++) {
                        int offset = FusedASHLayout.blockOffset(lane / blockSize, groups, 4, blockSize);
                        float expected = ASHScorer.toSimilarity(FusedASHLayout.scoreLane(
                                codes, offset, lane % blockSize, groups, 4, blockSize, lut, new float[] {landmarkDot}));
                        assertEquals("groups=" + groups + " B=" + blockSize + " capacity=" + capacity + " lane=" + lane,
                                expected, actual[lane], 3e-5f + 3e-5f * Math.abs(expected));
                    }
                    assertTrue(Float.isNaN(actual[capacity]));
                    int header = FusedASHLayout.blockOffset(0, groups, 4, blockSize);
                    // Isolate half conversion from cancellation and reduction-order error.
                    float[] headerLut = new float[lut.length];
                    Arrays.fill(headerLut, .25f);
                    for (float scale : new float[] {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 0f,
                                 -0f, 0x1.0p-24f, 0x1.0p-14f, 65504f}) {
                        FusedASHLayout.writeLaneHeader(codes, header, 0, groups, 4, blockSize, scale, 0f, (byte) 0);
                        backend.ashScoreNeighborhood(
                                codes, groups, blockSize, capacity, headerLut, landmarkDot, actual);
                        float expected = ASHScorer.toSimilarity(FusedASHLayout.scoreLane(
                                codes, header, 0, groups, 4, blockSize, headerLut, new float[] {landmarkDot}));
                        if (Float.isNaN(expected))
                            assertTrue(Float.isNaN(actual[0]));
                        else
                            assertEquals("scale=" + scale + " groups=" + groups, expected, actual[0],
                                    Float.isInfinite(expected) ? 0f : 3e-5f + 3e-5f * Math.abs(expected));
                    }
                    FusedASHLayout.writeLaneHeader(codes, header, 0, groups, 4, blockSize, 1f, 0f, (byte) 1);
                    assertThrows(IllegalStateException.class,
                            ()
                                    -> backend.ashScoreNeighborhood(
                                            codes, groups, blockSize, capacity, lut, landmarkDot, actual));
                }
            }
        }
    }

    @Test
    public void nativeEntryRejectsInvalidShapesBeforeFFM() {
        var backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        org.junit.Assume.assumeTrue(backend.supportsAshNeighborhoodScoring());
        var codes = new byte[192]; // one code row + five header bytes per lane, B32
        var lut = new float[16];
        var out = new float[32];
        assertThrows(
                IllegalArgumentException.class, () -> backend.ashScoreNeighborhood(codes, 0, 32, 32, lut, 0f, out));
        assertThrows(
                IllegalArgumentException.class, () -> backend.ashScoreNeighborhood(codes, 1, 31, 32, lut, 0f, out));
        assertThrows(IndexOutOfBoundsException.class,
                () -> backend.ashScoreNeighborhood(new byte[0], 1, 32, 32, lut, 0f, out));
        assertThrows(IndexOutOfBoundsException.class,
                () -> backend.ashScoreNeighborhood(codes, 1, 32, 32, new float[15], 0f, out));
        assertThrows(IndexOutOfBoundsException.class,
                () -> backend.ashScoreNeighborhood(codes, 1, 32, 32, lut, 0f, new float[31]));
        assertThrows(IndexOutOfBoundsException.class,
                () -> backend.ashScoreNeighborhood(codes, Integer.MAX_VALUE, 32, 32, lut, 0f, out));
        float[] shared = new float[32];
        assertThrows(IllegalArgumentException.class,
                () -> backend.ashScoreNeighborhood(codes, 1, 32, 32, shared, 0f, shared));
        assertThrows(NullPointerException.class, () -> backend.ashScoreNeighborhood(null, 1, 32, 32, lut, 0f, out));
    }
}

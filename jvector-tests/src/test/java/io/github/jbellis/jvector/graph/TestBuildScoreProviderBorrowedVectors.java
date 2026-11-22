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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.*;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.util.*;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestBuildScoreProviderBorrowedVectors {
    private static class Borrowed implements RandomAccessVectorValues {
        final RandomAccessVectorValues source;
        final VectorFloat<?> scratch;
        Borrowed(RandomAccessVectorValues source) {
            this.source = source;
            scratch = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(source.dimension());
        }
        public int size() { return source.size(); }
        public int dimension() { return source.dimension(); }
        public boolean isValueShared() { return true; }
        public RandomAccessVectorValues copy() { return new Borrowed(source); }
        public VectorFloat<?> getVector(int ordinal) { source.getVectorInto(ordinal, scratch, 0); return scratch; }
    }
    @Test public void borrowedNodeScorersDoNotMaterializeAnchorsAtCreation() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var resident = new ListRandomAccessVectorValues(List.of(
                vts.createFloatVector(new float[] {1f, 0f}),
                vts.createFloatVector(new float[] {0f, 1f})), 2);
        var reads = new java.util.concurrent.atomic.AtomicInteger();
        class Counting extends Borrowed {
            Counting() { super(resident); }
            @Override public RandomAccessVectorValues copy() { return new Counting(); }
            @Override public VectorFloat<?> getVector(int ordinal) {
                reads.incrementAndGet();
                return super.getVector(ordinal);
            }
        }
        var bsp = BuildScoreProvider.randomAccessScoreProvider(new Counting(), VectorSimilarityFunction.DOT_PRODUCT);
        var search = bsp.searchProviderFor(0).scoreFunction();
        var diversity = bsp.diversityProviderFor(0).scoreFunction();
        var direct = bsp.diversityScoreFunctionFor(0);
        assertEquals(0, reads.get());
        float expected = VectorSimilarityFunction.DOT_PRODUCT.compare(resident.getVector(0), resident.getVector(1));
        assertEquals(expected, search.similarityTo(1), 0f);
        assertEquals(expected, diversity.similarityTo(1), 0f);
        assertEquals(expected, direct.similarityTo(1), 0f);
        assertEquals(6, reads.get());
    }

    @Test public void centroidNormalizesItsSizeSnapshot() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var resident = new ListRandomAccessVectorValues(Arrays.asList(
                vts.createFloatVector(new float[] {1f, 0f}), null,
                vts.createFloatVector(new float[] {0f, 1f})), 2);
        var sizes = new java.util.concurrent.atomic.AtomicInteger();
        var ranges = new java.util.concurrent.atomic.AtomicInteger();
        var source = new BatchedVectorValues() {
            public int size() { return sizes.getAndIncrement() == 0 ? 3 : 4; }
            public int dimension() { return 2; }
            public boolean isValueShared() { return false; }
            public RandomAccessVectorValues copy() { throw new UnsupportedOperationException(); }
            public VectorFloat<?> getVector(int ordinal) { return resident.getVector(ordinal); }
            public VectorCursor openRange(int start, int end) {
                ranges.incrementAndGet();
                throw new AssertionError("Unexpected bulk centroid optimization");
            }
            public VectorCursor openSelection(int[] ordinals, int offset, int count) {
                throw new AssertionError("Selection");
            }
        };
        var centroid = BuildScoreProvider.randomAccessScoreProvider(source,
                VectorSimilarityFunction.DOT_PRODUCT).approximateCentroid();
        assertEquals(0, ranges.get());
        assertEquals(1f / 3, centroid.get(0), 0f);
        assertEquals(1f / 3, centroid.get(1), 0f);
    }

    @Test public void residentAnchorsAreNotCopied() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var zero = vts.createFloatVector(new float[] {1f, 0f, 0f});
        var one = vts.createFloatVector(new float[] {0f, 1f, 0f});
        var source = new ListRandomAccessVectorValues(Arrays.asList(zero, one), 3);
        var bsp = BuildScoreProvider.randomAccessScoreProvider(source, VectorSimilarityFunction.DOT_PRODUCT);
        var search = bsp.searchProviderFor(0).scoreFunction();
        var diversity = bsp.diversityProviderFor(0).scoreFunction();
        var direct = bsp.diversityScoreFunctionFor(0);
        zero.set(0, 0f);
        zero.set(1, 1f);
        float expected = VectorSimilarityFunction.DOT_PRODUCT.compare(zero, one);
        assertEquals(expected, search.similarityTo(1), 0f);
        assertEquals(expected, diversity.similarityTo(1), 0f);
        assertEquals(expected, direct.similarityTo(1), 0f);
    }

    @Test public void nodeScorersRetainTheirAnchorAcrossOtherFactoryCalls() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        for (int dimensions : new int[] {3, 17, 768}) {
            var rows = new ArrayList<VectorFloat<?>>();
            for (int i = 0; i < 3; i++) {
                var v = vts.createFloatVector(dimensions);
                v.set(i == 1 ? 1 : 0, 1f);
                if (i == 2) v.set(2, 1f);
                VectorUtil.l2normalize(v);
                rows.add(v);
            }
            var resident = new ListRandomAccessVectorValues(rows, dimensions);
            for (var metric : VectorSimilarityFunction.values()) {
                var bsp = BuildScoreProvider.randomAccessScoreProvider(new Borrowed(resident), metric);
                var search = bsp.searchProviderFor(0).scoreFunction();
                var diversity = bsp.diversityProviderFor(0).scoreFunction();
                var direct = bsp.diversityScoreFunctionFor(0);
                bsp.searchProviderFor(1); // Reuses the same thread's borrowed source scratch.
                float expected = metric.compare(rows.get(0), rows.get(2));
                assertEquals("search " + metric, expected, search.similarityTo(2), 0f);
                assertEquals("diversity " + metric, expected, diversity.similarityTo(2), 0f);
                assertEquals("direct " + metric, expected, direct.similarityTo(2), 0f);
            }
        }
    }
}

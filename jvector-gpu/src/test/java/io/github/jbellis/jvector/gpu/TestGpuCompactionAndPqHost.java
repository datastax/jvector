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

package io.github.jbellis.jvector.gpu;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The host-side parts of GPU compaction and GPU product quantization, and their CPU fallbacks, without a GPU. */
class TestGpuCompactionAndPqHost {

    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.COSINE;

    @TempDir
    Path dir;

    /** Two on-disk segments of 300 random 16-dimensional vectors each. */
    private List<Path> segments(List<VectorFloat<?>> all) throws Exception {
        List<Path> paths = new ArrayList<>();
        for (int s = 0; s < 2; s++) {
            var vectors = new ListRandomAccessVectorValues(all.subList(s * 300, (s + 1) * 300), 16);
            var builder = new GraphIndexBuilder(vectors, SIMILARITY, 16, 100, 1.2f, 1.2f, true);
            Path path = dir.resolve("segment-" + s);
            OnDiskGraphIndex.write(builder.build(vectors), vectors, path);
            paths.add(path);
        }
        return paths;
    }

    private static List<VectorFloat<?>> random(int n, int dim) {
        var random = new Random(7);
        List<VectorFloat<?>> rows = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++) {
                v[d] = (float) random.nextGaussian();
            }
            rows.add(io.github.jbellis.jvector.vector.VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(v));
        }
        return rows;
    }

    @Test
    void collectsLiveVectorsAtTheirNewOrdinalsAndFallsBackToTheCompactor() throws Exception {
        List<VectorFloat<?>> all = random(600, 16);
        List<Path> paths = segments(all);
        List<ReaderSupplier> suppliers = new ArrayList<>();
        List<OnDiskGraphIndex> sources = new ArrayList<>();
        try {
            for (Path p : paths) {
                var rs = ReaderSupplierFactory.open(p);
                suppliers.add(rs);
                sources.add(OnDiskGraphIndex.load(rs));
            }
            // delete every 10th node of each segment; the survivors are numbered densely, segment by segment
            List<FixedBitSet> live = new ArrayList<>();
            List<Integer> expected = new ArrayList<>();
            for (int s = 0; s < 2; s++) {
                var bits = new FixedBitSet(300);
                for (int i = 0; i < 300; i++) {
                    if (i % 10 != 0) {
                        bits.set(i);
                        expected.add(s * 300 + i);
                    }
                }
                live.add(bits);
            }
            var compaction = new GpuCompaction(sources, live, SIMILARITY);
            var vectors = compaction.liveVectors();
            assertEquals(expected.size(), vectors.size());
            for (int ordinal = 0; ordinal < vectors.size(); ordinal++) {
                assertTrue(all.get(expected.get(ordinal)).equals(vectors.getVector(ordinal)), "vector at new ordinal " + ordinal);
            }

            assertThrows(IllegalStateException.class, compaction::ordinalMappers, "no mapping before compact()");

            // no TornadoVM runtime in the tests: compact() runs JVector's compactor, which assigns the ordinals;
            // ordinalMappers() must give every live node a distinct ordinal holding its vector
            Path out = dir.resolve("compacted");
            compaction.compact(out);
            List<OrdinalMapper> mappers = compaction.ordinalMappers();
            try (var rs = ReaderSupplierFactory.open(out); var merged = OnDiskGraphIndex.load(rs); var view = merged.getView()) {
                assertEquals(expected.size(), merged.size(0));
                var seen = new java.util.HashSet<Integer>();
                for (int s = 0; s < 2; s++) {
                    var bits = live.get(s);
                    for (int i = bits.nextSetBit(0); i != io.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS; i = bits.nextSetBit(i + 1)) {
                        int ordinal = mappers.get(s).oldToNew(i);
                        assertTrue(seen.add(ordinal), "ordinal " + ordinal + " assigned twice");
                        assertTrue(all.get(s * 300 + i).equals(view.getVector(ordinal)), "vector of source " + s + " node " + i);
                    }
                }
                assertEquals(expected.size(), seen.size());
            }
        } finally {
            for (var g : sources) {
                g.close();
            }
            for (var rs : suppliers) {
                rs.close();
            }
        }
    }

    @Test
    void pqFallsBackToJVectorWithoutTheRuntime() {
        var vectors = new ListRandomAccessVectorValues(random(2000, 32), 32);
        ProductQuantization gpu = GpuProductQuantization.compute(vectors, 8, 16, false);
        assertEquals(8, gpu.getSubspaceCount());
        assertEquals(16, gpu.getClusterCount());
        var codes = GpuProductQuantization.encodeAll(gpu, vectors);
        var reference = gpu.encodeAll(vectors, PhysicalCoreExecutor.pool());
        for (int i = 0; i < vectors.size(); i++) {
            assertTrue(reference.get(i).equals(codes.get(i)), "codes of " + i);
        }
    }

    @Test
    void pqCodebooksRoundTripThroughJVectorsFormat() {
        var vectors = new ListRandomAccessVectorValues(random(2000, 24), 24);
        for (boolean center : new boolean[] { false, true }) {
            ProductQuantization pq = ProductQuantization.compute(vectors, 5, 32, center); // uneven subspaces: 5,5,5,5,4
            var cb = GpuProductQuantization.codebooksOf(pq);
            assertArrayEquals(new int[] { 5, 0 }, cb.subspaces()[0]);
            assertArrayEquals(new int[] { 4, 20 }, cb.subspaces()[4]);
            var copy = GpuProductQuantization.toProductQuantization(
                    uk.ac.manchester.tornado.api.types.arrays.FloatArray.fromArray(cb.codebooks()), cb.subspaces(), cb.clusterCount(), cb.centroid());
            var original = pq.encodeAll(vectors, PhysicalCoreExecutor.pool());
            var roundTripped = copy.encodeAll(vectors, PhysicalCoreExecutor.pool());
            for (int i = 0; i < vectors.size(); i++) {
                assertTrue(original.get(i).equals(roundTripped.get(i)), "codes of " + i + ", centered " + center);
            }
        }
    }
}

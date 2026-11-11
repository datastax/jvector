/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedASH;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedNVQ;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;

/**
 * A feature header embedded in a graph serializes its compressor at the graph's version, not at
 * {@link OnDiskGraphIndex#CURRENT_VERSION}, so a graph written at an older version is laid out as
 * that version wrote it.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestFeatureHeaderVersion extends RandomizedTest {
    private static final int DIMENSION = 32, MAX_DEGREE = 16;

    private interface Body {
        void write(IndexWriter out, int version) throws IOException;
    }

    @Test
    public void featureHeadersAreWrittenAtTheGraphVersion() throws IOException {
        var ravv = new ListRandomAccessVectorValues(TestUtil.createRandomVectors(512, DIMENSION), DIMENSION);
        var pq = ProductQuantization.compute(ravv, 8, 256, false);
        var nvq = NVQuantization.compute(ravv, 2);
        var ash = AsymmetricHashing.initialize(ravv, AsymmetricHashing.RANDOM,
                AsymmetricHashing.HEADER_BITS + DIMENSION, 1);
        var fusedAsh = new FusedASH(MAX_DEGREE, ash);
        int blockSize = fusedAsh.blockSize();

        // Each feature's header is its compressor at the graph's version, plus any fields of its own.
        var cases = List.of(
                new Case(FeatureId.FUSED_PQ, new FusedPQ(MAX_DEGREE, pq), pq::write),
                new Case(FeatureId.NVQ_VECTORS, new NVQ(nvq), nvq::write),
                new Case(FeatureId.SEPARATED_NVQ, new SeparatedNVQ(nvq, 1234L), (out, v) -> {
                    nvq.write(out, v);
                    out.writeLong(1234L);
                }),
                new Case(FeatureId.FUSED_ASH, fusedAsh, (out, v) -> {
                    ash.write(out, v);
                    out.writeInt(blockSize);
                }));

        for (int version = 2; version <= OnDiskGraphIndex.CURRENT_VERSION; version++) {
            var format = GraphIndexFormatFactory.forVersion(version);
            for (var c : cases) {
                if (!format.getSupportedFeatures().contains(c.id)) continue;
                var features = new EnumMap<FeatureId, Feature>(FeatureId.class);
                features.put(c.id, c.feature);
                byte[] written = bytes(out -> format.writeHeaderFeatures(out, features));
                int v = version;
                byte[] expected = bytes(out -> c.body.write(out, v));
                // The format's own framing (feature ids, counts) precedes the header.
                byte[] header = Arrays.copyOfRange(written, written.length - expected.length, written.length);
                assertArrayEquals(c.id + " at version " + version, expected, header);
            }
        }
    }

    private static final class Case {
        final FeatureId id;
        final Feature feature;
        final Body body;

        Case(FeatureId id, Feature feature, Body body) {
            this.id = id;
            this.feature = feature;
            this.body = body;
        }
    }

    private interface Writes {
        void to(IndexWriter out) throws IOException;
    }

    private static byte[] bytes(Writes writes) throws IOException {
        try (var out = ByteBufferIndexWriter.create(1024 * 1024, false)) {
            writes.to(out);
            ByteBuffer data = out.getWrittenData();
            byte[] result = new byte[data.remaining()];
            data.get(result);
            return result;
        }
    }
}

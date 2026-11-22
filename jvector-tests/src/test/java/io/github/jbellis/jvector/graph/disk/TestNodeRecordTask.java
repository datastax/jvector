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

package io.github.jbellis.jvector.graph.disk;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestNodeRecordTask extends LuceneTestCase {
    @Test
    public void testCompleteRecordBatches() throws Exception {
        var root = Files.createTempDirectory("node-record-batches");
        var values = new ArrayList<VectorFloat<?>>();
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        for (int n = 0; n < 131; n++) {
            var vector = vts.createFloatVector(17);
            for (int d = 0; d < 17; d++) {
                vector.set(d, (float) Math.sin(n * 17 + d));
            }
            values.add(vector);
        }
        var ravv = new ListRandomAccessVectorValues(values, 17);
        try (var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.DOT_PRODUCT,
                4, 10, 1.2f, 1.2f, false)) {
            var graph = TestUtil.buildSequentially(builder, ravv);
            var feature = new InlineVectors(17);
            for (boolean holes : new boolean[]{false, true}) {
                int count = holes ? 262 : 131;
                OrdinalMapper mapper = holes ? new OrdinalMapper() {
                    public int maxOrdinal() { return 261; }
                    public int oldToNew(int n) { return 2 * n; }
                    public int newToOld(int n) { return n % 2 == 0 ? n / 2 : OMITTED; }
                } : new OrdinalMapper.IdentityMapper(130);
                int payload = Integer.BYTES * (2 + graph.getDegree(0)) + feature.featureSize();
                for (int stride : new int[]{payload}) {
                    for (boolean direct : new boolean[]{false, true}) {
                        var target = root.resolve("records-" + holes + "-" + stride + "-" + direct);
                        // Preserve bytes before L0 and force full batches plus a partial tail.
                        final int baseOffset = 83;
                        byte[] prefix = new byte[baseOffset];
                        Arrays.fill(prefix, (byte) 0x5a);
                        Files.write(target, prefix);
                        var buffer = direct ? ByteBuffer.allocateDirect(stride * 64)
                                : ByteBuffer.allocate(stride * 64);
                        buffer.order(ByteOrder.BIG_ENDIAN);
                        try (var view = graph.getView();
                             var channel = AsynchronousFileChannel.open(target, StandardOpenOption.WRITE)) {
                            var task = new NodeRecordTask(0, count, mapper, graph, view, List.of(feature),
                                    Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                                            n -> new InlineVectors.State(values.get(n))),
                                    stride, baseOffset, channel, buffer);
                            var padded = new NodeRecordTask(0, count, mapper, graph, view, List.of(feature),
                                    Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                                            n -> new InlineVectors.State(values.get(n))),
                                    4096, baseOffset, channel, buffer);
                            org.junit.Assert.assertThrows(IllegalArgumentException.class, padded::call);
                            task.call();
                        }
                        byte[] actual = Files.readAllBytes(target);
                        assertEquals(baseOffset + count * stride, actual.length);
                        assertArrayEquals(prefix, Arrays.copyOf(actual, baseOffset));
                        try (var view = graph.getView()) {
                            for (int n = 0; n < count; n++) {
                                var expected = ByteBuffer.allocate(stride).order(ByteOrder.BIG_ENDIAN);
                                expected.putInt(mapper.newToOld(n) == OrdinalMapper.OMITTED ? -1 : n);
                                int old = mapper.newToOld(n);
                                if (old == OrdinalMapper.OMITTED) {
                                    expected.position(expected.position() + feature.featureSize());
                                    expected.putInt(0);
                                    for (int d = 0; d < graph.getDegree(0); d++) expected.putInt(-1);
                                } else {
                                    for (int d = 0; d < 17; d++) expected.putFloat(values.get(old).get(d));
                                    var neighbors = view.getNeighborsIterator(0, old);
                                    expected.putInt(neighbors.size());
                                    int d = 0;
                                    while (neighbors.hasNext()) {
                                        expected.putInt(mapper.oldToNew(neighbors.nextInt()));
                                        d++;
                                    }
                                    for (; d < graph.getDegree(0); d++) expected.putInt(-1);
                                }
                                assertEquals("Record " + n + " holes=" + holes + " direct=" + direct, -1,
                                        Arrays.mismatch(actual, baseOffset + n * stride,
                                                baseOffset + (n + 1) * stride, expected.array(), 0, stride));
                            }
                        }
                    }
                }
            }
        } finally {
            TestUtil.deleteQuietly(root);
        }
    }
}

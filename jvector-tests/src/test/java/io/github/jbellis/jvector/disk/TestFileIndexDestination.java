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

package io.github.jbellis.jvector.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskSequentialGraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestFileIndexDestination extends RandomizedTest {
    private Path dir;
    private Path target;

    @Before
    public void setup() throws IOException {
        dir = Files.createTempDirectory(getClass().getSimpleName());
        target = dir.resolve("graph.index");
        Files.write(target, bytes("OLD"));
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(dir);
    }

    private static byte[] bytes(String s) {
        return s.getBytes(StandardCharsets.US_ASCII);
    }

    private static String text(Path p) throws IOException {
        return new String(Files.readAllBytes(p), StandardCharsets.US_ASCII);
    }

    /** Files in the test directory other than the named ones. */
    private List<String> strayFiles(String... expected) throws IOException {
        List<String> keep = List.of(expected);
        try (Stream<Path> s = Files.list(dir)) {
            return s.map(p -> p.getFileName().toString())
                    .filter(n -> !keep.contains(n))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    private static ImmutableGraphIndex buildGraph(RandomAccessVectorValues ravv) throws IOException {
        GraphIndexBuilder builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.EUCLIDEAN, 8, 32, 1.2f, 1.2f, false);
        ImmutableGraphIndex graph = TestUtil.buildSequentially(builder, ravv);
        builder.close();
        return graph;
    }

    private static RandomAccessVectorValues randomVectors(int size, int dimension, long seed) {
        List<VectorFloat<?>> vectors = new ArrayList<>(size);
        Random random = new Random(seed);
        for (int i = 0; i < size; i++) {
            vectors.add(TestUtil.randomVector(random, dimension));
        }
        return new ListRandomAccessVectorValues(vectors, dimension);
    }

    private static void streamGraph(ImmutableGraphIndex graph, RandomAccessVectorValues ravv, OutputReservation r) throws IOException {
        // The existing sequential writer runs unchanged over the reservation's stream and closes
        // it when done; the reservation stays open for complete().
        try (OnDiskSequentialGraphIndexWriter writer = new OnDiskSequentialGraphIndexWriter.Builder(graph, r.stream())
                .with(new InlineVectors(ravv.dimension()))
                .build()) {
            writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                    i -> new InlineVectors.State(ravv.getVector(i))));
        }
    }

    @Test
    public void standaloneCommitPublishesAtomically() throws IOException {
        IndexDestination dest = IndexDestination.toFile(target);
        try (OutputSession session = dest.open()) {
            try (OutputReservation r = session.reserve(OutputArtifact.GRAPH)) {
                assertEquals(OutputArtifact.GRAPH, r.artifact());
                IndexWriter out = r.stream();
                assertSame("one stream per reservation", out, r.stream());
                out.write(bytes("NEW-"));
                out.write(bytes("BODY"));
                assertEquals(8, out.position());
                assertEquals("target untouched while the session is open", "OLD", text(target));
                assertEquals(1, strayFiles("graph.index").size());
                r.complete();
            }
            assertEquals("target untouched until commit", "OLD", text(target));
            session.commit();
            assertEquals("NEW-BODY", text(target));
        }
        assertEquals(List.of(), strayFiles("graph.index"));
    }

    @Test
    public void abortLeavesTheTargetUntouched() throws IOException {
        IndexDestination dest = IndexDestination.toFile(target);
        try (OutputSession session = dest.open()) {
            try (OutputReservation r = session.reserve(OutputArtifact.GRAPH)) {
                r.stream().write(bytes("PARTIAL"));
                // no complete(): abort
            }
        }
        assertEquals("OLD", text(target));
        assertEquals(List.of(), strayFiles("graph.index"));
    }

    @Test
    public void completedButUncommittedIsDiscarded() throws IOException {
        IndexDestination dest = IndexDestination.toFile(target);
        try (OutputSession session = dest.open()) {
            try (OutputReservation r = session.reserve(OutputArtifact.GRAPH)) {
                r.stream().write(bytes("DONE"));
                r.complete();
            }
            // no commit(): abort
        }
        assertEquals("OLD", text(target));
        assertEquals(List.of(), strayFiles("graph.index"));
    }

    @Test
    public void lifecycleIsEnforced() throws IOException {
        IndexDestination dest = IndexDestination.toFile(target);
        OutputSession session = dest.open();
        try { session.reserve(OutputArtifact.COMPRESSED_VECTORS); fail("no placement"); } catch (IllegalArgumentException expected) { }

        OutputReservation r = session.reserve(OutputArtifact.GRAPH);
        try { session.reserve(OutputArtifact.GRAPH); fail("double reserve"); } catch (IllegalStateException expected) { }
        try { session.commit(); fail("commit with an open reservation"); } catch (IllegalStateException expected) { }

        r.stream().write(bytes("abc"));
        r.complete();
        try { r.complete(); fail("double complete"); } catch (IllegalStateException expected) { }
        try { r.stream(); fail("stream after complete"); } catch (IllegalStateException expected) { }
        try { session.commit(); fail("commit with a completed but unclosed reservation"); } catch (IllegalStateException expected) { }
        r.close();
        r.close(); // idempotent
        try { r.stream(); fail("stream after close"); } catch (IllegalStateException expected) { }

        session.commit();
        try { session.commit(); fail("double commit"); } catch (IllegalStateException expected) { }
        try { session.reserve(OutputArtifact.GRAPH); fail("reserve after commit"); } catch (IllegalStateException expected) { }
        session.close();
        session.close(); // idempotent
        try { session.commit(); fail("commit after close"); } catch (IllegalStateException expected) { }
        assertEquals("abc", text(target));
        assertEquals(List.of(), strayFiles("graph.index"));
    }

    @Test
    public void streamEncodesLikeDataOutputStream() throws IOException {
        float[] floats = {1.5f, -2.25f, 3e10f, Float.NaN};
        byte[] big = new byte[100_000];
        new Random(7).nextBytes(big);
        IndexDestination dest = IndexDestination.toFile(target);
        try (OutputSession session = dest.open()) {
            try (OutputReservation r = session.reserve(OutputArtifact.GRAPH)) {
                IndexWriter w = r.stream();
                w.writeInt(0x01020304);
                w.writeLong(-1L);
                w.writeFloat(2.5f);
                w.writeDouble(Math.PI);
                w.writeShort(-2);
                w.writeChar('x');
                w.writeByte(7);
                w.writeBoolean(true);
                w.writeUTF("héllo wörld");
                w.writeBytes("abc");
                w.writeChars("de");
                w.writeFloats(floats, 1, 2);
                w.write(big, 100, 5000);       // smaller than the buffer
                w.write(big);                  // larger than the buffer: written straight through
                w.write(new byte[]{9, 8, 7});
                assertEquals(4 + 8 + 4 + 8 + 2 + 2 + 1 + 1 + (2 + 13) + 3 + 4 + 8 + 5000 + big.length + 3, w.position());
                r.complete();
            }
            session.commit();
        }
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(Files.readAllBytes(target)));
        assertEquals(0x01020304, in.readInt());
        assertEquals(-1L, in.readLong());
        assertEquals(2.5f, in.readFloat(), 0f);
        assertEquals(Math.PI, in.readDouble(), 0d);
        assertEquals(-2, in.readShort());
        assertEquals('x', in.readChar());
        assertEquals(7, in.readByte());
        assertTrue(in.readBoolean());
        assertEquals("héllo wörld", in.readUTF());
        byte[] abc = new byte[3];
        in.readFully(abc);
        assertArrayEquals(new byte[]{'a', 'b', 'c'}, abc);
        assertEquals('d', in.readChar());
        assertEquals('e', in.readChar());
        assertEquals(-2.25f, in.readFloat(), 0f);
        assertEquals(3e10f, in.readFloat(), 0f);
        byte[] middle = new byte[5000];
        in.readFully(middle);
        assertArrayEquals(Arrays.copyOfRange(big, 100, 5100), middle);
        byte[] whole = new byte[big.length];
        in.readFully(whole);
        assertArrayEquals(big, whole);
        byte[] tail = new byte[3];
        in.readFully(tail);
        assertArrayEquals(new byte[]{9, 8, 7}, tail);
        assertEquals(-1, in.read());
    }

    @Test
    public void regionModeWritesOnlyItsRegion() throws IOException {
        Path container = dir.resolve("container.bin");
        byte[] original = new byte[96];
        new Random(1).nextBytes(original);
        Files.write(container, original);

        IndexDestination dest = IndexDestination.inFile(container, 64);
        try (OutputSession session = dest.open()) {
            try (OutputReservation r = session.reserve(OutputArtifact.GRAPH)) {
                IndexWriter out = r.stream();
                assertEquals("positions are region-relative", 0, out.position());
                out.write(bytes("REGION-BYTES"));
                assertEquals(12, out.position());
                r.complete();
            }
            session.commit();
        }
        byte[] after = Files.readAllBytes(container);
        assertEquals("region mode never truncates", original.length, after.length);
        assertArrayEquals(Arrays.copyOfRange(original, 0, 64), Arrays.copyOfRange(after, 0, 64));
        assertArrayEquals(bytes("REGION-BYTES"), Arrays.copyOfRange(after, 64, 76));
        assertArrayEquals(Arrays.copyOfRange(original, 76, 96), Arrays.copyOfRange(after, 76, 96));

        // Abort in region mode leaves the file in place as well.
        try (OutputSession session = dest.open()) {
            try (OutputReservation r = session.reserve(OutputArtifact.GRAPH)) {
                r.stream().write(bytes("XX"));
            }
        }
        assertTrue(Files.exists(container));
        assertEquals(List.of(), strayFiles("graph.index", "container.bin"));
    }

    @Test
    public void multipleArtifactsPublishTogether() throws IOException {
        Path pq = dir.resolve("graph.pq");
        Map<OutputArtifact, Path> paths = new EnumMap<>(OutputArtifact.class);
        paths.put(OutputArtifact.GRAPH, target);
        paths.put(OutputArtifact.COMPRESSED_VECTORS, pq);
        IndexDestination dest = IndexDestination.toFiles(paths);
        try (OutputSession session = dest.open()) {
            try (OutputReservation g = session.reserve(OutputArtifact.GRAPH)) {
                g.stream().write(bytes("GRAPH"));
                g.complete();
            }
            try (OutputReservation c = session.reserve(OutputArtifact.COMPRESSED_VECTORS)) {
                c.stream().write(bytes("CODES"));
                c.complete();
            }
            assertFalse(Files.exists(pq));
            session.commit();
        }
        assertEquals("GRAPH", text(target));
        assertEquals("CODES", text(pq));
        assertEquals(List.of(), strayFiles("graph.index", "graph.pq"));

        // The destination is reusable: a second session replaces what it reserves.
        try (OutputSession session = dest.open()) {
            try (OutputReservation g = session.reserve(OutputArtifact.GRAPH)) {
                g.stream().write(bytes("G2"));
                g.complete();
            }
            session.commit();
        }
        assertEquals("G2", text(target));
        assertEquals("CODES", text(pq));
    }

    @Test
    public void graphStreamedIntoAStandaloneFileLoads() throws IOException {
        int dimension = 16;
        RandomAccessVectorValues ravv = randomVectors(200, dimension, 99);
        ImmutableGraphIndex graph = buildGraph(ravv);

        IndexDestination dest = IndexDestination.toFile(target);
        try (OutputSession session = dest.open()) {
            try (OutputReservation r = session.reserve(OutputArtifact.GRAPH)) {
                streamGraph(graph, ravv, r);
                r.complete();
            }
            session.commit();
        }
        try (SimpleMappedReader.Supplier supplier = new SimpleMappedReader.Supplier(target);
             OnDiskGraphIndex loaded = OnDiskGraphIndex.load(supplier)) {
            TestUtil.assertGraphEquals(graph, loaded);
        }
        assertEquals(List.of(), strayFiles("graph.index"));
    }

    @Test
    public void graphStreamedIntoAContainerRegionLoads() throws IOException {
        int dimension = 8;
        int prefix = 128;
        RandomAccessVectorValues ravv = randomVectors(60, dimension, 42);
        ImmutableGraphIndex graph = buildGraph(ravv);

        // A host container: its own header, then the graph, then its own trailer.
        Path container = dir.resolve("container.bin");
        byte[] header = new byte[prefix];
        new Random(8).nextBytes(header);
        Files.write(container, header);

        IndexDestination dest = IndexDestination.inFile(container, prefix);
        try (OutputSession session = dest.open()) {
            try (OutputReservation r = session.reserve(OutputArtifact.GRAPH)) {
                streamGraph(graph, ravv, r);
                r.complete();
            }
            session.commit();
        }
        long graphEnd = Files.size(container);
        assertTrue(graphEnd > prefix);
        Files.write(container, new byte[64], java.nio.file.StandardOpenOption.APPEND);

        byte[] after = Files.readAllBytes(container);
        assertArrayEquals("host header untouched", header, Arrays.copyOfRange(after, 0, prefix));

        // The host reads the graph back from the known offset. The stream's positions started at
        // 0 at the region start, so the offsets the footer stores are relative to that origin;
        // a whole-file reader therefore loads header-first from the offset rather than trusting
        // the container's end.
        try (SimpleMappedReader.Supplier supplier = new SimpleMappedReader.Supplier(container);
             OnDiskGraphIndex loaded = OnDiskGraphIndex.load(supplier, prefix, false)) {
            TestUtil.assertGraphEquals(graph, loaded);
        }
    }
}

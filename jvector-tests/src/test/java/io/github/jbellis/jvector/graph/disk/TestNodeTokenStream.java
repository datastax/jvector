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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.disk.SimpleWriter;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** The token stream codec on its own: both encodings round-trip, the locator is strict, the encoder is strict. */
public class TestNodeTokenStream extends RandomizedTest {
    private Path dir;

    @Before
    public void setup() throws IOException {
        dir = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(dir);
    }

    /** A synthetic base layer: dead holes, levels, near and far neighbours in both directions. */
    private static final class Synthetic {
        final int n;
        final int degree;
        final int maxLevel;
        final boolean[] live;
        final int[] level;
        final int[] key;
        final int[][] neighbors;

        Synthetic(int n, int degree, int maxLevel, Random rnd) {
            this.n = n;
            this.degree = degree;
            this.maxLevel = maxLevel;
            live = new boolean[n];
            level = new int[n];
            key = new int[n];
            neighbors = new int[n][];
            for (int i = 0; i < n; i++) {
                live[i] = rnd.nextInt(10) != 0;
                level[i] = live[i] ? rnd.nextInt(maxLevel + 1) : 0;
                key[i] = live[i] ? rnd.nextInt() : 0;
                if (!live[i]) {
                    neighbors[i] = new int[0];
                    continue;
                }
                int count = rnd.nextInt(degree + 1);
                neighbors[i] = new int[count];
                for (int k = 0; k < count; k++) {
                    // mostly near (the similarity-ordinal case), sometimes anywhere
                    neighbors[i][k] = rnd.nextInt(4) == 0 ? rnd.nextInt(n)
                                    : Math.floorMod(i + rnd.nextInt(64) - 32, n);
                }
            }
        }
    }

    private Path write(Synthetic s, byte encoding) throws IOException {
        return write(s, encoding, SimilarityKey.NONE);
    }

    private Path write(Synthetic s, byte encoding, byte keyFunction) throws IOException {
        Path path = dir.resolve("stream_" + encoding + "_" + keyFunction);
        try (var out = new SimpleWriter(path)) {
            var enc = new NodeTokenStream.Encoder(out, encoding, s.n, s.degree, s.maxLevel, keyFunction);
            for (int i = 0; i < s.n; i++) {
                enc.node(i, s.live[i], s.level[i], keyFunction == SimilarityKey.NONE ? 0 : s.key[i]);
                for (int nb : s.neighbors[i]) {
                    enc.neighbor(nb);
                }
            }
            long len = enc.finish();
            assertEquals(len, enc.bytes());
            assertEquals(s.n, enc.nodes());
        }
        return path;
    }

    private void verify(Synthetic s, Path path, byte encoding) throws IOException {
        verify(s, path, encoding, SimilarityKey.NONE);
    }

    private void verify(Synthetic s, Path path, byte encoding, byte keyFunction) throws IOException {
        try (var supplier = new SimpleMappedReader.Supplier(path)) {
            NodeTokenStream.Section section;
            try (var in = supplier.get()) {
                // the file is exactly [section][trailer]: the "footer" begins at its end
                section = NodeTokenStream.locate(in, in.length(), 0);
            }
            assertNotNull(section);
            assertEquals(0, section.offset);
            assertEquals(Files.size(path) - NodeTokenStream.TRAILER_SIZE, section.length);
            try (var r = new NodeTokenStream.Reader(supplier.get(), section)) {
                assertEquals(encoding, r.encoding);
                assertEquals(s.n, r.nodeCount);
                assertEquals(s.degree, r.degree);
                assertEquals(s.maxLevel, r.maxLevel);
                assertEquals(keyFunction, r.keyFunction);
                assertEquals(NodeTokenStream.VERSION, r.version);
                int i = 0;
                while (r.next()) {
                    assertEquals(i, r.ordinal());
                    assertEquals(s.live[i], r.live());
                    assertEquals(s.level[i], r.level());
                    assertEquals(keyFunction == SimilarityKey.NONE ? 0 : s.key[i], r.key());
                    assertArrayEquals("node " + i, s.neighbors[i], Arrays.copyOf(r.neighbors(), r.neighborCount()));
                    i++;
                }
                assertEquals(s.n, i);
            }
        }
    }

    @Test
    public void testRoundTripBothEncodings() throws IOException {
        // 70k nodes puts NODE deltas at one byte and neighbour deltas across the varint width boundaries.
        var s = new Synthetic(70_000, 12, 3, new Random(0xC0FFEE));
        Path delta = write(s, NodeTokenStream.ENCODING_DELTA);
        Path raw = write(s, NodeTokenStream.ENCODING_RAW);
        verify(s, delta, NodeTokenStream.ENCODING_DELTA);
        verify(s, raw, NodeTokenStream.ENCODING_RAW);
        assertTrue("delta form must be smaller than raw: " + Files.size(delta) + " vs " + Files.size(raw),
                Files.size(delta) < Files.size(raw));
    }

    @Test
    public void testRoundTripWithKeys() throws IOException {
        var s = new Synthetic(20_000, 8, 2, new Random(42));
        for (byte enc : new byte[] {NodeTokenStream.ENCODING_DELTA, NodeTokenStream.ENCODING_RAW}) {
            Path p = write(s, enc, SimilarityKey.RANDOM_PROJECTION);
            verify(s, p, enc, SimilarityKey.RANDOM_PROJECTION);
        }
    }

    /** A version-1 section (no key function byte, no keys) written by hand must still decode. */
    @Test
    public void testReadsVersionOne() throws IOException {
        Path p = dir.resolve("v1");
        try (var out = new SimpleWriter(p)) {
            long start = out.position();
            out.writeInt(NodeTokenStream.SECTION_MAGIC);
            out.writeInt(NodeTokenStream.VERSION_1);
            out.writeByte(NodeTokenStream.ENCODING_RAW);
            out.writeInt(2);   // nodes
            out.writeInt(4);   // degree
            out.writeByte(1);  // max level
            out.writeByte(0x80 | 0x40 | 1); out.writeInt(0);   // node 0: live, level 1
            out.writeByte(0x01); out.writeInt(1);              //   edge to 1
            out.writeByte(0x80 | 0x40);      out.writeInt(1);  // node 1: live, level 0
            out.writeByte(0x01); out.writeInt(0);              //   edge to 0
            long length = out.position() - start;
            out.writeLong(length);
            out.writeInt(NodeTokenStream.TRAILER_MAGIC);
        }
        try (var supplier = new SimpleMappedReader.Supplier(p)) {
            NodeTokenStream.Section section;
            try (var in = supplier.get()) {
                section = NodeTokenStream.locate(in, in.length(), 0);
            }
            assertNotNull(section);
            try (var r = new NodeTokenStream.Reader(supplier.get(), section)) {
                assertEquals(NodeTokenStream.VERSION_1, r.version);
                assertEquals(SimilarityKey.NONE, r.keyFunction);
                assertTrue(r.next());
                assertEquals(0, r.ordinal());
                assertEquals(1, r.level());
                assertEquals(1, r.neighborCount());
                assertEquals(1, r.neighbors()[0]);
                assertEquals(0, r.key());
                assertTrue(r.next());
                assertEquals(1, r.ordinal());
                assertEquals(0, r.neighbors()[0]);
                assertFalse(r.next());
            }
        }
    }

    @Test
    public void testEmptyAndSingleNode() throws IOException {
        var s = new Synthetic(1, 4, 0, new Random(1));
        s.live[0] = true;
        s.neighbors[0] = new int[0];
        Path p = write(s, NodeTokenStream.ENCODING_DELTA);
        verify(s, p, NodeTokenStream.ENCODING_DELTA);
    }

    @Test
    public void testLocateRejectsFilesWithoutASection() throws IOException {
        Path junk = dir.resolve("junk");
        byte[] bytes = new byte[4096];
        new Random(7).nextBytes(bytes);
        Files.write(junk, bytes);
        try (var supplier = new SimpleMappedReader.Supplier(junk); var in = supplier.get()) {
            assertNull(NodeTokenStream.locate(in, in.length(), 0));
            assertNull("too short for a trailer", NodeTokenStream.locate(in, 8, 0));
            assertNull("below the lower bound", NodeTokenStream.locate(in, in.length(), in.length() - 4));
        }
    }

    @Test
    public void testEncoderIsStrict() throws IOException {
        Path p = dir.resolve("strict");
        try (var out = new SimpleWriter(p)) {
            var enc = new NodeTokenStream.Encoder(out, NodeTokenStream.ENCODING_DELTA, 3, 2, 0);
            try {
                enc.neighbor(1);
                fail("neighbour before any node");
            } catch (IllegalStateException expected) {
            }
            enc.node(0, true, 0);
            try {
                enc.node(2, true, 0);
                fail("ordinals must be contiguous");
            } catch (IllegalStateException expected) {
            }
            enc.node(1, false, 0);
            try {
                enc.finish();
                fail("every ordinal below nodeCount must be written");
            } catch (IllegalStateException expected) {
            }
            enc.node(2, true, 0);
            enc.finish();
        }
        assertFalse(Files.size(p) == 0);
    }
}

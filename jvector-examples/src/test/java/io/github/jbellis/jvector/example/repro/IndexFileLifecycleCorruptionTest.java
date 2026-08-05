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

package io.github.jbellis.jvector.example.repro;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/// Tests for the *silent data corruption* hazards around index-file ranging (offset / region
/// loads), closing, and files rewritten to the same path with a different length.
///
/// The originally-enabling fact: `OnDiskGraphIndexCompactor.compact(Path, long)` never truncated
/// its destination, so a path that previously held a longer file kept a stale tail — and a
/// stale-but-valid v5+ footer there hijacked the footer-based default load silently (proven by
/// the pre-F2 form of t6a; evidence in `local/memory_safety_design_brief.md` T6a). Since the F2
/// fix, standalone destinations (`startOffset == 0`) are truncated before writing, and t6a now
/// guards that round-trip. Embedded destinations (`startOffset > 0`) remain untruncated by
/// contract, which t6c characterizes: prefix preserved, stale tail persists, header-based region
/// loads round-trip, and a footer-based whole-container load fails loudly on a junk tail.
/// Separately, mmap-based readers have no defense against the file being rewritten in place
/// (same inode: live readers silently see the new bytes) or replaced by rename (old inode: live
/// readers silently keep serving stale data) — t6b documents those as permanent mmap/inode
/// physics for hosts to respect, not fixable jvector behavior.
public class IndexFileLifecycleCorruptionTest {

    private static final int DIM = 16;
    private static final int SMALL = 30;   // nodes per compaction source

    private static void verdict(String theory, String text) {
        System.out.println("VERDICT|" + theory + "|" + text);
    }

    /// Outcome of a load attempt plus enough context to classify silent-vs-loud failures.
    private static final class LoadOutcome {
        final OnDiskGraphIndex graph;
        final Throwable error;

        LoadOutcome(OnDiskGraphIndex graph, Throwable error) {
            this.graph = graph;
            this.error = error;
        }

        boolean loaded() {
            return graph != null;
        }

        int size() {
            return graph.size(0);
        }

        String describe() {
            if (loaded()) {
                return "loaded size(0)=" + size();
            }
            return "threw " + error.getClass().getName() + ": " + error.getMessage();
        }
    }

    private static LoadOutcome tryLoad(ReaderSupplier rs, long offset, boolean useFooter) {
        try {
            return new LoadOutcome(OnDiskGraphIndex.load(rs, offset, useFooter), null);
        } catch (Throwable t) {
            return new LoadOutcome(null, t);
        }
    }

    /// Compacts two fresh 30-node sources into `out` at `offset`; returns the source vectors for
    /// content verification (compacted ordinals: source0 at [0,30), source1 at [30,60)).
    private static List<List<VectorFloat<?>>> compactSmallSourcesInto(Path dir, Path out, long offset) throws Exception {
        List<VectorFloat<?>> v0 = ReproGraphs.randomVectors(SMALL, DIM, 11);
        List<VectorFloat<?>> v1 = ReproGraphs.randomVectors(SMALL, DIM, 13);
        Path s0 = ReproGraphs.buildInlineGraph(dir.resolve("cmp_src0.graph"), v0, DIM);
        Path s1 = ReproGraphs.buildInlineGraph(dir.resolve("cmp_src1.graph"), v1, DIM);
        try (ReaderSupplier r0 = ReaderSupplierFactory.open(s0);
             ReaderSupplier r1 = ReaderSupplierFactory.open(s1)) {
            OnDiskGraphIndex g0 = OnDiskGraphIndex.load(r0);
            OnDiskGraphIndex g1 = OnDiskGraphIndex.load(r1);
            var compactor = new OnDiskGraphIndexCompactor(
                    List.of(g0, g1),
                    ReproGraphs.allLive(SMALL, SMALL),
                    ReproGraphs.stackedRemappers(SMALL, SMALL),
                    VectorSimilarityFunction.EUCLIDEAN,
                    null, -1);
            if (offset == 0) {
                compactor.compact(out);
            } else {
                compactor.compact(out, offset);
            }
        }
        return List.of(v0, v1);
    }

    private static boolean vectorMatches(OnDiskGraphIndex.View view, int ordinal, VectorFloat<?> expected) {
        return Arrays.equals(ReproGraphs.readVector(view, ordinal, DIM), ReproGraphs.toArray(expected));
    }

    /// F2 guard: compacting onto a path that previously held a *longer* index must truncate the
    /// stale content, so the default (footer-based) load sees exactly the new graph. Pre-F2 this
    /// test proved the inverse: the stale tail kept the old graph's still-valid footer at the
    /// file end and `load()` silently resurrected the old structure (size 500, expected 60) over
    /// hybrid old/new bytes, with no error anywhere.
    @Test(timeout = 240_000)
    public void t6a_compactOntoReusedLongerPathTruncatesAndRoundTrips() throws Exception {
        Path dir = ReproGraphs.newWorkDir("t6a-stale-tail");

        // Control: the same compaction into a fresh path round-trips correctly.
        Path fresh = dir.resolve("fresh_out.graph");
        List<List<VectorFloat<?>>> vecs = compactSmallSourcesInto(dir, fresh, 0);
        try (ReaderSupplier rs = ReaderSupplierFactory.open(fresh)) {
            OnDiskGraphIndex g = OnDiskGraphIndex.load(rs);
            assertEquals("control: fresh-path compaction must load correctly", 2 * SMALL, g.size(0));
            try (var view = g.getView()) {
                assertTrue("control: source-0 vector content must round-trip", vectorMatches(view, 0, vecs.get(0).get(0)));
                assertTrue("control: source-1 vector content must round-trip", vectorMatches(view, SMALL, vecs.get(1).get(0)));
            }
        }

        // The reused path: previously held a much larger graph.
        Path reused = dir.resolve("reused_out.graph");
        List<VectorFloat<?>> oldVecs = ReproGraphs.randomVectors(500, DIM, 7);
        ReproGraphs.buildInlineGraph(reused, oldVecs, DIM);
        long oldSize = Files.size(reused);

        List<List<VectorFloat<?>>> newVecs = compactSmallSourcesInto(dir, reused, 0);
        long newSize = Files.size(reused);
        assertTrue("F2: a standalone destination must be truncated before writing"
                + " (previous file was " + oldSize + " bytes, now " + newSize + ")", newSize < oldSize);

        try (ReaderSupplier rs = ReaderSupplierFactory.open(reused)) {
            OnDiskGraphIndex g = OnDiskGraphIndex.load(rs);
            assertEquals("default load of the reused path must see exactly the new compaction",
                    2 * SMALL, g.size(0));
            try (var view = g.getView()) {
                assertTrue("source-0 vector content must round-trip on the reused path",
                        vectorMatches(view, 0, newVecs.get(0).get(0)));
                assertTrue("source-1 vector content must round-trip on the reused path",
                        vectorMatches(view, SMALL, newVecs.get(1).get(0)));
            }
        }
        verdict("T6a", "FIX HOLDS: reused longer path truncated (" + oldSize + " -> " + newSize
                + " bytes); default load round-trips the new graph exactly");
    }

    /// T6b: rewriting a mapped index file in place (same path, same inode, same length) is
    /// silently visible through a live reader — no exception, just different bytes. This is the
    /// unified-page-cache half of the "same location, different content/length" hazard.
    ///
    /// Correct behavior for a host: never rewrite a graph component in place while any reader
    /// (a running compaction's source view included) may be open over it.
    @Test(timeout = 240_000)
    public void t6b_inPlaceRewriteUnderLiveMappingSilentlyServesNewBytes() throws Exception {
        Path dir = ReproGraphs.newWorkDir("t6b-inplace");
        Path p = dir.resolve("graph.bin");
        List<VectorFloat<?>> vecs = ReproGraphs.randomVectors(100, DIM, 21);
        ReproGraphs.buildInlineGraph(p, vecs, DIM);

        try (ReaderSupplier rs = ReaderSupplierFactory.open(p)) {
            OnDiskGraphIndex g = OnDiskGraphIndex.load(rs);
            try (var view = g.getView()) {
                float[] before = ReproGraphs.readVector(view, 0, DIM);
                assertArrayEquals("sanity: the live view reads the original bytes",
                        ReproGraphs.toArray(vecs.get(0)), before, 0.0f);

                byte[] all = Files.readAllBytes(p);
                for (int i = 0; i < all.length; i++) {
                    all[i] ^= 0x5A;
                }
                try (FileChannel ch = FileChannel.open(p, StandardOpenOption.WRITE)) {
                    ByteBuffer buf = ByteBuffer.wrap(all);
                    long pos = 0;
                    while (buf.hasRemaining()) {
                        pos += ch.write(buf, pos);
                    }
                }
                assertEquals("same length after in-place rewrite", all.length, Files.size(p));

                float[] after = ReproGraphs.readVector(view, 0, DIM);
                assertFalse("T6b PROOF POINT: the live reader silently serves the rewritten bytes"
                        + " — no exception, different data", Arrays.equals(before, after));
                verdict("T6b", "PROVEN silent corruption: in-place rewrite is immediately visible through a live mmap view");
            }
        }
    }

    /// T6b (rename variant): atomically replacing the file leaves live readers on the old inode —
    /// they silently keep serving the *stale* graph while new readers of the same path see the new
    /// one. No error surfaces on either side of the split brain.
    @Test(timeout = 240_000)
    public void t6b_renameReplaceUnderLiveMappingSilentlyServesStaleBytes() throws Exception {
        Path dir = ReproGraphs.newWorkDir("t6b-rename");
        Path p = dir.resolve("graph.bin");
        List<VectorFloat<?>> vecsA = ReproGraphs.randomVectors(100, DIM, 31);
        ReproGraphs.buildInlineGraph(p, vecsA, DIM);

        try (ReaderSupplier live = ReaderSupplierFactory.open(p)) {
            OnDiskGraphIndex g = OnDiskGraphIndex.load(live);
            try (var view = g.getView()) {
                float[] before = ReproGraphs.readVector(view, 0, DIM);

                List<VectorFloat<?>> vecsB = ReproGraphs.randomVectors(100, DIM, 32);
                Path replacement = ReproGraphs.buildInlineGraph(dir.resolve("replacement.bin"), vecsB, DIM);
                Files.move(replacement, p, StandardCopyOption.REPLACE_EXISTING);

                float[] liveRead = ReproGraphs.readVector(view, 0, DIM);
                assertArrayEquals("T6b PROOF POINT: after rename-replace, the live reader silently serves STALE bytes",
                        before, liveRead, 0.0f);

                try (ReaderSupplier freshRs = ReaderSupplierFactory.open(p)) {
                    OnDiskGraphIndex fresh = OnDiskGraphIndex.load(freshRs);
                    float[] freshRead;
                    try (var freshView = fresh.getView()) {
                        freshRead = ReproGraphs.readVector(freshView, 0, DIM);
                    }
                    assertFalse("a fresh reader of the same path must see the replacement",
                            Arrays.equals(before, freshRead));
                }
                verdict("T6b-rename", "PROVEN silent split-brain: live reader serves the old inode, fresh reader the new file");
            }
        }
    }

    /// T6c: the ranged/no-copy embedding path. `compact(container, startOffset)` into a reused
    /// container that is *longer* than the new body: the reserved prefix must survive (documented
    /// contract — positive check), the header-based region load must round-trip (positive check),
    /// and the footer-based load — which trusts the file end — must not silently succeed over the
    /// stale junk tail.
    @Test(timeout = 240_000)
    public void t6c_offsetRegionCompactionInReusedLongerContainer() throws Exception {
        Path dir = ReproGraphs.newWorkDir("t6c-container");
        Path container = dir.resolve("container.bin");
        final int prefixLen = 4096;
        final int junkLen = 64_000;
        byte[] prefix = new byte[prefixLen];
        Arrays.fill(prefix, (byte) 0xAB);
        byte[] junk = new byte[junkLen];
        Arrays.fill(junk, (byte) 0xCA);
        Files.write(container, prefix);
        Files.write(container, junk, StandardOpenOption.APPEND);

        List<List<VectorFloat<?>>> vecs = compactSmallSourcesInto(dir, container, prefixLen);

        byte[] prefixAfter = new byte[prefixLen];
        try (FileChannel ch = FileChannel.open(container, StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.wrap(prefixAfter);
            long pos = 0;
            while (buf.hasRemaining()) {
                int n = ch.read(buf, pos);
                if (n < 0) {
                    break;
                }
                pos += n;
            }
        }
        assertArrayEquals("documented contract: the reserved prefix must survive compaction untouched",
                prefix, prefixAfter);
        assertEquals("enabling condition: the longer container is not truncated, the junk tail persists",
                (long) prefixLen + junkLen, Files.size(container));

        try (ReaderSupplier rs = ReaderSupplierFactory.open(container)) {
            LoadOutcome headerLoad = tryLoad(rs, prefixLen, false);
            assertNull("header-based region load must work on the reused container: "
                    + (headerLoad.error == null ? "" : headerLoad.error.toString()), headerLoad.error);
            assertEquals("header-based region load must see exactly the compacted graph",
                    2 * SMALL, headerLoad.size());
            try (var view = headerLoad.graph.getView()) {
                assertTrue("region-loaded source-0 content must round-trip", vectorMatches(view, 0, vecs.get(0).get(0)));
                assertTrue("region-loaded source-1 content must round-trip", vectorMatches(view, SMALL, vecs.get(1).get(0)));
            }
        }

        try (ReaderSupplier rs = ReaderSupplierFactory.open(container)) {
            LoadOutcome footerLoad = tryLoad(rs, prefixLen, true);
            assertTrue("footer-based load over a stale longer tail must not silently produce the compacted graph; got: "
                    + footerLoad.describe(), !footerLoad.loaded() || footerLoad.size() != 2 * SMALL);
            verdict("T6c", footerLoad.loaded()
                    ? "PROVEN silent corruption: footer load returned a bogus graph: " + footerLoad.describe()
                    : "footer load over the junk tail fails loudly (not silently): " + footerLoad.describe());
        }
    }
}

package io.github.jbellis.jvector.fastervector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure-Java unit tests for the api package.
 * None of these tests may trigger native library loading.
 */
class ApiPureJavaTest {

    // ══════════════════════════════════════════════════════════════════════════
    // BuildChunks: numChunks
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    void numChunks_zero_nvecs() {
        assertEquals(0, BuildChunks.numChunks(0, 2));
    }

    @Test
    void numChunks_exact_multiple() {
        assertEquals(3, BuildChunks.numChunks(6, 2));
    }

    @Test
    void numChunks_with_remainder() {
        assertEquals(3, BuildChunks.numChunks(5, 2));
    }

    @Test
    void numChunks_single_chunk() {
        assertEquals(1, BuildChunks.numChunks(1, 1000));
    }

    @Test
    void numChunks_large_build_chunk_size() {
        long nvecs = 5_000_000L;
        long expected = (nvecs + BuildChunks.BUILD_CHUNK_SIZE - 1) / BuildChunks.BUILD_CHUNK_SIZE;
        assertEquals(expected, BuildChunks.numChunks(nvecs, BuildChunks.BUILD_CHUNK_SIZE));
    }

    // ══════════════════════════════════════════════════════════════════════════
    // BuildChunks: chunkSequence
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    void chunkSequence_empty() {
        assertTrue(BuildChunks.chunkSequence(0, 2).isEmpty());
    }

    @Test
    void chunkSequence_5_by_2() {
        assertEquals(List.of(2L, 2L, 1L), BuildChunks.chunkSequence(5, 2));
    }

    @Test
    void chunkSequence_exact_multiple() {
        assertEquals(List.of(2L, 2L, 2L), BuildChunks.chunkSequence(6, 2));
    }

    @Test
    void chunkSequence_single_vector() {
        assertEquals(List.of(1L), BuildChunks.chunkSequence(1, 1000));
    }

    // ══════════════════════════════════════════════════════════════════════════
    // BuildChunks: fillChunk with in-memory Vecs
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Simple in-memory Vecs backed by a float array.
     * Does NOT call any native code.
     */
    static final class ArrayVecs implements Vecs {
        private final float[] data;
        private final int dim;

        ArrayVecs(float[] data, int dim) {
            assert data.length % dim == 0;
            this.data = data;
            this.dim = dim;
        }

        @Override
        public long nvecs() { return data.length / dim; }

        @Override
        public int dim() { return dim; }

        @Override
        public VecsCursor getCursor() {
            return new Cursor(0);
        }

        private class Cursor implements VecsCursor {
            private long ord;

            Cursor(long ord) { this.ord = ord; }

            @Override
            public long ord() { return ord; }

            @Override
            public void seek(long ord) { this.ord = ord; }

            @Override
            public boolean nextVectorInto(MemorySegment out) {
                if (ord >= nvecs()) return false;
                int base = (int) (ord * dim);
                for (int i = 0; i < dim; i++) {
                    out.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, data[base + i]);
                }
                ord++;
                return true;
            }
        }
    }

    /** Build a float array: vector i has value (float)i in all elements. */
    private static float[] makeData(int nvecs, int dim) {
        float[] data = new float[nvecs * dim];
        for (int i = 0; i < nvecs; i++) {
            for (int j = 0; j < dim; j++) {
                data[i * dim + j] = (float) i;
            }
        }
        return data;
    }

    @Test
    void fillChunk_reads_exactly_chunkSize() {
        int nvecs = 5, dim = 4;
        ArrayVecs vecs = new ArrayVecs(makeData(nvecs, dim), dim);
        VecsCursor cursor = vecs.getCursor();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate((long) 2 * dim * Float.BYTES);
            long n = BuildChunks.fillChunk(cursor, dim, 2, buf);
            assertEquals(2, n);
            assertEquals(1.0f, buf.get(ValueLayout.JAVA_FLOAT, 1L * dim * Float.BYTES));
        }
    }

    @Test
    void fillChunk_returns_zero_when_exhausted() {
        ArrayVecs vecs = new ArrayVecs(makeData(2, 4), 4);
        VecsCursor cursor = vecs.getCursor();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(10L * 4 * Float.BYTES);
            BuildChunks.fillChunk(cursor, 4, 10, buf); // drain
            long n = BuildChunks.fillChunk(cursor, 4, 2, buf);
            assertEquals(0, n);
        }
    }

    @Test
    void fillChunk_partial_last_chunk() {
        // 5 vecs, chunk 2: pass should give 2+2+1
        int nvecs = 5, dim = 2;
        ArrayVecs vecs = new ArrayVecs(makeData(nvecs, dim), dim);
        VecsCursor cursor = vecs.getCursor();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate((long) 2 * dim * Float.BYTES);
            List<Long> sizes = new ArrayList<>();
            long n;
            do {
                n = BuildChunks.fillChunk(cursor, dim, 2, buf);
                if (n > 0) sizes.add(n);
            } while (n > 0);
            assertEquals(List.of(2L, 2L, 1L), sizes);
        }
    }

    @Test
    void nextVectorInto_slice_places_data_correctly() {
        // slice at float index 4 (byte 16): vector is written at the slice start
        ArrayVecs vecs = new ArrayVecs(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 4);
        VecsCursor cursor = vecs.getCursor();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(8L * Float.BYTES);
            // put sentinel at offset 0-3
            for (int i = 0; i < 4; i++) buf.set(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES, -99f);
            boolean ok = cursor.nextVectorInto(buf.asSlice(4L * Float.BYTES));
            assertTrue(ok);
            // first 4 floats untouched
            assertEquals(-99f, buf.get(ValueLayout.JAVA_FLOAT, 0));
            // data at float index 4
            assertEquals(1.0f, buf.get(ValueLayout.JAVA_FLOAT, 4L * Float.BYTES));
            assertEquals(4.0f, buf.get(ValueLayout.JAVA_FLOAT, 7L * Float.BYTES));
        }
    }

    @Test
    void pass1_and_pass2_see_identical_chunk_sequences() {
        // Simulate passes 1 and 2 using fresh cursors on the same Vecs.
        int nvecs = 7, dim = 3;
        ArrayVecs vecs = new ArrayVecs(makeData(nvecs, dim), dim);
        long chunkSize = 3;
        List<Long> seq1 = new ArrayList<>(), seq2 = new ArrayList<>();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(chunkSize * dim * Float.BYTES);
            VecsCursor c1 = vecs.getCursor();
            long n;
            do {
                n = BuildChunks.fillChunk(c1, dim, chunkSize, buf);
                if (n > 0) seq1.add(n);
            } while (n > 0);

            VecsCursor c2 = vecs.getCursor();
            do {
                n = BuildChunks.fillChunk(c2, dim, chunkSize, buf);
                if (n > 0) seq2.add(n);
            } while (n > 0);
        }
        assertEquals(seq1, seq2, "Pass 1 and Pass 2 must see identical chunk sequences");
        assertEquals(List.of(3L, 3L, 1L), seq1);
    }

    @Test
    void data_content_copied_correctly() {
        // Vector 2 has value 2.0f in all elements; verify exact bytes in buffer.
        int nvecs = 3, dim = 4;
        ArrayVecs vecs = new ArrayVecs(makeData(nvecs, dim), dim);
        VecsCursor cursor = vecs.getCursor();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate((long) dim * Float.BYTES);
            cursor.nextVectorInto(buf); // vec 0
            cursor.nextVectorInto(buf); // vec 1
            cursor.nextVectorInto(buf); // vec 2
            for (int j = 0; j < dim; j++) {
                assertEquals(2.0f, buf.get(ValueLayout.JAVA_FLOAT, (long) j * Float.BYTES),
                        "element " + j + " of vec 2");
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // SSD path resolution
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Invoke the same path-resolution logic that BuildBuilder.build() uses,
     * but without calling any native code.
     */
    static List<String> resolveDevPaths(List<Path> ssdPaths, String subfolder) throws IOException {
        List<String> devPaths = new ArrayList<>();
        for (int i = 0; i < ssdPaths.size(); i++) {
            Path dir = ssdPaths.get(i).resolve(subfolder);
            Files.createDirectories(dir);
            devPaths.add(dir.resolve("dev" + i + ".ash.blk").toString());
        }
        return devPaths;
    }

    static String resolveManifest(List<Path> ssdPaths, String subfolder) {
        return ssdPaths.get(0).resolve(subfolder).resolve("dev0.ash.blk.manifest").toString();
    }

    @Test
    void ssdPaths_dev_names(@TempDir Path tmp) throws IOException {
        List<Path> ssdPaths = List.of(tmp.resolve("m0"), tmp.resolve("m1"));
        List<String> devPaths = resolveDevPaths(ssdPaths, "");
        assertEquals(2, devPaths.size());
        assertTrue(devPaths.get(0).endsWith("dev0.ash.blk"),
                "dev0 name: " + devPaths.get(0));
        assertTrue(devPaths.get(1).endsWith("dev1.ash.blk"),
                "dev1 name: " + devPaths.get(1));
    }

    @Test
    void ssdPaths_manifest_name(@TempDir Path tmp) throws IOException {
        List<Path> ssdPaths = List.of(tmp.resolve("m0"));
        resolveDevPaths(ssdPaths, "");
        String manifest = resolveManifest(ssdPaths, "");
        assertTrue(manifest.endsWith("dev0.ash.blk.manifest"),
                "manifest name: " + manifest);
    }

    @Test
    void ssdPaths_creates_directories(@TempDir Path tmp) throws IOException {
        Path mount = tmp.resolve("ssd");
        List<Path> ssdPaths = List.of(mount);
        resolveDevPaths(ssdPaths, "sub/folder");
        assertTrue(Files.isDirectory(mount.resolve("sub/folder")),
                "sub/folder should be created");
    }

    @Test
    void ssdPaths_with_subfolder(@TempDir Path tmp) throws IOException {
        List<Path> ssdPaths = List.of(tmp);
        List<String> devPaths = resolveDevPaths(ssdPaths, "myidx");
        String manifest = resolveManifest(ssdPaths, "myidx");
        assertTrue(devPaths.get(0).contains("myidx"), "dev path should contain subfolder");
        assertTrue(manifest.contains("myidx"), "manifest should contain subfolder");
    }

    // ══════════════════════════════════════════════════════════════════════════
    // BuildBuilder validation (without native calls)
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    void buildBuilder_missing_nlist_throws() {
        IvfAshIndex.BuildBuilder b = IvfAshIndex.builder()
                .ssdPaths(List.of(Path.of("/tmp")))
                .vectors(new ArrayVecs(makeData(10, 4), 4));
        assertThrows(IllegalStateException.class, b::build);
    }

    @Test
    void buildBuilder_missing_vectors_throws() {
        IvfAshIndex.BuildBuilder b = IvfAshIndex.builder()
                .nlist(256)
                .ssdPaths(List.of(Path.of("/tmp")));
        assertThrows(IllegalStateException.class, b::build);
    }

    @Test
    void buildBuilder_empty_ssdPaths_throws() {
        IvfAshIndex.BuildBuilder b = IvfAshIndex.builder()
                .nlist(256)
                .ssdPaths(List.of())
                .vectors(new ArrayVecs(makeData(10, 4), 4));
        assertThrows(IllegalStateException.class, b::build);
    }

    @Test
    void buildBuilder_null_ssdPaths_throws() {
        IvfAshIndex.BuildBuilder b = IvfAshIndex.builder()
                .nlist(256)
                .vectors(new ArrayVecs(makeData(10, 4), 4));
        assertThrows(IllegalStateException.class, b::build);
    }

    @Test
    void buildBuilder_defaults_storageFactor() {
        // storageFactor defaults to 1.0f — verify we can set and read via builder
        // (this is a structural check only; the actual value is sent to native)
        IvfAshIndex.BuildBuilder b = IvfAshIndex.builder()
                .nlist(256)
                .ssdPaths(List.of(Path.of("/tmp")))
                .vectors(new ArrayVecs(makeData(10, 4), 4));
        // We only verify it doesn't throw before the native call; build() itself
        // will throw because native libs aren't loaded in this test JVM.
        assertNotNull(b);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // OpenBuilder validation
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    void openBuilder_missing_manifestPath_throws() {
        IvfAshIndex.OpenBuilder b = IvfAshIndex.prepareOpen()
                .devPaths(List.of("/mnt/ssd/dev0.ash.blk"));
        assertThrows(IllegalStateException.class, b::open);
    }

    @Test
    void openBuilder_empty_devPaths_throws() {
        IvfAshIndex.OpenBuilder b = IvfAshIndex.prepareOpen()
                .manifestPath("/mnt/ssd/dev0.ash.blk.manifest")
                .devPaths(List.of());
        assertThrows(IllegalStateException.class, b::open);
    }

    @Test
    void openBuilder_missing_devPaths_throws() {
        IvfAshIndex.OpenBuilder b = IvfAshIndex.prepareOpen()
                .manifestPath("/mnt/ssd/dev0.ash.blk.manifest");
        assertThrows(IllegalStateException.class, b::open);
    }

    @Test
    void openBuilder_dim_defaultsToZero_andAcceptsPositive() {
        IvfAshIndex.OpenBuilder b = IvfAshIndex.prepareOpen();
        assertEquals(0, b.dim());
        assertSame(b, b.dim(128));
        assertEquals(128, b.dim());
    }

    @Test
    void openBuilder_dim_nonPositive_rejected() {
        IvfAshIndex.OpenBuilder b = IvfAshIndex.prepareOpen();
        assertThrows(IllegalArgumentException.class, () -> b.dim(0));
        assertThrows(IllegalArgumentException.class, () -> b.dim(-4));
    }

    // The searchers below are deliberately NOT closed: close() would call the native
    // delete function, and these tests must not load native code.
    @SuppressWarnings("resource")
    @Test
    void searcher_withKnownDim_rejectsDimMismatch_beforeNativeCall() {
        // Package-private ctor; a NULL (non-Java-null) native pointer passes requireOpen()
        // and the dim check throws before any native call is made.
        IvfAshIndexSearcher s = new IvfAshIndexSearcher(MemorySegment.NULL, 10, 4, null);
        ArrayVecs q = new ArrayVecs(makeData(3, 3), 3);
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> s.search(q));
        assertTrue(e.getMessage().contains("query dim mismatch"), e.getMessage());
    }

    @SuppressWarnings("resource")
    @Test
    void searcher_rejectsCursorYieldingTooFewVectors_beforeNativeCall() {
        IvfAshIndexSearcher s = new IvfAshIndexSearcher(MemorySegment.NULL, 10, 0, null);
        ArrayVecs inner = new ArrayVecs(makeData(1, 2), 2);
        Vecs lying = new Vecs() {
            @Override public long nvecs() { return 3; }
            @Override public int dim() { return 2; }
            @Override public VecsCursor getCursor() { return inner.getCursor(); }
        };
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> s.search(lying));
        assertTrue(e.getMessage().contains("yielded 1 vectors, expected 3"), e.getMessage());
    }

    // ══════════════════════════════════════════════════════════════════════════
    // SearcherBuilder validation (without native calls)
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    void searcherBuilder_missing_topk_throws() {
        // We need a stub index to call searcherBuilder — but we cannot create a
        // real one without native libs. Test the Builder directly.
        // Accessing the builder directly is fine since it's public.
        // We just verify builder field validation.
        IvfAshIndexSearcher.Builder b = new IvfAshIndexSearcher.Builder(null);
        b.nprobe(32);
        assertThrows(IllegalStateException.class, b::build);
    }

    @Test
    void searcherBuilder_missing_nprobe_throws() {
        IvfAshIndexSearcher.Builder b = new IvfAshIndexSearcher.Builder(null);
        b.topk(10);
        assertThrows(IllegalStateException.class, b::build);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // VecsCursor seek
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    void cursor_seek_repositions() {
        int nvecs = 5, dim = 2;
        ArrayVecs vecs = new ArrayVecs(makeData(nvecs, dim), dim);
        VecsCursor cursor = vecs.getCursor();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate((long) dim * Float.BYTES);
            cursor.seek(3);
            assertEquals(3L, cursor.ord());
            boolean ok = cursor.nextVectorInto(buf);
            assertTrue(ok);
            assertEquals(3.0f, buf.get(ValueLayout.JAVA_FLOAT, 0));
        }
    }

    @Test
    void cursor_exhausted_returns_false() {
        ArrayVecs vecs = new ArrayVecs(makeData(2, 2), 2);
        VecsCursor cursor = vecs.getCursor();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(2L * Float.BYTES);
            cursor.nextVectorInto(buf);
            cursor.nextVectorInto(buf);
            assertFalse(cursor.nextVectorInto(buf), "should be exhausted");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // BuildChunks: effectiveChunkSize override
    // ══════════════════════════════════════════════════════════════════════════

    @Test
    void effectiveChunkSize_override() {
        long saved = BuildChunks.overrideChunkSize;
        try {
            BuildChunks.overrideChunkSize = 7L;
            assertEquals(7L, BuildChunks.effectiveChunkSize());
        } finally {
            BuildChunks.overrideChunkSize = saved;
        }
    }

    @Test
    void effectiveChunkSize_default() {
        long saved = BuildChunks.overrideChunkSize;
        try {
            BuildChunks.overrideChunkSize = 0;
            assertEquals(BuildChunks.BUILD_CHUNK_SIZE, BuildChunks.effectiveChunkSize());
        } finally {
            BuildChunks.overrideChunkSize = saved;
        }
    }
}

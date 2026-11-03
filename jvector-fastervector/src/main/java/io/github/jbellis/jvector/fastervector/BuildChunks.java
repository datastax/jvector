package io.github.jbellis.jvector.fastervector;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * Package-private helper for the chunked streaming build.
 * Isolated here so it can be unit-tested without touching any native code.
 */
final class BuildChunks {

    /**
     * Default number of vectors per chunk during the streaming SSD build.
     * Mirrors {@code BUILD_CHUNK_SIZE} in {@code api/ivf_ash_index.rs}.
     */
    static final long BUILD_CHUNK_SIZE = 2_000_000L;

    /**
     * Overridable chunk size (for tests).  Package-private callers may set
     * this before calling {@code IvfAshIndex.BuilderBuild.build()} in a test.
     * Not thread-safe — intended for single-threaded test use only.
     */
    static long overrideChunkSize = 0; // 0 = not overridden

    private BuildChunks() {}

    /** Effective chunk size: override (if set) or the build constant. */
    static long effectiveChunkSize() {
        return overrideChunkSize > 0 ? overrideChunkSize : BUILD_CHUNK_SIZE;
    }

    /**
     * Compute the number of chunks needed to cover {@code nvecs} vectors at
     * {@code chunkSize} vectors per chunk (ceiling division).
     */
    static long numChunks(long nvecs, long chunkSize) {
        if (nvecs <= 0 || chunkSize <= 0) return 0;
        return (nvecs + chunkSize - 1) / chunkSize;
    }

    /**
     * Fill a chunk buffer from {@code cursor} with up to {@code chunkSize}
     * vectors of dimension {@code dim}, writing float data into the
     * pre-allocated off-heap segment {@code buf} starting at float offset 0.
     *
     * @param cursor    positioned cursor (may be mid-stream)
     * @param dim       floats per vector
     * @param chunkSize maximum number of vectors to read
     * @param buf       pre-allocated segment of at least
     *                  {@code chunkSize * dim * Float.BYTES} bytes
     * @return number of vectors actually read (0 = exhausted)
     */
    static long fillChunk(VecsCursor cursor, int dim, long chunkSize, MemorySegment buf) {
        long n = 0;
        while (n < chunkSize) {
            if (!cursor.nextVectorInto(buf.asSlice(n * dim * Float.BYTES))) {
                break;
            }
            n++;
        }
        return n;
    }

    /**
     * Compute the sequence of chunk vector-counts for a dataset of size
     * {@code nvecs} with the given {@code chunkSize}.  Used for validation
     * in tests (passes 1 and 2 must see identical sequences).
     */
    static List<Long> chunkSequence(long nvecs, long chunkSize) {
        List<Long> seq = new ArrayList<>();
        long remaining = nvecs;
        while (remaining > 0) {
            long n = Math.min(remaining, chunkSize);
            seq.add(n);
            remaining -= n;
        }
        return seq;
    }
}

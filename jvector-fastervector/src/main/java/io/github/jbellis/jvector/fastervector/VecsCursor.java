package io.github.jbellis.jvector.fastervector;

import java.lang.foreign.MemorySegment;

/**
 * A stateful cursor over a {@link Vecs} dataset.
 * <p>
 * Mirrors the Rust {@code VecsCursor} trait in {@code api/mod.rs}.
 * <p>
 * VecsCursor is not inherently closeable in this design — simple in-memory
 * implementations need no cleanup. Implementations that hold OS resources
 * (e.g. mapped files) should implement {@link AutoCloseable} directly. A
 * default no-op {@code close()} would silently hide accidental leaks in IO
 * implementations, so it is intentionally not provided here.
 */
public interface VecsCursor {

    /** Ordinal of the next vector to be read. */
    long ord();

    /** Reposition the cursor so that ordinal {@code ord} will be read next. */
    void seek(long ord);

    /**
     * Reads the vector at the current ordinal into the start of {@code out},
     * then advances the ordinal. Callers that want to write at a non-zero
     * position should pass a slice (e.g. {@link MemorySegment#asSlice(long)}).
     *
     * @param out off-heap (or heap) segment large enough to hold {@code dim} floats
     * @return {@code true} if a vector was written; {@code false} when exhausted
     */
    boolean nextVectorInto(MemorySegment out);
}

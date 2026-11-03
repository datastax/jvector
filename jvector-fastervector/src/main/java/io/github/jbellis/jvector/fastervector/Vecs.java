package io.github.jbellis.jvector.fastervector;

/**
 * A read-only dataset of float vectors with random access via cursors.
 * <p>
 * Mirrors the Rust {@code Vecs} trait in {@code api/mod.rs}.
 */
public interface Vecs {

    /** Number of vectors in this dataset. */
    long nvecs();

    /** Dimension (number of floats) per vector. */
    int dim();

    /**
     * Create a new cursor positioned at ordinal 0.
     * The caller is responsible for closing the cursor if it implements {@link AutoCloseable}.
     */
    VecsCursor getCursor();
}

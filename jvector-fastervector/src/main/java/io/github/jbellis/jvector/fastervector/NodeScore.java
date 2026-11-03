package io.github.jbellis.jvector.fastervector;

/**
 * A single neighbour returned by a search.
 *
 * @param label neighbour ID (IdxT = int64)
 * @param score distance / score (float32)
 */
public record NodeScore(long label, float score) {}

/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/**
 * Provides utility classes for array manipulation, bit operations, memory estimation, and
 * concurrent data structures used throughout JVector.
 *
 * <p>This package contains low-level utility classes adapted from Apache Lucene and extended
 * for JVector's needs. The utilities focus on:
 *
 * <ul>
 *   <li><b>Array operations</b>: {@link io.github.jbellis.jvector.util.ArrayUtil} provides
 *       efficient methods for growing and copying arrays of various primitive types and objects,
 *       with memory-aligned size calculations for optimal performance.
 *   <li><b>Bit manipulation</b>: {@link io.github.jbellis.jvector.util.BitUtil},
 *       {@link io.github.jbellis.jvector.util.FixedBitSet},
 *       {@link io.github.jbellis.jvector.util.SparseFixedBitSet}, and
 *       {@link io.github.jbellis.jvector.util.GrowableBitSet} offer various bit set implementations
 *       and bitwise operations optimized for different use cases.
 *   <li><b>Memory estimation</b>: {@link io.github.jbellis.jvector.util.RamUsageEstimator} provides
 *       utilities for estimating object sizes and memory overhead.
 *   <li><b>Data structures</b>: Specialized collections including
 *       {@link io.github.jbellis.jvector.util.BoundedLongHeap},
 *       {@link io.github.jbellis.jvector.util.DenseIntMap},
 *       {@link io.github.jbellis.jvector.util.SparseIntMap} for efficient storage and retrieval.
 *   <li><b>Threading utilities</b>: {@link io.github.jbellis.jvector.util.PhysicalCoreExecutor}
 *       and {@link io.github.jbellis.jvector.util.ExplicitThreadLocal} for managing concurrent
 *       operations.
 * </ul>
 *
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * // Growing an array with optimal size calculation
 * float[] vectors = new float[100];
 * vectors = ArrayUtil.grow(vectors, 200); // Grows to >= 200, over-allocating for efficiency
 *
 * // Using bit sets for neighbor tracking
 * FixedBitSet visited = new FixedBitSet(graphSize);
 * visited.set(nodeId);
 * if (visited.get(neighborId)) {
 *     // neighbor already visited
 * }
 * }</pre>
 *
 * <p>Most classes in this package are final and provide only static methods. The implementations
 * prioritize performance and memory efficiency, making them suitable for use in vector search
 * operations where arrays and bit sets are manipulated frequently.
 *
 * @see io.github.jbellis.jvector.util.ArrayUtil
 * @see io.github.jbellis.jvector.util.RamUsageEstimator
 * @see io.github.jbellis.jvector.util.FixedBitSet
 */
package io.github.jbellis.jvector.util;

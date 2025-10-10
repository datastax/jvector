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

/**
 * Provides type abstractions and utilities for vector and byte sequence operations.
 * <p>
 * This package defines interfaces and utilities that enable flexible and efficient
 * manipulation of vector data and byte sequences in the JVector library. The primary
 * goals are to:
 * <ul>
 *   <li>Abstract over different storage implementations (arrays, buffers, memory-mapped files)</li>
 *   <li>Provide a uniform API for vector and byte operations</li>
 *   <li>Enable performance optimizations through pluggable storage backends</li>
 *   <li>Support both primitive and object-based vector representations</li>
 * </ul>
 *
 * <h2>Key Types</h2>
 * <dl>
 *   <dt>{@link io.github.jbellis.jvector.vector.types.ByteSequence}</dt>
 *   <dd>A generic interface for accessing and manipulating byte sequences with various
 *       backing storage types. Supports random access, bulk operations, slicing, and
 *       value-based equality comparison. Used extensively for low-level vector data
 *       storage and manipulation.</dd>
 *
 *   <dt>{@link io.github.jbellis.jvector.vector.types.VectorFloat}</dt>
 *   <dd>Provides abstraction over float vector representations, allowing implementations
 *       to choose between different storage strategies optimized for their use case.</dd>
 *
 *   <dt>{@link io.github.jbellis.jvector.vector.types.VectorTypeSupport}</dt>
 *   <dd>Factory and utility class for creating and managing vector type implementations.
 *       Serves as the primary entry point for obtaining vector and byte sequence instances.</dd>
 * </dl>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Creating a byte sequence
 * ByteSequence<byte[]> sequence = VectorTypeSupport.createByteSequence(1024);
 *
 * // Setting values
 * sequence.set(0, (byte) 42);
 * sequence.setLittleEndianShort(1, (short) 1000);
 *
 * // Creating a slice view
 * ByteSequence<byte[]> slice = sequence.slice(100, 200);
 *
 * // Copying data
 * ByteSequence<byte[]> copy = sequence.copy();
 * copy.copyFrom(sequence, 0, 512, 512);
 * }</pre>
 *
 * <h2>Design Principles</h2>
 * <ul>
 *   <li><strong>Abstraction:</strong> Interfaces abstract over concrete storage to allow
 *       flexibility in implementation choice without affecting client code.</li>
 *   <li><strong>Zero-copy operations:</strong> Methods like {@link io.github.jbellis.jvector.vector.types.ByteSequence#slice(int, int)}
 *       enable efficient sub-sequence access without data duplication.</li>
 *   <li><strong>Performance-first:</strong> All APIs are designed with performance-critical
 *       use cases in mind, minimizing overhead and enabling JIT optimizations.</li>
 *   <li><strong>Type safety:</strong> Generic type parameters ensure compile-time type safety
 *       while maintaining flexibility.</li>
 * </ul>
 *
 * @since 1.0
 */
package io.github.jbellis.jvector.vector.types;

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
 * Provides custom exception types used throughout JVector.
 * <p>
 * This package contains specialized exception classes that represent error conditions
 * specific to JVector operations. These exceptions extend standard Java exception types
 * to provide more specific error handling capabilities.
 *
 * <h2>Exception Types</h2>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.exceptions.ThreadInterruptedException} - An unchecked
 *       exception that wraps {@link InterruptedException}. This is used in contexts where
 *       methods cannot declare checked exceptions but need to propagate thread interruption
 *       signals. The wrapped {@code InterruptedException} is preserved as the cause.</li>
 * </ul>
 *
 * <h2>Usage Guidelines</h2>
 * <p>
 * {@code ThreadInterruptedException} is typically thrown by JVector when:
 * <ul>
 *   <li>A thread is interrupted during graph construction or search operations</li>
 *   <li>The operation is running in a context that does not allow checked exceptions
 *       (such as lambda expressions or stream operations)</li>
 * </ul>
 *
 * <h2>Exception Handling Example</h2>
 * <pre>{@code
 * try {
 *     GraphIndexBuilder builder = new GraphIndexBuilder(...);
 *     builder.build(vectors);
 * } catch (ThreadInterruptedException e) {
 *     // Thread was interrupted during graph construction
 *     Thread.currentThread().interrupt(); // Restore interrupt status
 *     logger.warn("Graph construction was interrupted", e);
 * }
 * }</pre>
 *
 * <p>
 * When catching {@code ThreadInterruptedException}, it is generally recommended to restore
 * the thread's interrupt status by calling {@code Thread.currentThread().interrupt()} unless
 * you are certain the interruption has been properly handled.
 *
 * @see io.github.jbellis.jvector.exceptions.ThreadInterruptedException
 * @see java.lang.InterruptedException
 */
package io.github.jbellis.jvector.exceptions;

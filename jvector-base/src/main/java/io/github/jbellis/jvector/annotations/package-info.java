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
 * Provides annotation types for documenting API stability and visibility constraints.
 * <p>
 * This package contains marker annotations used throughout JVector to communicate
 * API stability guarantees and visibility intentions to library users.
 *
 * <h2>Available Annotations</h2>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.annotations.Experimental} - Marks APIs that are
 *       experimental and may change or be removed in future releases without prior notice.
 *       Users should avoid depending on experimental APIs in production code.</li>
 *   <li>{@link io.github.jbellis.jvector.annotations.VisibleForTesting} - Marks classes,
 *       methods, or fields that are made visible (typically package-private or public)
 *       solely for testing purposes. These elements are internal implementation details
 *       and may change without warning despite their visibility level.</li>
 * </ul>
 *
 * <h2>Usage Guidelines</h2>
 * <p>
 * When using JVector as a library:
 * <ul>
 *   <li>Avoid using APIs marked with {@code @Experimental} in production code, as they
 *       may be modified or removed in any release.</li>
 *   <li>Do not rely on APIs marked with {@code @VisibleForTesting}, even if they are
 *       technically accessible. These are implementation details that may change without
 *       following semantic versioning rules.</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * @Experimental
 * public class NewFeature {
 *     // This feature is experimental and may change
 * }
 *
 * public class GraphIndexBuilder {
 *     @VisibleForTesting
 *     public void setEntryPoint(int level, int node) {
 *         // Made public for testing but intended as internal API
 *     }
 * }
 * }</pre>
 *
 * @see io.github.jbellis.jvector.annotations.Experimental
 * @see io.github.jbellis.jvector.annotations.VisibleForTesting
 */
package io.github.jbellis.jvector.annotations;

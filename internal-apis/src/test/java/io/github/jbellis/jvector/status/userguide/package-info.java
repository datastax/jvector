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
 * Complete, runnable examples from the Status API User Guide.
 *
 * <p>This package contains working examples that demonstrate status tracking
 * from the simplest possible usage to advanced enterprise patterns. Each example
 * can be run directly and builds incrementally on the previous level.
 *
 * <h2>Progressive Examples:</h2>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.Level1_AbsoluteMinimum} -
 *       Track a single task with one line of code</li>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.Level2_AddConsoleOutput} -
 *       Add console output to see progress</li>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.Level3_MultipleTasks} -
 *       Share a StatusContext across multiple tasks</li>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.Level4_OrganizeWithScopes} -
 *       Group related tasks using explicit scopes</li>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.Level5_CustomSinks} -
 *       Add metrics collection with multiple sinks</li>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.Level6_ConfigurePollInterval} -
 *       Adjust polling frequency</li>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.Level7_ParallelExecution} -
 *       Track parallel execution with AtomicLong</li>
 * </ul>
 *
 * <h2>Supporting Classes:</h2>
 * <ul>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.fauxtasks.DataLoader} -
 *       Example task with volatile long progress tracking</li>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.fauxtasks.DataValidator} -
 *       Example validation task</li>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.fauxtasks.DataProcessor} -
 *       Example processing task</li>
 *   <li>{@link io.github.jbellis.jvector.status.userguide.fauxtasks.ParallelDataLoader} -
 *       Example task using AtomicLong for parallel execution</li>
 * </ul>
 *
 * @see io.github.jbellis.jvector.status.StatusTracker
 * @see io.github.jbellis.jvector.status.StatusContext
 * @see io.github.jbellis.jvector.status.StatusScope
 * @since 4.0.0
 */
package io.github.jbellis.jvector.status.userguide;

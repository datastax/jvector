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
 * Provides implementations of {@link io.github.jbellis.jvector.status.eventing.StatusSink}
 * for various output formats and destinations. Sinks receive status events from the monitoring
 * framework and process them according to their specific purpose (console display, logging,
 * metrics collection, etc.).
 *
 * <h2>Available Sink Implementations</h2>
 *
 * <h3>Display Sinks</h3>
 * <ul>
 *   <li>{@link ConsoleLoggerSink} - Simple console output
 *       with progress bars and timestamps for standard terminal usage</li>
 *   <li>{@link ConsolePanelSink} - Advanced interactive
 *       terminal UI using JLine3 with hierarchical display, scrollable logs, and keyboard controls</li>
 * </ul>
 *
 * <h3>Integration Sinks</h3>
 * <ul>
 *   <li>{@link LoggerStatusSink} - Integration with Log4j 2
 *       for routing status updates through the logging framework</li>
 *   <li>{@link MetricsStatusSink} - Collects performance
 *       metrics and statistics for monitoring and analysis</li>
 * </ul>
 *
 * <h3>Utility Sinks</h3>
 * <ul>
 *   <li>{@link NoopStatusSink} - No-operation sink for
 *       disabling output or testing</li>
 * </ul>
 *
 * <h3>Supporting Classes</h3>
 * <ul>
 *   <li>{@link LogBuffer} - Log4j 2 appender that bridges
 *       logging output to ConsolePanelSink</li>
 *   <li>{@link OutputMode} - Enum for configuring output
 *       mode based on terminal capabilities</li>
 * </ul>
 *
 * <h2>Usage Patterns</h2>
 *
 * <h3>Single Sink</h3>
 * <pre>{@code
 * StatusContext context = new StatusContext("my-operation");
 * context.addSink(new ConsoleLoggerSink());
 * }</pre>
 *
 * <h3>Multiple Sinks</h3>
 * <pre>{@code
 * StatusContext context = new StatusContext("my-operation");
 * context.addSink(new ConsoleLoggerSink());
 * context.addSink(new LoggerStatusSink("app.tasks"));
 * context.addSink(new MetricsStatusSink());
 * }</pre>
 *
 * <h3>Conditional Sink Selection</h3>
 * <pre>{@code
 * OutputMode mode = OutputMode.detect();
 * StatusSink sink = mode == OutputMode.INTERACTIVE
 *     ? ConsolePanelSink.builder().build()
 *     : new ConsoleLoggerSink();
 * context.addSink(sink);
 * }</pre>
 *
 * @see io.github.jbellis.jvector.status.eventing.StatusSink
 * @see io.github.jbellis.jvector.status.StatusContext
 * @since 4.0.0
 */
package io.github.jbellis.jvector.status.sinks;

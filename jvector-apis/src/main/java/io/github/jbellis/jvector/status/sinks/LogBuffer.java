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

package io.github.jbellis.jvector.status.sinks;

import io.github.jbellis.jvector.status.LoggerConfig;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.Layout;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.plugins.Plugin;
import org.apache.logging.log4j.core.config.plugins.PluginAttribute;
import org.apache.logging.log4j.core.config.plugins.PluginElement;
import org.apache.logging.log4j.core.config.plugins.PluginFactory;
import org.apache.logging.log4j.core.layout.PatternLayout;

import java.io.Serializable;
import java.util.Objects;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Log4j 2 appender that captures and forwards log events to the active {@link ConsolePanelSink}.
 * This appender integrates the logging framework with the interactive console display, allowing
 * log messages from any logger to appear in the console panel's scrollable log section.
 *
 * <p>Key Responsibilities:
 * <ul>
 *   <li><strong>Event Capture:</strong> Intercepts all log events routed through Log4j 2</li>
 *   <li><strong>Message Formatting:</strong> Formats log events with level, logger name, and message</li>
 *   <li><strong>Buffering:</strong> Queues messages when no active sink is available (up to 1000 messages)</li>
 *   <li><strong>Forwarding:</strong> Delivers formatted messages to the active {@link ConsolePanelSink}</li>
 * </ul>
 *
 * <h2>Integration Pattern</h2>
 * <p>This appender is installed by {@link LoggerConfig} when
 * configuring for interactive mode:</p>
 * <pre>{@code
 * // Automatic installation
 * LoggerConfig.configure(OutputMode.INTERACTIVE);
 *
 * // Create ConsolePanelSink - it registers itself as the active sink
 * ConsolePanelSink sink = ConsolePanelSink.builder().build();
 *
 * // All logging now flows to the console panel
 * Logger logger = LogManager.getLogger(MyClass.class);
 * logger.info("This appears in the console panel");
 * }</pre>
 *
 * <h2>Lifecycle Management</h2>
 * <p>The appender coordinates with {@link ConsolePanelSink} through static methods:</p>
 * <ul>
 *   <li>{@link #setActiveSink(ConsolePanelSink)} - Called when sink is created; flushes buffer</li>
 *   <li>{@link #clearActiveSink()} - Called when sink is closed; resumes buffering</li>
 * </ul>
 *
 * <h2>Message Buffering</h2>
 * <p>When no sink is active, messages are buffered in memory (max 1000 entries). When a sink
 * becomes active, the buffer is flushed to the sink. This ensures log messages generated during
 * application startup are not lost.</p>
 *
 * <h2>Message Format</h2>
 * <p>Log events are formatted as:</p>
 * <pre>[LEVEL] LoggerName - Message</pre>
 * <p>For example:</p>
 * <pre>[INFO ] DemoTask - Starting task: DataLoad</pre>
 *
 * <h2>Thread Safety</h2>
 * <p>This appender is thread-safe through the use of {@link java.util.concurrent.ConcurrentLinkedQueue}
 * for buffering and volatile references for the active sink.</p>
 *
 * <h2>Performance Considerations</h2>
 * <p>The appender uses a lock-free queue for high-throughput logging scenarios. However, when
 * forwarding to {@link ConsolePanelSink}, the sink's thread-safe methods are called, which may
 * introduce synchronization overhead.</p>
 *
 * @see ConsolePanelSink
 * @see LoggerConfig
 * @see OutputMode
 * @since 4.0.0
 */
@Plugin(name = "LogBuffer", category = "Core", elementType = "appender", printObject = true)
public class LogBuffer extends AbstractAppender {

    /**
     * Represents a captured log entry with its level and formatted message.
     */
    public static class LogEntry {
        public final Level level;
        public final String formattedMessage;

        public LogEntry(Level level, String formattedMessage) {
            this.level = level;
            this.formattedMessage = formattedMessage;
        }
    }

    private static volatile ConsolePanelSink activeSink;
    private static final Queue<LogEntry> allLogEntries = new ConcurrentLinkedQueue<>();
    private static final Queue<String> bufferedMessages = new ConcurrentLinkedQueue<>();
    private static final int MAX_BUFFER_SIZE = 10000;
    private static volatile Level displayLevel = Level.INFO;
    private static final long START_TIME = System.currentTimeMillis();

    protected LogBuffer(String name, Layout<? extends Serializable> layout) {
        super(name,
                null,
                Objects.requireNonNullElse(layout, PatternLayout.newBuilder().withPattern("%msg").build()),
                false,
                null);
    }

    /**
     * Log4j 2 plugin factory method for creating LogBuffer appenders from configuration.
     * This method is called by Log4j 2 when the appender is configured in log4j2.xml or
     * programmatically via the Configuration API.
     *
     * @param name the name of the appender instance
     * @param layout the layout for formatting log events (optional, defaults to simple pattern)
     * @return a started LogBuffer appender instance
     */
    @PluginFactory
    public static LogBuffer createAppender(@PluginAttribute("name") String name,
                                           @PluginElement("Layout") Layout<? extends Serializable> layout) {
        LogBuffer appender = new LogBuffer(Objects.requireNonNullElse(name, "LogBuffer"), layout);
        appender.start();
        return appender;
    }

    /**
     * Convenience factory method for creating a LogBuffer with default layout.
     *
     * @param name the name of the appender instance
     * @return a started LogBuffer appender instance with default pattern layout
     */
    public static LogBuffer createAppender(String name) {
        return createAppender(name, null);
    }

    /**
     * Registers a {@link ConsolePanelSink} as the active sink for receiving log messages.
     * When a sink is set, all buffered messages are immediately flushed to the sink.
     * This method is called automatically by {@link ConsolePanelSink} during initialization.
     *
     * @param sink the console panel sink to receive log messages, or null to clear
     */
    public static void setActiveSink(ConsolePanelSink sink) {
        activeSink = sink;
        if (sink != null) {
            String msg;
            while ((msg = bufferedMessages.poll()) != null) {
                sink.addLogMessage(msg);
            }
        }
    }

    /**
     * Clears the active sink reference, causing subsequent log messages to be buffered
     * instead of forwarded. This method is called automatically by {@link ConsolePanelSink}
     * during cleanup.
     */
    public static void clearActiveSink() {
        activeSink = null;
    }

    /**
     * Sets the display level filter. Only log entries at or above this level will be
     * visible to the sink.
     *
     * @param level the minimum level to display
     */
    public static void setDisplayLevel(Level level) {
        displayLevel = level;
        if (activeSink != null) {
            String stats = activeSink.refreshDisplayBuffer();
            // Log the stats as a system message (will go through normal logging)
            activeSink.addLogMessage("[SYSTEM] " + stats);
        }
    }

    /**
     * Gets the current display level filter.
     *
     * @return the current display level
     */
    public static Level getDisplayLevel() {
        return displayLevel;
    }

    /**
     * Gets all log entries (unfiltered) for rebuilding the display buffer.
     *
     * @return all captured log entries
     */
    public static Queue<LogEntry> getAllLogEntries() {
        return allLogEntries;
    }

    /**
     * Appends a log event to the buffer or forwards it to the active sink.
     * This method is called by Log4j 2 for each log event that passes through this appender.
     * <p>
     * The event is formatted with level, logger name (simple name only), and message.
     * If an exception is present, its message is appended on a new line.
     * <p>
     * All events are stored in the complete buffer. Events are only forwarded to the
     * active sink if they meet or exceed the current display level.
     *
     * @param event the log event to append
     */
    @Override
    public void append(LogEvent event) {
        String loggerName = event.getLoggerName();
        if (loggerName != null) {
            int lastDot = loggerName.lastIndexOf('.');
            if (lastDot >= 0 && lastDot < loggerName.length() - 1) {
                loggerName = loggerName.substring(lastDot + 1);
            }
        }

        // Format timestamp as relative time from start (e.g., "+00:02.345" for 2.345 seconds)
        long timestamp = event.getTimeMillis();
        long elapsedMs = timestamp - START_TIME;
        long seconds = elapsedMs / 1000;
        long millis = elapsedMs % 1000;
        long minutes = seconds / 60;
        seconds = seconds % 60;
        String timeStr = String.format("+%02d:%02d.%03d", minutes, seconds, millis);

        String formattedMessage = String.format("[%s] [%-5s] %s - %s",
                timeStr,
                event.getLevel(),
                Objects.requireNonNullElse(loggerName, "root"),
                event.getMessage().getFormattedMessage());

        if (event.getThrown() != null) {
            formattedMessage += "\n" + event.getThrown().getMessage();
        }

        // Store in complete buffer (up to max size)
        Level eventLevel = event.getLevel();
        if (allLogEntries.size() < MAX_BUFFER_SIZE) {
            allLogEntries.offer(new LogEntry(eventLevel, formattedMessage));
        } else {
            // Remove oldest entry to make room
            allLogEntries.poll();
            allLogEntries.offer(new LogEntry(eventLevel, formattedMessage));
        }

        ConsolePanelSink sink = activeSink;
        if (sink != null) {
            // Debug: log every 100th entry to show backing buffer is being populated
            if (allLogEntries.size() % 100 == 0) {
                sink.addLogMessage("[DEBUG] LogBuffer backing buffer now has " + allLogEntries.size() + " entries");
            }

            // Only forward if event level is at or above display level
            // Note: Lower intLevel = more severe (ERROR=200, WARN=300, INFO=400, DEBUG=500, TRACE=600)
            // So to show INFO and above, we want intLevel <= INFO.intLevel()
            if (eventLevel.intLevel() <= displayLevel.intLevel()) {
                sink.addLogMessage(formattedMessage);
            }
        } else if (bufferedMessages.size() < MAX_BUFFER_SIZE) {
            bufferedMessages.offer(formattedMessage);
        }
    }
}

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

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.Queue;

/**
 * Custom Logback appender that buffers log messages for the ConsolePanelSink.
 * This ensures all logging framework output is captured and displayed
 * in the console panel instead of interfering with the display.
 */
public class LogBuffer extends AppenderBase<ILoggingEvent> {

    private static volatile ConsolePanelSink activeSink;
    private static final Queue<String> bufferedMessages = new ConcurrentLinkedQueue<>();
    private static final int MAX_BUFFER_SIZE = 1000; // Limit buffer size to prevent memory issues

    /**
     * Register the active ConsolePanelSink to receive log messages
     */
    public static void setActiveSink(ConsolePanelSink sink) {
        activeSink = sink;
        // Flush buffered messages to the new sink
        if (sink != null) {
            String msg;
            while ((msg = bufferedMessages.poll()) != null) {
                sink.addLogMessage(msg);
            }
        }
    }

    /**
     * Clear the active sink when it's closed
     */
    public static void clearActiveSink() {
        activeSink = null;
    }

    @Override
    protected void append(ILoggingEvent event) {
        // Format the log message
        String level = event.getLevel().toString();
        String loggerName = event.getLoggerName();
        String message = event.getFormattedMessage();

        // Simplify logger name (take last component)
        int lastDot = loggerName.lastIndexOf('.');
        if (lastDot >= 0 && lastDot < loggerName.length() - 1) {
            loggerName = loggerName.substring(lastDot + 1);
        }

        String formattedMessage = String.format("[%-5s] %s - %s", level, loggerName, message);

        // Add any exception stack trace
        if (event.getThrowableProxy() != null) {
            formattedMessage += "\n" + event.getThrowableProxy().getMessage();
        }

        ConsolePanelSink sink = activeSink;
        if (sink != null) {
            // Send directly to the sink
            sink.addLogMessage(formattedMessage);
        } else {
            // Buffer the message for later delivery
            if (bufferedMessages.size() < MAX_BUFFER_SIZE) {
                bufferedMessages.offer(formattedMessage);
            }
            // If buffer is full, drop oldest messages (this is a safeguard)
        }
    }
}
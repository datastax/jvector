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

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.layout.PatternLayout;
import org.slf4j.bridge.SLF4JBridgeHandler;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Centralized logging configuration utility for applications using the status tracking framework.
 * This class configures Log4j 2 to integrate with {@link ConsolePanelSink}
 * by installing a {@link LogBuffer} appender that captures
 * log output for display in the interactive console UI.
 *
 * <p>When configured for interactive mode, this class:
 * <ul>
 *   <li>Removes existing console appenders to prevent duplicate output</li>
 *   <li>Installs a LogBuffer appender that captures all log messages</li>
 *   <li>Configures log formatting with timestamp, level, logger name, and message</li>
 *   <li>Bridges java.util.logging to SLF4J for unified log capture</li>
 * </ul>
 *
 * <p>Usage with ConsolePanel:
 * <pre>{@code
 * // Configure logging before creating ConsolePanelSink
 * LoggerConfig.configure(OutputMode.INTERACTIVE);
 *
 * // Create sink with console panel
 * ConsolePanelSink sink = ConsolePanelSink.builder().build();
 *
 * // Now all logging will appear in the console panel
 * Logger logger = LogManager.getLogger(MyClass.class);
 * logger.info("This will appear in the console panel");
 * }</pre>
 *
 * <p>This class is designed for application startup configuration and should be
 * called once before creating any {@link ConsolePanelSink}
 * instances.
 *
 * @see ConsolePanelSink
 * @see LogBuffer
 * @see OutputMode
 * @since 4.0.0
 */
public final class ConsolePanelLogIntercept {

    private static final AtomicBoolean CONFIGURING = new AtomicBoolean(false);
    private static final String APPENDER_NAME = "ConsolePanelLogBuffer";

    private ConsolePanelLogIntercept() {
    }

    /**
     * Configures logging for the specified output mode. For {@link OutputMode#INTERACTIVE}
     * mode, installs a Log4j 2 appender that captures and forwards log output to the console panel.
     * Other output modes have no effect.
     * <p>
     * This method is idempotent and thread-safe. Multiple calls will only configure logging once.
     *
     * @param outputMode the desired output mode (only INTERACTIVE triggers configuration)
     */
    public static void configure(OutputMode outputMode) {
        if (outputMode == OutputMode.INTERACTIVE) {
            configureForConsolePanel();
        }
    }

    /**
     * Convenience method for configuration from static initializers. Detects the appropriate
     * output mode based on environment and configures logging accordingly.
     * <p>
     * This method uses {@link OutputMode#detect()}
     * to determine the best mode based on terminal capabilities.
     */
    public static void configureForStaticInit() {
        configure(OutputMode.detect());
    }

    private static void configureForConsolePanel() {
        if (!CONFIGURING.compareAndSet(false, true)) {
            return;
        }

        try {
            LoggerContext context = (LoggerContext) LogManager.getContext(false);
            Configuration configuration = context.getConfiguration();

            org.apache.logging.log4j.core.config.LoggerConfig rootConfig = configuration.getRootLogger();

            PatternLayout layout = PatternLayout.newBuilder()
                    .withPattern("[%d{HH:mm:ss}] %-5level %logger{1} - %msg")
                    .withConfiguration(configuration)
                    .build();

            LogBuffer appender = LogBuffer.createAppender(APPENDER_NAME, layout);
            configuration.addAppender(appender);

            // Remove existing appenders before registering the console panel buffer.
            List<String> existingAppenders = new ArrayList<>(rootConfig.getAppenders().keySet());
            for (String appenderName : existingAppenders) {
                rootConfig.removeAppender(appenderName);
            }

            // Set appender level to ALL so LogBuffer receives all events
            // LogBuffer will apply its own display-level filtering
            rootConfig.addAppender(appender, Level.ALL, null);
            rootConfig.setLevel(Level.ALL);

            // Ensure child logger configurations also inherit from the root and do not keep stale appenders.
            for (org.apache.logging.log4j.core.config.LoggerConfig loggerConfig : configuration.getLoggers().values()) {
                if (loggerConfig != rootConfig) {
                    for (String appenderName : new ArrayList<>(loggerConfig.getAppenders().keySet())) {
                        loggerConfig.removeAppender(appenderName);
                    }
                    loggerConfig.setLevel(null);
                    loggerConfig.setAdditive(true);
                }
            }

            context.updateLoggers();

            installJulBridge();
        } finally {
            CONFIGURING.set(false);
        }
    }

    private static void installJulBridge() {
        try {
            java.util.logging.LogManager.getLogManager().reset();
            SLF4JBridgeHandler.removeHandlersForRootLogger();
            SLF4JBridgeHandler.install();
        } catch (Exception ignored) {
            // Bridge installation is best effort.
        }
    }
}

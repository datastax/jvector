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

package io.github.jbellis.jvector.example.util;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.spi.LoggerContextListener;
import ch.qos.logback.classic.turbo.TurboFilter;
import ch.qos.logback.core.spi.FilterReply;
import io.github.jbellis.jvector.status.sinks.LogBuffer;
import io.github.jbellis.jvector.status.sinks.OutputMode;
import org.slf4j.LoggerFactory;
import org.slf4j.Marker;

/**
 * Centralized logging configuration for JVector examples.
 * This class handles the setup of logging for different output modes,
 * particularly for JLine interactive mode which requires special handling.
 */
public class LoggerConfig {

    private static final ThreadLocal<Boolean> configuring = ThreadLocal.withInitial(() -> false);

    /**
     * Configure logging based on the output mode.
     * This should be called as early as possible in the application lifecycle.
     * The output mode should already be resolved (not AUTO).
     *
     * @param outputMode The resolved output mode to configure for
     */
    public static void configure(OutputMode outputMode) {
        if (outputMode == OutputMode.INTERACTIVE) {
            // Prevent Logback from auto-configuring
            System.setProperty("logback.configurationFile", "NONE");

            // Prevent Log4j2 from auto-configuring and force it to use SLF4J
            System.setProperty("log4j2.configurationFile", "NONE");
            System.setProperty("log4j.configurationFile", "NONE");

            // Force Log4j2 to use the SLF4J bridge
            System.setProperty("log4j2.disable.jmx", "true");
            System.setProperty("org.apache.logging.log4j.simplelog.StatusLogger.level", "OFF");

            configureForJLine();
        }
        // For other modes (BASIC, ENHANCED), use default logback.xml configuration
    }

    /**
     * Configure logging for static initialization.
     * This method auto-detects the output mode and configures accordingly.
     * Used in static initializers where we need early logging configuration.
     */
    public static void configureForStaticInit() {
        // Auto-detect and configure
        configure(OutputMode.detect());
    }

    private static void configureForJLine() {
        // Prevent infinite recursion
        if (configuring.get()) {
            return;
        }
        configuring.set(true);

        try {
            // Install JUL to SLF4J bridge first
            // This must be done before any JUL loggers are created
            try {
                // Remove existing JUL handlers
                java.util.logging.LogManager.getLogManager().reset();
                // Install SLF4J bridge handler
                org.slf4j.bridge.SLF4JBridgeHandler.removeHandlersForRootLogger();
                org.slf4j.bridge.SLF4JBridgeHandler.install();
            } catch (Exception e) {
                // If jul-to-slf4j is not available, continue without it
                System.err.println("[LoggerConfig] Warning: Could not install JUL to SLF4J bridge: " + e.getMessage());
            }

        // Get the logger context and reset it completely
        LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();

        // First, stop the context to prevent any logging during reconfiguration
        loggerContext.stop();

        // Clear all turbo filters that might exist
        loggerContext.getTurboFilterList().clear();

        // Now reset it - this removes ALL appenders and configurations
        loggerContext.reset();

        // Ensure AWS SDK logging is configured to use SLF4J
        // AWS SDK v2 uses SLF4J by default, but we make sure it's at the right level
        System.setProperty("aws.java.v2.log.level", "warn");

        // Create and configure the LogBuffer
        LogBuffer logBufferAppender = new LogBuffer();
        logBufferAppender.setContext(loggerContext);
        logBufferAppender.setName("LOGBUFFER");
        logBufferAppender.start();

        // Configure root logger to capture EVERYTHING, including future loggers
        Logger rootLogger = loggerContext.getLogger(Logger.ROOT_LOGGER_NAME);
        rootLogger.setLevel(Level.TRACE); // Set to TRACE to catch everything, filter later
        rootLogger.setAdditive(false); // Don't propagate to parent appenders
        rootLogger.addAppender(logBufferAppender);

        // Force all existing loggers to inherit from root
        // This includes any static loggers that were created before our configuration
        for (Logger logger : loggerContext.getLoggerList()) {
            if (logger != rootLogger) {
                // Remove ALL appenders from this logger
                logger.detachAndStopAllAppenders();
                // Force inheritance from root
                logger.setLevel(null); // Inherit level from parent
                logger.setAdditive(true); // Use parent's appenders
            }
        }

        // Special handling for AWS SDK loggers that might have been statically initialized
        // Force reconfiguration even if they don't exist yet
        String[] awsLoggerNames = {
            "software.amazon.awssdk.transfer.s3.S3TransferManager",
            "software.amazon.awssdk.transfer.s3",
            "software.amazon.awssdk.transfer",
            "software.amazon.awssdk.core",
            "software.amazon.awssdk.http",
            "software.amazon.awssdk.request",
            "software.amazon.awssdk",
            "software.amazon"
        };

        for (String loggerName : awsLoggerNames) {
            Logger awsLogger = loggerContext.getLogger(loggerName);
            awsLogger.detachAndStopAllAppenders();
            awsLogger.setLevel(null); // Inherit from parent
            awsLogger.setAdditive(true);
        }

        // Set up a Turbo Filter to ensure ALL logging goes through our appender
        // This will apply to all loggers, even those created later
        TurboFilter turboFilter = new TurboFilter() {
            @Override
            public FilterReply decide(
                    Marker marker,
                    Logger logger,
                    Level level,
                    String format,
                    Object[] params,
                    Throwable t) {

                // Force all loggers to use our configuration
                if (logger != rootLogger) {
                    // Remove any appenders that might have been added
                    logger.iteratorForAppenders().forEachRemaining(logger::detachAppender);
                    // Force inheritance from root
                    logger.setAdditive(true);
                    if (logger.getLevel() != null) {
                        logger.setLevel(null); // Clear level to inherit from parent
                    }
                }

                // Single uniform filtering rule for ALL loggers
                // Accept INFO and above, reject everything else
                return level.isGreaterOrEqual(Level.INFO)
                    ? FilterReply.NEUTRAL
                    : FilterReply.DENY;
            }
        };
        turboFilter.start();
        loggerContext.addTurboFilter(turboFilter);

        // Add a LoggerContextListener to configure any loggers created in the future
        loggerContext.addListener(new LoggerContextListener() {
            @Override
            public boolean isResetResistant() {
                return true; // Survive context resets
            }

            @Override
            public void onStart(LoggerContext context) {
                // Called when context starts
            }

            @Override
            public void onReset(LoggerContext context) {
                // Don't re-apply configuration during reset to prevent infinite recursion
                // The reset was likely triggered by our own configuration process
            }

            @Override
            public void onStop(LoggerContext context) {
                // Called when context stops
            }

            @Override
            public void onLevelChange(Logger logger, Level level) {
                // Ensure new loggers use root's appender
                if (!logger.iteratorForAppenders().hasNext() && logger != rootLogger) {
                    logger.setAdditive(true);
                }
            }
        });

            // Start the context
            loggerContext.start();
        } finally {
            configuring.set(false);
        }
    }
}
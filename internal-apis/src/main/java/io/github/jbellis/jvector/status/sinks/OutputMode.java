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

/**
 * Defines output modes for status display and logging configuration. The mode determines
 * how status information is rendered to the console and which terminal features are used.
 *
 * <p>Mode Selection:
 * <ul>
 *   <li><strong>INTERACTIVE:</strong> Use for applications running in real terminals with full
 *       JLine support. Provides hierarchical display, keyboard controls, and log panel.</li>
 *   <li><strong>ENHANCED:</strong> Use for terminals with ANSI color support but limited control
 *       (e.g., some CI/CD environments, piped output with TERM set).</li>
 *   <li><strong>BASIC:</strong> Use for dumb terminals or when output is captured/piped
 *       (e.g., logs, basic shells).</li>
 *   <li><strong>AUTO:</strong> Let the framework detect the best mode based on environment.</li>
 * </ul>
 *
 * <p>Environment Detection:
 * The {@link #detect()} method determines the appropriate mode using:
 * <ul>
 *   <li>{@code TERM} environment variable (null or "dumb" → BASIC)</li>
 *   <li>{@code System.console()} availability (null → ENHANCED or BASIC)</li>
 *   <li>Full terminal capabilities (present → INTERACTIVE)</li>
 * </ul>
 *
 * @see LoggerConfig
 * @see ConsolePanelSink
 * @since 4.0.0
 */
public enum OutputMode {
    /**
     * Interactive mode with full JLine terminal control, hierarchical display, and keyboard input.
     * Best for development and interactive use in real terminals.
     */
    INTERACTIVE("interactive", "Full terminal control with hierarchical display and keyboard interaction"),

    /**
     * Enhanced mode with colors and ANSI formatting but no terminal control.
     * Suitable for CI/CD environments and piped output where colors are supported.
     */
    ENHANCED("enhanced", "Color-enabled output with ANSI formatting"),

    /**
     * Basic mode with plain text output, no colors or special formatting.
     * Use for dumb terminals, log files, and environments without ANSI support.
     */
    BASIC("basic", "Plain text output without colors or special formatting"),

    /**
     * Auto-detect the best mode based on environment variables and terminal capabilities.
     * Uses {@link #detect()} to choose INTERACTIVE, ENHANCED, or BASIC.
     */
    AUTO("auto", "Automatically detect the best output mode");

    private final String name;
    private final String description;

    OutputMode(String name, String description) {
        this.name = name;
        this.description = description;
    }

    /**
     * Returns the lowercase string name of this mode.
     *
     * @return the mode name (e.g., "interactive", "enhanced", "basic", "auto")
     */
    public String getName() {
        return name;
    }

    /**
     * Returns a human-readable description of this mode.
     *
     * @return the mode description
     */
    public String getDescription() {
        return description;
    }

    /**
     * Parses a string value to an OutputMode enum value. Accepts both lowercase names
     * (e.g., "interactive") and uppercase enum names (e.g., "INTERACTIVE").
     *
     * @param value the string to parse, case-insensitive
     * @return the corresponding OutputMode, or AUTO if the value is null or unrecognized
     */
    public static OutputMode fromString(String value) {
        if (value == null) {
            return AUTO;
        }

        String lower = value.toLowerCase().trim();
        for (OutputMode mode : values()) {
            if (mode.name.equals(lower)) {
                return mode;
            }
        }

        // Try to match by enum name as well
        try {
            return OutputMode.valueOf(value.toUpperCase());
        } catch (IllegalArgumentException e) {
            System.err.println("Unknown output mode: " + value + ". Using AUTO.");
            return AUTO;
        }
    }

    /**
     * Automatically detects the best output mode based on the runtime environment.
     * Uses terminal capabilities and environment variables to determine which mode
     * will work best.
     * <p>
     * Detection Logic:
     * <ol>
     *   <li>If {@code TERM} is null or "dumb" → returns BASIC</li>
     *   <li>If {@code System.console()} is null (piped/redirected) → returns ENHANCED</li>
     *   <li>If real terminal detected → returns INTERACTIVE</li>
     * </ol>
     *
     * @return the detected output mode (never returns AUTO)
     */
    public static OutputMode detect() {
        // Check TERM environment variable
        String term = System.getenv("TERM");

        // If TERM is not set or is "dumb", use basic mode
        if (term == null || term.equals("dumb")) {
            return BASIC;
        }

        // Check if output is being piped (System.console() returns null when piped)
        if (System.console() == null) {
            // Output is piped, but TERM is set - use enhanced mode for colors
            return ENHANCED;
        }

        // We have a real terminal - use interactive mode
        return INTERACTIVE;
    }
}
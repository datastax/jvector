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

/**
 * Output modes for status display
 */
public enum OutputMode {
    /**
     * Interactive mode with full JLine terminal control, hierarchical display, and keyboard input
     */
    INTERACTIVE("interactive", "Full terminal control with hierarchical display and keyboard interaction"),

    /**
     * Enhanced mode with colors and ANSI formatting but no terminal control
     */
    ENHANCED("enhanced", "Color-enabled output with ANSI formatting"),

    /**
     * Basic mode with plain text output, no colors or special formatting
     */
    BASIC("basic", "Plain text output without colors or special formatting"),

    /**
     * Auto-detect the best mode based on environment
     */
    AUTO("auto", "Automatically detect the best output mode");

    private final String name;
    private final String description;

    OutputMode(String name, String description) {
        this.name = name;
        this.description = description;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

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
     * Detect the best output mode based on environment
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
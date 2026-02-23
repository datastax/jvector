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

package io.github.jbellis.jvector.example.reporting;

import org.apache.commons.io.FilenameUtils;
import java.time.Clock;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;

/**
 * Utility for generating a per-run directory name ("runId") for logging artifacts.
 *
 * <p>The runId is rendered from a user-supplied template containing the placeholder {@code "{ts}"},
 * which is replaced with a UTC timestamp formatted as {@code yyyyMMdd-HHmmssZ} (e.g. {@code 20260213-232254Z}).
 * If the template is blank or missing, {@code "{ts}"} is used. If the template does not include {@code "{ts}"},
 * {@code _{ts}} is appended to ensure uniqueness.</p>
 *
 * <p>The rendered value is then sanitized to be a single, cross-platform-safe path component:
 * path traversal is removed, only {@code [A-Za-z0-9._-]} is permitted (others become {@code _}),
 * trailing dots/spaces are stripped (Windows), and Windows reserved device names are prefixed.</p>
 */
public final class RunIdUtil {
    private static final DateTimeFormatter TS_UTC =
            DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss'Z'").withZone(ZoneOffset.UTC);

    public static String renderRunId(String template, Clock clock) {
        String t = (template == null || template.isBlank()) ? "{ts}" : template.trim();
        if (!t.contains("{ts}")) t = t + "_" + "{ts}";

        String ts = TS_UTC.format(clock.instant());
        String rendered = t.replace("{ts}", ts);

        // Prevent directory climbing
        String nameOnly = FilenameUtils.getName(rendered.replace('\\', '/'));

        // Allow only alphanumeric, dots, hyphens, underscores
        // This covers Linux, Windows, and Mac
        String sanitized = nameOnly.replaceAll("[^A-Za-z0-9._-]", "_")
                .replaceAll("_{2,}", "_")
                .replaceAll("[. ]+$", ""); // Windows: no trailing dot/space

        if (sanitized.isEmpty()) sanitized = "run_" + ts;

        // Windows reserved device names (case-insensitive)
        String base = sanitized;
        int dot = base.indexOf('.');
        if (dot >= 0) base = base.substring(0, dot);
        String upper = base.toUpperCase(java.util.Locale.ROOT);

        if (upper.equals("CON") || upper.equals("PRN") || upper.equals("AUX") || upper.equals("NUL")
                || upper.matches("COM[1-9]") || upper.matches("LPT[1-9]")) {
            sanitized = "run_" + sanitized;
        }

        return sanitized;
    }
}

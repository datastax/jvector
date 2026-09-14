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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Detects the current git commit hash for tagging benchmark results.
 */
public final class GitInfo {
    private static final Logger log = LoggerFactory.getLogger(GitInfo.class);

    private GitInfo() {}

    // Lazy holder pattern — computed once on first access
    private static class Holder {
        static final String SHORT_HASH;
        static {
            String hash;
            try {
                var process = new ProcessBuilder("git", "rev-parse", "HEAD").redirectErrorStream(true).start();
                hash = new String(process.getInputStream().readAllBytes()).trim();
                process.waitFor();
                if (hash.length() >= 8) {
                    hash = hash.substring(hash.length() - 8);
                }
            } catch (Exception e) {
                log.warn("Could not determine git hash", e);
                hash = "unknown";
            }
            SHORT_HASH = hash;
        }
    }

    /** Returns the last 8 characters of {@code git rev-parse HEAD}, or {@code "unknown"} on failure. */
    public static String getShortHash() {
        return Holder.SHORT_HASH;
    }
}

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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Map;

/**
 * Factory for creating version-specific graph index formats.
 * This centralizes version detection and format instantiation.
 */
public class GraphIndexFormatFactory {
    private static final Logger logger = LoggerFactory.getLogger(GraphIndexFormatFactory.class);

    private static final Map<Integer, GraphIndexFormat> FORMATS = Map.of(
        2, new GraphIndexFormatV2(),
        3, new GraphIndexFormatV3(),
        4, new GraphIndexFormatV4(),
        5, new GraphIndexFormatV5(),
        6, new GraphIndexFormatV6()
    );

    /**
     * Gets a format for a specific version.
     * @param version the version number
     * @return the format for that version
     * @throws UnsupportedVersionException if the version is not supported
     */
    public static GraphIndexFormat forVersion(int version) {
        GraphIndexFormat format = FORMATS.get(version);
        if (format == null) {
            throw new UnsupportedVersionException("Version " + version + " is not supported. " +
                "Supported versions: " + FORMATS.keySet());
        }
        return format;
    }

    /**
     * Detects the version from the input stream and returns the appropriate format.
     * The reader position will be reset to where it started.
     *
     * @param in the input reader
     * @return the format for the detected version
     * @throws IOException if an I/O error occurs
     * @throws UnsupportedVersionException if the version is not supported
     */
    public static GraphIndexFormat detectVersion(RandomAccessReader in) throws IOException {
        long startPosition = in.getPosition();

        try {
            int maybeMagic = in.readInt();

            if (maybeMagic == OnDiskGraphIndex.MAGIC) {
                // Version 3+ with magic number
                int version = in.readInt();
                logger.debug("Detected version {} (with magic number)", version);
                return forVersion(version);
            } else {
                // Version 2 (no magic number)
                logger.debug("Detected version 2 (no magic number)");
                return forVersion(2);
            }
        } finally {
            // Reset to starting position
            in.seek(startPosition);
        }
    }

    /**
     * Gets the current/latest version number.
     * @return the current version
     */
    public static int getCurrentVersion() {
        return OnDiskGraphIndex.CURRENT_VERSION;
    }

    /**
     * Exception thrown when an unsupported version is encountered.
     */
    public static class UnsupportedVersionException extends RuntimeException {
        public UnsupportedVersionException(String message) {
            super(message);
        }
    }
}

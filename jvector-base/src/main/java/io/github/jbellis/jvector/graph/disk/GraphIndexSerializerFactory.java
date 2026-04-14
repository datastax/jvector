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
 * Factory for creating version-specific graph index serializers.
 * This centralizes version detection and serializer instantiation.
 */
public class GraphIndexSerializerFactory {
    private static final Logger logger = LoggerFactory.getLogger(GraphIndexSerializerFactory.class);

    private static final Map<Integer, GraphIndexSerializer> SERIALIZERS = Map.of(
        2, new GraphIndexSerializerV2(),
        3, new GraphIndexSerializerV3(),
        4, new GraphIndexSerializerV4(),
        5, new GraphIndexSerializerV5(),
        6, new GraphIndexSerializerV6()
    );

    /**
     * Gets a serializer for a specific version.
     * @param version the version number
     * @return the serializer for that version
     * @throws UnsupportedVersionException if the version is not supported
     */
    public static GraphIndexSerializer forVersion(int version) {
        GraphIndexSerializer serializer = SERIALIZERS.get(version);
        if (serializer == null) {
            throw new UnsupportedVersionException("Version " + version + " is not supported. " +
                "Supported versions: " + SERIALIZERS.keySet());
        }
        return serializer;
    }

    /**
     * Detects the version from the input stream and returns the appropriate serializer.
     * The reader position will be reset to where it started.
     * 
     * @param in the input reader
     * @return the serializer for the detected version
     * @throws IOException if an I/O error occurs
     * @throws UnsupportedVersionException if the version is not supported
     */
    public static GraphIndexSerializer detectVersion(RandomAccessReader in) throws IOException {
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

// Made with Bob

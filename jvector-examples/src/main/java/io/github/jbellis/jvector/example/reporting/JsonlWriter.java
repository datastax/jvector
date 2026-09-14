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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

/**
 * Append-only JSONL file writer that serializes one map per line using GSON.
 */
public final class JsonlWriter {
    private static final Logger log = LoggerFactory.getLogger(JsonlWriter.class);
    private static final Gson GSON = new GsonBuilder()
            .disableHtmlEscaping()
            .serializeNulls()
            .create(); // No pretty printing for JSONL

    private final Path outputFile;

    public JsonlWriter(Path outputFile) {
        this.outputFile = outputFile;
    }

    /** Serializes the map as a single JSON line and appends it to the output file. */
    public void writeLine(Map<String, Object> result) {
        String json = GSON.toJson(result) + "\n";
        try {
            Files.writeString(outputFile, json,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            log.error("Failed to persist result to {}", outputFile, e);
        }
    }
}

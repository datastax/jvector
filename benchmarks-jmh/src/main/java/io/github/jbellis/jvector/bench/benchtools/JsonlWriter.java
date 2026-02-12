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

package io.github.jbellis.jvector.bench.benchtools;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Append-only JSONL file writer that serializes one map per line.
 */
public final class JsonlWriter {
    private static final Logger log = LoggerFactory.getLogger(JsonlWriter.class);

    private final Path outputFile;

    public JsonlWriter(Path outputFile) {
        this.outputFile = outputFile;
    }

    /** Serializes the map as a single JSON line and appends it to the output file. Preserves insertion order. */
    public void writeLine(LinkedHashMap<String, Object> result) {
        StringBuilder json = new StringBuilder();
        appendObject(json, result);
        json.append('\n');


        try {
            Files.writeString(outputFile, json.toString(),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            log.error("Failed to persist result to {}", outputFile, e);
        }
    }

    @SuppressWarnings("unchecked")
    private static void appendValue(StringBuilder sb, Object val) {
        if (val instanceof String) {
            String s = (String) val;
            sb.append('"').append(s.replace("\\", "\\\\").replace("\"", "\\\"")).append('"');
        } else if (val instanceof Map) {
            appendObject(sb, (Map<String, Object>) val);
        } else {
            sb.append(val);
        }
    }

    private static void appendObject(StringBuilder sb, Map<String, Object> map) {
        sb.append('{');
        boolean first = true;
        for (var entry : map.entrySet()) {
            if (!first) sb.append(", ");
            first = false;
            sb.append('"').append(entry.getKey()).append("\": ");
            appendValue(sb, entry.getValue());
        }
        sb.append('}');
    }
}

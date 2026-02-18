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

/**
 * JSON utility for writing run artifacts (e.g., sys_info.json) using the GSON library.
 * <p>
 * Provides structured, pretty-printed output for machine-readable logs and human debugging.
 * Unlike manual serialization, this supports complex POJOs via reflection, while maintaining
 * literal string integrity by disabling HTML escaping.
 */
public final class JsonUtil {
    // Create a single, reusable, thread-safe instance
    private static final Gson GSON = new GsonBuilder()
            .disableHtmlEscaping() // Keeps characters like < and > as-is
            .serializeNulls()      // Force nulls to show up in JSON
            .setPrettyPrinting()
            .create();

    private JsonUtil() {}

    public static String toJson(Object o) {
        // Gson handles Maps, Lists, POJOs, and Primitives automatically
        return GSON.toJson(o) + "\n";
    }
}

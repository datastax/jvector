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
package io.github.jbellis.jvector.bench.compaction;

import java.util.HashMap;
import java.util.Map;

public final class Cli {
    private final Map<String, String> kv = new HashMap<>();

    public Cli(String[] args) {
        for (String a : args) {
            if (!a.startsWith("--")) continue;
            int eq = a.indexOf('=');
            String k = (eq < 0) ? a.substring(2) : a.substring(2, eq);
            String v = (eq < 0) ? "true" : a.substring(eq + 1);
            kv.put(k, v);
        }
    }

    public String get(String key, String def) { return kv.getOrDefault(key, def); }
    public int getInt(String key, int def) { return kv.containsKey(key) ? Integer.parseInt(kv.get(key)) : def; }
    public double getDouble(String key, double def) { return kv.containsKey(key) ? Double.parseDouble(kv.get(key)) : def; }
    public boolean getBool(String key, boolean def) { return kv.containsKey(key) ? Boolean.parseBoolean(kv.get(key)) : def; }
}

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

import java.util.Iterator;
import java.util.List;
import java.util.Map;

public final class JsonUtil {
    private JsonUtil() {}

    public static String toJson(Object o) {
        StringBuilder sb = new StringBuilder(1024);
        writeValue(sb, o);
        sb.append('\n');
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private static void writeValue(StringBuilder sb, Object o) {
        if (o == null) {
            sb.append("null");
        } else if (o instanceof String) {
            sb.append('"').append(escape((String) o)).append('"');
        } else if (o instanceof Number || o instanceof Boolean) {
            sb.append(o.toString());
        } else if (o instanceof Map) {
            writeMap(sb, (Map<String, Object>) o);
        } else if (o instanceof List) {
            writeList(sb, (List<?>) o);
        } else {
            // fallback: stringify
            sb.append('"').append(escape(o.toString())).append('"');
        }
    }

    private static void writeMap(StringBuilder sb, Map<String, Object> m) {
        sb.append('{');
        Iterator<Map.Entry<String, Object>> it = m.entrySet().iterator();
        while (it.hasNext()) {
            var e = it.next();
            sb.append('"').append(escape(e.getKey())).append('"').append(':');
            writeValue(sb, e.getValue());
            if (it.hasNext()) sb.append(',');
        }
        sb.append('}');
    }

    private static void writeList(StringBuilder sb, List<?> list) {
        sb.append('[');
        for (int i = 0; i < list.size(); i++) {
            writeValue(sb, list.get(i));
            if (i + 1 < list.size()) sb.append(',');
        }
        sb.append(']');
    }

    private static String escape(String s) {
        StringBuilder out = new StringBuilder(s.length() + 16);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': out.append("\\\""); break;
                case '\\': out.append("\\\\"); break;
                case '\b': out.append("\\b"); break;
                case '\f': out.append("\\f"); break;
                case '\n': out.append("\\n"); break;
                case '\r': out.append("\\r"); break;
                case '\t': out.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
            }
        }
        return out.toString();
    }
}

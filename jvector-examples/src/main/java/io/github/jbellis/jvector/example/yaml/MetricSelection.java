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

package io.github.jbellis.jvector.example.yaml;

import java.util.LinkedHashMap;
import java.util.List;

/**
 * Named metric selection, organized by category.
 *
 * YAML shape:
 * metrics:
 *   system: [max_heap_mb, max_offheap_mb]
 *   disk: [total_file_size_mb, file_count]
 *   construction: [index_build_time_s]
 *
 * This is intentionally a map subtype so we can deserialize the YAML
 * mapping directly into this type without extra nesting.
 */
public class MetricSelection extends LinkedHashMap<String, List<String>> {
}


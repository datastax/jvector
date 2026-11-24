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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Builds one graph for a {@link BuildConfig} and writes it once per feature set. */
interface IndexBuildStrategy {
    BuiltIndexes build(List<? extends Set<FeatureId>> featureSets, BuildConfig build, DataSet ds) throws IOException;

    /** The loaded indexes, keyed by feature set; the caller closes them. */
    final class BuiltIndexes {
        final Map<Set<FeatureId>, GraphIndex> indexes;
        /** The file each on-disk index was loaded from; empty for in-memory indexes. */
        final Map<Set<FeatureId>, Path> files;
        /** Seconds spent building, or null if every index was loaded from the cache. */
        final Double buildTimeS;
        /** Traversal codes stored with an index that does not fuse them, loaded when the index was opened. */
        final Map<Set<FeatureId>, StoredCodes.Loaded> storedCodes;

        BuiltIndexes(Map<Set<FeatureId>, GraphIndex> indexes, Map<Set<FeatureId>, Path> files, Double buildTimeS) {
            this(indexes, files, buildTimeS, Map.of());
        }

        BuiltIndexes(Map<Set<FeatureId>, GraphIndex> indexes, Map<Set<FeatureId>, Path> files, Double buildTimeS,
                     Map<Set<FeatureId>, StoredCodes.Loaded> storedCodes) {
            this.indexes = indexes;
            this.files = files;
            this.buildTimeS = buildTimeS;
            this.storedCodes = storedCodes;
        }

        /** Size of the codes file stored with the index for {@code features}, or null if it has none in use. */
        Long codesFileSize(Set<FeatureId> features) {
            var file = files.get(features);
            if (file == null || !storedCodes.containsKey(features)) return null;
            try {
                return Files.size(StoredCodes.pathFor(file));
            } catch (IOException unreadable) {
                return null;
            }
        }

        /** Size of the index file for {@code features}, or null for an in-memory or unreadable index. */
        Long fileSize(Set<FeatureId> features) {
            var file = files.get(features);
            if (file == null) return null;
            try {
                return Files.size(file);
            } catch (IOException unreadable) {
                return null;
            }
        }
    }
}

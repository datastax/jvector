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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import java.nio.file.Path;

class DataSetFiles {
    private final Path baseFvecsPath;
    private final Path queryFvecsPath;
    private final Path gtIvecsPath;

    DataSetFiles(Path baseFvecsPath, Path queryFvecsPath, Path gtIvecsPath) {
        this.baseFvecsPath = baseFvecsPath;
        this.queryFvecsPath = queryFvecsPath;
        this.gtIvecsPath = gtIvecsPath;
    }

    Path getBaseFvecsPath() {
        return baseFvecsPath;
    }

    Path getQueryFvecsPath() {
        return queryFvecsPath;
    }

    Path getGtIvecsPath() {
        return gtIvecsPath;
    }
}

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

import io.github.jbellis.jvector.example.benchmarks.datasets.ByteDataSet;
import io.github.jbellis.jvector.example.benchmarks.datasets.DataSet;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.benchmarks.datasets.FloatDataSet;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class CommonParameters {
    public List<Compression> compression;

    public List<Function<FloatDataSet, CompressorParameters>> getCompressorParameters(DataSet<?> ds) {
        if (ds instanceof ByteDataSet) {
            if (compression != null) {
                for (var c : compression) {
                    if (c.type != null && !c.type.equalsIgnoreCase("None")) {
                        throw new IllegalArgumentException(String.format(
                                "Compression type '%s' is not supported for INT8 dataset '%s'. INT8 datasets do not support compression.",
                                c.type, ds.getName()));
                    }
                }
            }
            return List.of(__ -> CompressorParameters.NONE);
        }

        if (compression == null) {
            return List.of(__ -> CompressorParameters.NONE);
        }
        return compression.stream().map(Compression::getCompressorParameters).collect(Collectors.toList());
    }
}

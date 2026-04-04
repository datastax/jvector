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
package io.github.jbellis.jvector.example.datasets;

import io.github.jbellis.jvector.example.benchmarks.datasets.DataSetMetadataReader;
import io.nosqlbench.vectordata.VectorTestData;
import io.nosqlbench.vectordata.downloader.Catalog;
import io.nosqlbench.vectordata.downloader.DatasetEntry;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class DataSetEnumerator {

    @Test
    public void testLoadOfficialSet() {
        DataSetMetadataReader metadata = DataSetMetadataReader.load();
        Catalog catalog = VectorTestData.catalogs().configure().catalog();
        List<String> found = new ArrayList<>();
        List<String> missing = new ArrayList<>();
        for (String rawname : metadata.keySet()) {
            String dsname = rawname.endsWith(".hdf5") ? rawname.substring(0, rawname.length() - 5) : rawname;
            Optional<DatasetEntry> entry = catalog.findExact(dsname);
            if (entry.isPresent()) {
                found.add(dsname);
            } else {
                missing.add(dsname);
            }
        }
        found.forEach(name -> System.out.println("FOUND: " + name));
        missing.forEach(name -> System.out.println("MISSING: " + name));
    }
    @Test
    public void testEnumerateDatasets() {

        List<String> datasetNames = List.of(
                "cohere-english-v3-100k",
                "ada002-100k",
                "ada002-1M",
                "openai-v3-large-1536-100k",
                "openai-v3-large-3072-100k",
                "openai-v3-small-100k",
                "colbert-1M",
                "gecko-100k",
                "e5-small-v2-100k",
                "e5-large-v2-100k",
                "e5-base-v2-100k",
                "glove-25-angular",
                "glove-50-angular",
                "glove-100-angular",
                "glove-200-angular",

                "dpr-1M",
                "dpr-10M",
                "cap-1M",
                "cap-6M",

                "lastfm-64-dot",
                "nytimes-256-angular",
                "sift-128-euclidean");


        Catalog catalog = VectorTestData.catalogs().configure().catalog();

        for (String dsname : datasetNames) {
            Optional<DatasetEntry> entry = catalog.findExact(dsname);
            if (entry.isPresent()) {
                System.out.println("FOUND: " + dsname);
            } else {
                System.out.println("MISSING: " + dsname);
            }

        }
    }
}

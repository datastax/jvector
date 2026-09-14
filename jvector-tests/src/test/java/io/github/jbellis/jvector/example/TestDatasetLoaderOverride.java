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

import io.github.jbellis.jvector.example.benchmarks.datasets.BaseVectorLoading;
import io.github.jbellis.jvector.example.yaml.ConstructionParameters;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.SearchParameters;
import io.github.jbellis.jvector.example.yaml.Compression;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** The VM override selects loading without changing resident construction/query routing. */
public class TestDatasetLoaderOverride {
    private static final String PROPERTY = "jvector.dataset_loader";
    private String previous;
    private final List<String> warnings = new ArrayList<>();
    private final Logger logger = Logger.getLogger(BaseVectorLoading.class.getName());
    private final Handler handler = new Handler() {
        @Override public void publish(LogRecord record) { warnings.add(record.getMessage()); }
        @Override public void flush() { }
        @Override public void close() { }
    };

    @Before public void saveProperty() {
        previous = System.getProperty(PROPERTY);
        logger.addHandler(handler);
    }

    @After public void restoreProperty() {
        logger.removeHandler(handler);
        if (previous == null) System.clearProperty(PROPERTY);
        else System.setProperty(PROPERTY, previous);
    }

    private static MultiConfig config(String build, String search) {
        var config = new MultiConfig();
        config.construction = new ConstructionParameters();
        config.search = new SearchParameters();
        var bc = new Compression(); bc.type = build;
        var sc = new Compression(); sc.type = search;
        config.construction.compression = List.of(bc);
        config.construction.fusedGraph = List.of(false);
        config.construction.reranking = List.of("NVQ");
        config.search.compression = List.of(sc);
        return config;
    }

    @Test public void explicitBufferedPropertyRetainsExactSearchRoutingAndWarns() {
        System.setProperty(PROPERTY, "buffered");
        assertEquals(BaseVectorLoading.PRELOAD, BaseVectorLoading.forBenchmark(config("PQ", "None")));
        assertTrue(warnings.stream().anyMatch(message -> message.contains("raw-input exact search")));
    }

    @Test public void explicitBufferedPropertyRetainsUncompressedConstructionAndWarns() {
        System.setProperty(PROPERTY, "buffered");
        assertEquals(BaseVectorLoading.PRELOAD, BaseVectorLoading.forBenchmark(config("None", "PQ")));
        assertTrue(warnings.stream().anyMatch(message -> message.contains("uncompressed construction")));
    }

    @Test public void explicitBufferedPropertyOverridesDisabledFlagForQuantizedWorkloads() {
        System.setProperty(PROPERTY, "buffered");
        assertEquals(BaseVectorLoading.BUFFERED, BaseVectorLoading.forBenchmark(config("PQ", "PQ")));
        assertTrue(warnings.isEmpty());
    }

    @Test public void explicitPreloadPropertyOverridesEnabledFlag() {
        System.setProperty(PROPERTY, "preload");
        var config = config("PQ", "PQ");
        config.useDatasetPrefetching = true;
        assertEquals(BaseVectorLoading.PRELOAD, BaseVectorLoading.forBenchmark(config));
        assertTrue(warnings.isEmpty());
    }
}

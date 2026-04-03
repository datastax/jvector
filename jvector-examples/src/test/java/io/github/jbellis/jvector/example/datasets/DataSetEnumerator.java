package io.github.jbellis.jvector.example.datasets;

import io.nosqlbench.vectordata.VectorTestData;
import io.nosqlbench.vectordata.downloader.Catalog;
import io.nosqlbench.vectordata.downloader.DatasetEntry;
import org.junit.Test;

import java.util.List;
import java.util.Optional;

public class DataSetEnumerator {

    @Test
    public void testEnumerateDatasets() {

        List<String> datasetNames = List.of(
                "cohere-english-v3-100k",
                "ada002-100k",
                "openai-v3-small-100k",
                "gecko-100k",
                "openai-v3-large-3072-100k",
                "openai-v3-large-1536-100k",
                "e5-small-v2-100k",
                "e5-base-v2-100k",
                "e5-large-v2-100k",
                "ada002-1M",
                "colbert-1M",
                "glove-25-angular",
                "glove-50-angular",
                "lastfm-64-dot",
                "glove-100-angular",
                "glove-200-angular",
                "nytimes-256-angular",
                "sift-128-euclidean");


        Catalog catalog = VectorTestData.catalogs().catalog();

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

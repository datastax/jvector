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

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/// Tests for {@link DataSetLoaderSimpleMFD} using local files only, with no remote endpoint.
public class DataSetLoaderSimpleMFDTest {

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    private Path cacheDir;
    private DataSetMetadataReader testMetadata;

    @Before
    public void setUp() throws IOException {
        cacheDir = tempFolder.newFolder("datasets").toPath();

        // create a test-only metadata file instead of modifying the production one
        Path metadataFile = tempFolder.newFile("test_metadata.yml").toPath();
        Files.writeString(metadataFile,
                "test-ds:\n" +
                "  similarity_function: COSINE\n" +
                "  load_behavior: NO_SCRUB\n");
        testMetadata = DataSetMetadataReader.load(metadataFile.toString());
    }

    @Test
    public void loadsDatasetFromLocalCatalogAndFiles() throws IOException {
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        // use a non-connectable remote URL — the local catalog should be sufficient
        var loader = new DataSetLoaderSimpleMFD(
                "http://0.0.0.0:1/datasets.yaml",
                cacheDir.toString(),
                false,
                testMetadata
        );

        var info = loader.loadDataSet("test-ds");
        assertTrue(info.isPresent(), "Dataset should be found in local catalog");
        assertEquals("test-ds", info.get().getName());

        var ds = info.get().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
        assertEquals(2, ds.getQueryVectors().size());
        assertEquals(2, ds.getGroundTruth().size());
        assertEquals(4, ds.getDimension());
    }

    @Test
    public void returnsEmptyForUnknownDataset() throws IOException {
        writeTestCatalog(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                "http://0.0.0.0:1/datasets.yaml",
                cacheDir.toString(),
                false,
                testMetadata
        );

        var info = loader.loadDataSet("nonexistent-dataset");
        assertFalse(info.isPresent());
    }

    @Test
    public void failsWhenLocalFilesMissing() throws IOException {
        // catalog exists but data files don't, and remote is unreachable
        writeTestCatalog(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                "http://0.0.0.0:1/datasets.yaml",
                cacheDir.toString(),
                false,
                testMetadata
        );

        assertThrows(RuntimeException.class, () -> loader.loadDataSet("test-ds"));
    }

    @Test
    public void failsWhenNoCatalogAndRemoteUnreachable() {
        // no local catalog, and remote is unreachable
        assertThrows(RuntimeException.class, () -> new DataSetLoaderSimpleMFD(
                "http://0.0.0.0:1/datasets.yaml",
                cacheDir.toString(),
                false,
                testMetadata
        ));
    }

    @Test
    public void checkForUpdatesDoesNotFailWhenRemoteUnreachable() throws IOException {
        // local catalog exists; checkForUpdates=true but remote is down — should still work
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        // should not throw — it logs a warning but proceeds with the local catalog
        var loader = new DataSetLoaderSimpleMFD(
                "http://0.0.0.0:1/datasets.yaml",
                cacheDir.toString(),
                true,
                testMetadata
        );

        var info = loader.loadDataSet("test-ds");
        assertTrue(info.isPresent());
    }

    @Test
    public void rejectsCatalogWithMissingFields() throws IOException {
        // write a catalog with a dataset missing the 'gt' field
        String yaml = "bad-ds:\n  base: b.fvecs\n  query: q.fvecs\n";
        Files.writeString(cacheDir.resolve("datasets.yaml"), yaml);

        var loader = new DataSetLoaderSimpleMFD(
                "http://0.0.0.0:1/datasets.yaml",
                cacheDir.toString(),
                false,
                testMetadata
        );

        var info = loader.loadDataSet("bad-ds");
        assertFalse(info.isPresent(), "Should return empty for dataset with missing catalog fields");
    }

    @Test
    public void loadsFromLocalPathWithoutRemote() throws IOException {
        // use a local file path instead of an HTTP URL
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        var catalogPath = cacheDir.resolve("datasets.yaml").toString();
        var loader = new DataSetLoaderSimpleMFD(
                catalogPath,
                cacheDir.toString(),
                false,
                testMetadata
        );

        var info = loader.loadDataSet("test-ds");
        assertTrue(info.isPresent());

        var ds = info.get().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
        assertEquals(4, ds.getDimension());
    }

    @Test
    public void nullCatalogUrlWorksWithLocalCatalog() throws IOException {
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null,
                cacheDir.toString(),
                false,
                testMetadata
        );

        var info = loader.loadDataSet("test-ds");
        assertTrue(info.isPresent());

        var ds = info.get().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
        assertEquals(4, ds.getDimension());
    }

    @Test
    public void emptyCatalogUrlWorksWithLocalCatalog() throws IOException {
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                "",
                cacheDir.toString(),
                false,
                testMetadata
        );

        var info = loader.loadDataSet("test-ds");
        assertTrue(info.isPresent());

        var ds = info.get().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void nullCatalogUrlFailsWithoutLocalCatalog() {
        // no remote, no local catalog — must fail
        assertThrows(IllegalArgumentException.class, () -> new DataSetLoaderSimpleMFD(
                null,
                cacheDir.toString(),
                false,
                testMetadata
        ));
    }

    @Test
    public void emptyCatalogUrlFailsWithoutLocalCatalog() {
        // no remote, no local catalog — must fail
        assertThrows(IllegalArgumentException.class, () -> new DataSetLoaderSimpleMFD(
                "",
                cacheDir.toString(),
                false,
                testMetadata
        ));
    }

    @Test
    public void nullCatalogUrlIgnoresCheckForUpdates() throws IOException {
        // checkForUpdates=true should be harmless when there's no remote
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null,
                cacheDir.toString(),
                true,
                testMetadata
        );

        var info = loader.loadDataSet("test-ds");
        assertTrue(info.isPresent());
    }

    @Test
    public void localOnlyFailsWhenDataFileMissing() throws IOException {
        // catalog exists but data files don't — with no remote, should fail clearly
        writeTestCatalog(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null,
                cacheDir.toString(),
                false,
                testMetadata
        );

        var ex = assertThrows(RuntimeException.class, () -> loader.loadDataSet("test-ds"));
        assertTrue(ex.getCause().getMessage().contains("no remote URL configured"),
                "Error should indicate no remote is available: " + ex.getCause().getMessage());
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    private static void writeTestCatalog(Path dir) throws IOException {
        String yaml = "test-ds:\n"
                + "  base: test_base.fvecs\n"
                + "  query: test_query.fvecs\n"
                + "  gt: test_gt.ivecs\n";
        Files.writeString(dir.resolve("datasets.yaml"), yaml);
    }

    private static void writeTestDataFiles(Path dir) throws IOException {
        writeTestFvecs(dir.resolve("test_base.fvecs"), 4, new float[][] {
                {1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 0.0f, 1.0f},
                {0.5f, 0.5f, 0.5f, 0.5f},
        });
        writeTestFvecs(dir.resolve("test_query.fvecs"), 4, new float[][] {
                {1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 1.0f, 0.0f},
        });
        writeTestIvecs(dir.resolve("test_gt.ivecs"), new int[][] {
                {0, 4, 1, 2, 3},
                {2, 4, 0, 1, 3},
        });
    }

    /// Writes vectors in the standard fvecs format: for each vector, a 4-byte LE int (dimension)
    /// followed by dimension * 4-byte LE floats.
    private static void writeTestFvecs(Path path, int dimension, float[][] vectors) throws IOException {
        int bytesPerVector = Integer.BYTES + dimension * Float.BYTES;
        var buf = ByteBuffer.allocate(vectors.length * bytesPerVector).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] vec : vectors) {
            buf.putInt(dimension);
            for (float v : vec) {
                buf.putFloat(v);
            }
        }
        Files.write(path, buf.array());
    }

    /// Writes ground truth in the standard ivecs format: for each entry, a 4-byte LE int (count)
    /// followed by count * 4-byte LE ints.
    private static void writeTestIvecs(Path path, int[][] entries) throws IOException {
        int totalBytes = 0;
        for (int[] entry : entries) {
            totalBytes += Integer.BYTES + entry.length * Integer.BYTES;
        }
        var buf = ByteBuffer.allocate(totalBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int[] entry : entries) {
            buf.putInt(entry.length);
            for (int v : entry) {
                buf.putInt(v);
            }
        }
        Files.write(path, buf.array());
    }
}

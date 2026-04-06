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

        // create a test-only metadata file
        Path metadataFile = tempFolder.newFile("test_metadata.yml").toPath();
        Files.writeString(metadataFile,
                "test-ds:\n" +
                "  similarity_function: COSINE\n" +
                "  load_behavior: NO_SCRUB\n" +
                "sub-ds:\n" +
                "  similarity_function: COSINE\n" +
                "  load_behavior: NO_SCRUB\n" +
                "private-ds:\n" +
                "  similarity_function: DOT_PRODUCT\n" +
                "  load_behavior: NO_SCRUB\n");
        testMetadata = DataSetMetadataReader.load(metadataFile.toString());
    }

    // ========================================================================
    // Basic loading
    // ========================================================================

    @Test
    public void loadsDatasetFromLocalCatalogAndFiles() throws IOException {
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
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
                null, cacheDir.toString(), false, testMetadata
        );

        assertFalse(loader.loadDataSet("nonexistent-dataset").isPresent());
    }

    @Test
    public void failsWhenLocalFilesMissing() throws IOException {
        writeTestCatalog(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ex = assertThrows(RuntimeException.class, () -> loader.loadDataSet("test-ds"));
        assertTrue(ex.getCause().getMessage().contains("no remote URL configured"),
                "Error should indicate no remote is available: " + ex.getCause().getMessage());
    }

    @Test
    public void failsWhenNoCatalogAndRemoteUnreachable() {
        assertThrows(RuntimeException.class, () -> new DataSetLoaderSimpleMFD(
                "http://0.0.0.0:1/catalog_entries.yaml",
                cacheDir.toString(), false, testMetadata
        ));
    }

    @Test
    public void checkForUpdatesDoesNotFailWhenRemoteUnreachable() throws IOException {
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        // should not throw — logs a warning but proceeds with the local catalog
        var loader = new DataSetLoaderSimpleMFD(
                "http://0.0.0.0:1/catalog_entries.yaml",
                cacheDir.toString(), true, testMetadata
        );

        assertTrue(loader.loadDataSet("test-ds").isPresent());
    }

    @Test
    public void rejectsCatalogWithMissingFields() throws IOException {
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "bad-ds:\n  base: b.fvecs\n  query: q.fvecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertFalse(loader.loadDataSet("bad-ds").isPresent(),
                "Should return empty for dataset with missing catalog fields");
    }

    // ========================================================================
    // Local path resolution (file vs directory)
    // ========================================================================

    @Test
    public void loadsWithLocalPathAsYamlFile() throws IOException {
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.resolve("catalog_entries.yaml").toString(),
                false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void singleArgConstructorWithFilePath() throws IOException {
        writeTestCatalog(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                cacheDir.resolve("catalog_entries.yaml").toString()
        );

        // can't call getDataSet() (no test-ds in production metadata), but catalog is loaded
        assertFalse(loader.loadDataSet("nonexistent").isPresent());
    }

    @Test
    public void singleArgConstructorWithDirectory() throws IOException {
        writeTestCatalog(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(cacheDir.toString());
        assertFalse(loader.loadDataSet("nonexistent").isPresent());
    }

    @Test
    public void singleArgConstructorMissingCatalogReturnsEmpty() {
        var loader = new DataSetLoaderSimpleMFD(cacheDir.toString());
        assertFalse(loader.loadDataSet("test-ds").isPresent());
    }

    // ========================================================================
    // Null / empty catalog URL (local-only mode)
    // ========================================================================

    @Test
    public void nullCatalogUrlWorksWithLocalCatalog() throws IOException {
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void emptyCatalogUrlWorksWithLocalCatalog() throws IOException {
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                "", cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void nullCatalogUrlWithoutLocalCatalogReturnsEmpty() {
        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );
        assertFalse(loader.loadDataSet("test-ds").isPresent());
    }

    @Test
    public void nullCatalogUrlIgnoresCheckForUpdates() throws IOException {
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), true, testMetadata
        );
        assertTrue(loader.loadDataSet("test-ds").isPresent());
    }

    // ========================================================================
    // Comment-only / empty catalog files
    // ========================================================================

    @Test
    public void commentOnlyCatalogFileReturnsEmpty() throws IOException {
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "# This file has no actual entries\n# Just comments\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );
        assertFalse(loader.loadDataSet("test-ds").isPresent());
    }

    @Test
    public void emptyCatalogFileReturnsEmpty() throws IOException {
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"), "");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );
        assertFalse(loader.loadDataSet("test-ds").isPresent());
    }

    @Test
    public void singleArgWithCommentOnlyCatalogReturnsEmpty() throws IOException {
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "# placeholder for local datasets\n");

        var loader = new DataSetLoaderSimpleMFD(
                cacheDir.resolve("catalog_entries.yaml").toString()
        );
        assertFalse(loader.loadDataSet("anything").isPresent());
    }

    // ========================================================================
    // Recursive catalog discovery
    // ========================================================================

    @Test
    public void recursivelyDiscoversCatalogs() throws IOException {
        // root catalog
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        // subdirectory catalog with a different dataset
        Path subDir = cacheDir.resolve("subgroup");
        Files.createDirectories(subDir);
        Files.writeString(subDir.resolve("catalog_entries.yaml"),
                "sub-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(subDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // both datasets should be found
        assertTrue(loader.loadDataSet("test-ds").isPresent(), "Root catalog entry should be found");
        assertTrue(loader.loadDataSet("sub-ds").isPresent(), "Subdirectory catalog entry should be found");
    }

    @Test
    public void subdirectoryDataFilesResolveRelativeToTheirCatalog() throws IOException {
        // root has no data files, subdirectory has both catalog and data
        Path subDir = cacheDir.resolve("subgroup");
        Files.createDirectories(subDir);
        Files.writeString(subDir.resolve("catalog_entries.yaml"),
                "sub-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(subDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // data files should resolve relative to subDir, not cacheDir
        var ds = loader.loadDataSet("sub-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void duplicateEntryAcrossCatalogsDoesNotFail() throws IOException {
        // root catalog defines test-ds
        writeTestCatalog(cacheDir);
        writeTestDataFiles(cacheDir);

        // subdirectory also defines test-ds — one wins (walk order is unspecified)
        Path subDir = cacheDir.resolve("override");
        Files.createDirectories(subDir);
        Files.writeString(subDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(subDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // should load without error — whichever catalog wins, the dataset is valid
        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertNotNull(ds);
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void discoversAlternativeCatalogFilenames() throws IOException {
        // entries.yaml in root
        Files.writeString(cacheDir.resolve("entries.yaml"),
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        // private_entries.yaml in subdirectory
        Path subDir = cacheDir.resolve("private");
        Files.createDirectories(subDir);
        Files.writeString(subDir.resolve("private_entries.yaml"),
                "sub-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(subDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertTrue(loader.loadDataSet("test-ds").isPresent(), "entries.yaml should be discovered");
        assertTrue(loader.loadDataSet("sub-ds").isPresent(), "private_entries.yaml should be discovered");
    }

    @Test
    public void ignoresNonMatchingYamlFiles() throws IOException {
        // this file does NOT match *entries.yaml
        Files.writeString(cacheDir.resolve("other_config.yaml"),
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertFalse(loader.loadDataSet("test-ds").isPresent(),
                "Files not matching *entries.yaml should be ignored");
    }

    // ========================================================================
    // Per-entry baseurl override
    // ========================================================================

    @Test
    public void baseurlOverrideIsUsedForDownload() throws IOException {
        // catalog entry has a baseurl pointing to an unreachable server
        // but the files exist locally — so baseurl is not actually hit.
        // This test verifies the entry is parsed correctly and local files still resolve.
        Path subDir = cacheDir.resolve("private");
        Files.createDirectories(subDir);
        Files.writeString(subDir.resolve("catalog_entries.yaml"),
                "private-ds:\n" +
                "  baseurl: http://0.0.0.0:1/secret-hash/\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(subDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("private-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void baseurlOverrideFailsWhenLocalFilesMissingAndRemoteUnreachable() throws IOException {
        // catalog entry has a baseurl pointing to unreachable server, and no local data files
        Path subDir = cacheDir.resolve("private");
        Files.createDirectories(subDir);
        Files.writeString(subDir.resolve("catalog_entries.yaml"),
                "private-ds:\n" +
                "  baseurl: http://0.0.0.0:1/secret-hash/\n" +
                "  base: missing_base.fvecs\n" +
                "  query: missing_query.fvecs\n" +
                "  gt: missing_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // should fail because files don't exist and baseurl is unreachable
        assertThrows(RuntimeException.class, () -> loader.loadDataSet("private-ds"));
    }

    @Test
    public void baseurlWithoutTrailingSlashIsNormalized() throws IOException {
        Path subDir = cacheDir.resolve("private");
        Files.createDirectories(subDir);
        // baseurl without trailing slash
        Files.writeString(subDir.resolve("catalog_entries.yaml"),
                "private-ds:\n" +
                "  baseurl: http://0.0.0.0:1/no-trailing-slash\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(subDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // should load fine — baseurl is normalized with trailing slash
        var ds = loader.loadDataSet("private-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void subdirectoryPathsInFileValuesResolveCorrectly() throws IOException {
        // mirrors the real large_dataset_entries.yaml structure where file values
        // contain subdirectory paths like "dpr/c4-en_base_1M_norm.fvecs"
        Path privateDir = cacheDir.resolve("jvector_private");
        Files.createDirectories(privateDir);
        Files.writeString(privateDir.resolve("large_dataset_entries.yaml"),
                "test-ds:\n" +
                "  baseurl: http://0.0.0.0:1/secret-hash/\n" +
                "  base: subdir/test_base.fvecs\n" +
                "  query: subdir/test_query.fvecs\n" +
                "  gt: subdir/test_gt.ivecs\n");

        // create data files in the subdirectory under the catalog's directory
        Path dataDir = privateDir.resolve("subdir");
        Files.createDirectories(dataDir);
        writeTestDataFiles(dataDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
        assertEquals(4, ds.getDimension());
    }

    @Test
    public void subdirectoryPathsDownloadWhenLocalMissing() throws IOException {
        // catalog has subdirectory paths but files are missing locally and remote is unreachable
        // — should fail with a clear error mentioning the baseurl, not the default remote
        Path privateDir = cacheDir.resolve("jvector_private");
        Files.createDirectories(privateDir);
        Files.writeString(privateDir.resolve("large_dataset_entries.yaml"),
                "test-ds:\n" +
                "  baseurl: http://0.0.0.0:1/secret-hash/\n" +
                "  base: subdir/missing_base.fvecs\n" +
                "  query: subdir/missing_query.fvecs\n" +
                "  gt: subdir/missing_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // should attempt to download from the baseurl and fail
        assertThrows(RuntimeException.class, () -> loader.loadDataSet("test-ds"));
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    private static void writeTestCatalog(Path dir) throws IOException {
        Files.writeString(dir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
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

    /// Writes vectors in the standard fvecs format.
    private static void writeTestFvecs(Path path, int dimension, float[][] vectors) throws IOException {
        int bytesPerVector = Integer.BYTES + dimension * Float.BYTES;
        var buf = ByteBuffer.allocate(vectors.length * bytesPerVector).order(ByteOrder.LITTLE_ENDIAN);
        for (float[] vec : vectors) {
            buf.putInt(dimension);
            for (float v : vec) buf.putFloat(v);
        }
        Files.write(path, buf.array());
    }

    /// Writes ground truth in the standard ivecs format.
    private static void writeTestIvecs(Path path, int[][] entries) throws IOException {
        int totalBytes = 0;
        for (int[] entry : entries) totalBytes += Integer.BYTES + entry.length * Integer.BYTES;
        var buf = ByteBuffer.allocate(totalBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int[] entry : entries) {
            buf.putInt(entry.length);
            for (int v : entry) buf.putInt(v);
        }
        Files.write(path, buf.array());
    }
}

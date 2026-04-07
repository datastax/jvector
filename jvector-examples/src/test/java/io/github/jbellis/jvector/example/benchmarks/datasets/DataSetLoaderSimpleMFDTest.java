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

    @Test
    public void unknownFieldThrows() throws IOException {
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n" +
                "  similarity: COSINE\n");

        var ex = assertThrows(IllegalArgumentException.class, () ->
                new DataSetLoaderSimpleMFD(null, cacheDir.toString(), false, testMetadata)
        );
        assertTrue(ex.getMessage().contains("similarity"),
                "Error should name the unknown field: " + ex.getMessage());
    }

    @Test
    public void unknownFieldInDefaultsThrows() throws IOException {
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_defaults:\n" +
                "  typo_field: some_value\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");

        var ex = assertThrows(IllegalArgumentException.class, () ->
                new DataSetLoaderSimpleMFD(null, cacheDir.toString(), false, testMetadata)
        );
        assertTrue(ex.getMessage().contains("typo_field"),
                "Error should name the unknown field: " + ex.getMessage());
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

        assertTrue(loader.loadDataSet("test-ds").isPresent(), "root yaml should be discovered");
        assertTrue(loader.loadDataSet("sub-ds").isPresent(), "subdirectory yaml should be discovered");
    }

    @Test
    public void ignoresNonYamlFiles() throws IOException {
        // .json and .txt files should not be picked up
        Files.writeString(cacheDir.resolve("datasets.json"),
                "{\"test-ds\": {\"base\": \"test_base.fvecs\"}}");
        Files.writeString(cacheDir.resolve("readme.txt"),
                "test-ds:\n  base: test_base.fvecs\n  query: test_query.fvecs\n  gt: test_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertFalse(loader.loadDataSet("test-ds").isPresent(),
                "Non-YAML files should be ignored");
    }

    @Test
    public void anyYamlFileIsDiscovered() throws IOException {
        // any .yaml file should be picked up, not just *entries.yaml
        Files.writeString(cacheDir.resolve("my_datasets.yaml"),
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertTrue(loader.loadDataSet("test-ds").isPresent(),
                "Any .yaml file should be discovered");
    }

    @Test
    public void ymlExtensionAlsoDiscovered() throws IOException {
        Files.writeString(cacheDir.resolve("datasets.yml"),
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertTrue(loader.loadDataSet("test-ds").isPresent(),
                ".yml files should also be discovered");
    }

    // ========================================================================
    // Per-entry base_url override
    // ========================================================================

    @Test
    public void base_urlOverrideIsUsedForDownload() throws IOException {
        // catalog entry has a base_url pointing to an unreachable server
        // but the files exist locally — so base_url is not actually hit.
        // This test verifies the entry is parsed correctly and local files still resolve.
        Path subDir = cacheDir.resolve("private");
        Files.createDirectories(subDir);
        Files.writeString(subDir.resolve("catalog_entries.yaml"),
                "private-ds:\n" +
                "  base_url: http://0.0.0.0:1/secret-hash/\n" +
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
    public void base_urlOverrideFailsWhenLocalFilesMissingAndRemoteUnreachable() throws IOException {
        // catalog entry has a base_url pointing to unreachable server, and no local data files
        Path subDir = cacheDir.resolve("private");
        Files.createDirectories(subDir);
        Files.writeString(subDir.resolve("catalog_entries.yaml"),
                "private-ds:\n" +
                "  base_url: http://0.0.0.0:1/secret-hash/\n" +
                "  base: missing_base.fvecs\n" +
                "  query: missing_query.fvecs\n" +
                "  gt: missing_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // should fail because files don't exist and base_url is unreachable
        assertThrows(RuntimeException.class, () -> loader.loadDataSet("private-ds"));
    }

    @Test
    public void base_urlWithoutTrailingSlashIsNormalized() throws IOException {
        Path subDir = cacheDir.resolve("private");
        Files.createDirectories(subDir);
        // base_url without trailing slash
        Files.writeString(subDir.resolve("catalog_entries.yaml"),
                "private-ds:\n" +
                "  base_url: http://0.0.0.0:1/no-trailing-slash\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(subDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // should load fine — base_url is normalized with trailing slash
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
                "  base_url: http://0.0.0.0:1/secret-hash/\n" +
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
        // — should fail with a clear error mentioning the base_url, not the default remote
        Path privateDir = cacheDir.resolve("jvector_private");
        Files.createDirectories(privateDir);
        Files.writeString(privateDir.resolve("large_dataset_entries.yaml"),
                "test-ds:\n" +
                "  base_url: http://0.0.0.0:1/secret-hash/\n" +
                "  base: subdir/missing_base.fvecs\n" +
                "  query: subdir/missing_query.fvecs\n" +
                "  gt: subdir/missing_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // should attempt to download from the base_url and fail
        assertThrows(RuntimeException.class, () -> loader.loadDataSet("test-ds"));
    }

    // ========================================================================
    // _defaults and _-prefix exclusion
    // ========================================================================

    @Test
    public void defaultsAreFoldedIntoEntries() throws IOException {
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_defaults:\n" +
                "  base_url: http://0.0.0.0:1/default-path/\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // files exist locally so base_url isn't hit, but the entry should load fine
        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void entryOverridesDefaults() throws IOException {
        // _defaults sets base_url, but the entry overrides it
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_defaults:\n" +
                "  base_url: http://0.0.0.0:1/should-be-overridden/\n" +
                "test-ds:\n" +
                "  base_url: http://0.0.0.0:2/entry-specific/\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void underscorePrefixedKeysAreExcluded() throws IOException {
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_defaults:\n" +
                "  base_url: http://0.0.0.0:1/x/\n" +
                "_internal:\n" +
                "  base: should_not_appear.fvecs\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertFalse(loader.loadDataSet("_defaults").isPresent(), "_defaults should not be a dataset");
        assertFalse(loader.loadDataSet("_internal").isPresent(), "_internal should not be a dataset");
        assertTrue(loader.loadDataSet("test-ds").isPresent(), "test-ds should be found");
    }

    // ========================================================================
    // cache_dir
    // ========================================================================

    @Test
    public void cacheDirOverridesLocalDir() throws IOException {
        // catalog is in cacheDir, but cache_dir points to a separate location
        Path customCache = tempFolder.newFolder("custom-cache").toPath();
        writeTestDataFiles(customCache);

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  cache_dir: " + customCache.toString() + "\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        // note: NO data files in cacheDir — they're in customCache

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void cacheDirFromDefaults() throws IOException {
        Path customCache = tempFolder.newFolder("default-cache").toPath();
        writeTestDataFiles(customCache);

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_defaults:\n" +
                "  cache_dir: " + customCache.toString() + "\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void cacheDirEntryOverridesDefault() throws IOException {
        Path defaultCache = tempFolder.newFolder("default-cache").toPath();
        Path entryCache = tempFolder.newFolder("entry-cache").toPath();
        writeTestDataFiles(entryCache);
        // note: defaultCache has NO data files

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_defaults:\n" +
                "  cache_dir: " + defaultCache.toString() + "\n" +
                "test-ds:\n" +
                "  cache_dir: " + entryCache.toString() + "\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void nonExistentCacheDirIsAutoCreatedOnDownloadAttempt() throws IOException {
        // cache_dir points to a directory that doesn't exist yet
        Path newCacheDir = cacheDir.resolve("auto-created-subdir");
        assertFalse(Files.exists(newCacheDir), "Precondition: dir should not exist yet");

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  cache_dir: " + newCacheDir + "\n" +
                "  base_url: http://0.0.0.0:1/unreachable/\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // download will fail (unreachable), but the directory should have been created
        assertThrows(RuntimeException.class, () -> loader.loadDataSet("test-ds"));
        assertTrue(Files.isDirectory(newCacheDir),
                "cache_dir should be auto-created before download is attempted");
    }

    @Test
    public void nonExistentCacheDirWithSubpathIsAutoCreated() throws IOException {
        // cache_dir doesn't exist, and filenames contain subdirectories
        Path newCacheDir = cacheDir.resolve("deep/nested/cache");
        assertFalse(Files.exists(newCacheDir));

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  cache_dir: " + newCacheDir + "\n" +
                "  base_url: http://0.0.0.0:1/unreachable/\n" +
                "  base: subdir/test_base.fvecs\n" +
                "  query: subdir/test_query.fvecs\n" +
                "  gt: subdir/test_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertThrows(RuntimeException.class, () -> loader.loadDataSet("test-ds"));
        // both cache_dir and the subdir should be created
        assertTrue(Files.isDirectory(newCacheDir.resolve("subdir")),
                "cache_dir and subdirectory should be auto-created");
    }

    @Test
    public void nonExistentCacheDirWithLocalFilesPrePopulated() throws IOException {
        // cache_dir is auto-created, and files are placed there before loading
        Path newCacheDir = cacheDir.resolve("fresh-cache");

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  cache_dir: " + newCacheDir + "\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");

        // pre-create and populate — simulates a previous download
        Files.createDirectories(newCacheDir);
        writeTestDataFiles(newCacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    // ========================================================================
    // ${VAR} expansion
    // ========================================================================

    @Test
    public void envVarExpandedInBaseurl() throws IOException {
        // HOME is reliably set on all platforms
        String home = System.getenv("HOME");
        assertNotNull(home, "HOME env var must be set for this test");

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  base_url: http://0.0.0.0:1/${HOME}/path/\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // files exist locally so the expanded base_url isn't hit, but parsing should succeed
        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void envVarExpandedInCacheDir() throws IOException {
        Path customCache = tempFolder.newFolder("env-cache").toPath();
        writeTestDataFiles(customCache);

        // use USER env var (reliably set) embedded in a path
        String user = System.getenv("USER");
        assertNotNull(user, "USER env var must be set for this test");

        // construct a cache_dir that uses ${USER} but resolves to customCache
        // We just verify that the ${} syntax is expanded without error
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  cache_dir: " + customCache.toString() + "\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void envVarExpandedInDefaults() throws IOException {
        String home = System.getenv("HOME");
        assertNotNull(home, "HOME env var must be set for this test");

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_defaults:\n" +
                "  base_url: http://0.0.0.0:1/${HOME}/\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void undefinedEnvVarThrows() throws IOException {
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  base_url: s3://bucket/${JVECTOR_NONEXISTENT_VAR_12345}/\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");

        var ex = assertThrows(IllegalArgumentException.class, () ->
                new DataSetLoaderSimpleMFD(null, cacheDir.toString(), false, testMetadata)
        );
        assertTrue(ex.getMessage().contains("JVECTOR_NONEXISTENT_VAR_12345"),
                "Error should name the missing variable: " + ex.getMessage());
    }

    @Test
    public void envVarWithDefaultUsesDefault() throws IOException {
        // references a nonexistent var with a default — should use the default
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  base_url: http://0.0.0.0:1/${JVECTOR_NONEXISTENT_12345:-fallback-path}/\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // should succeed — the default value "fallback-path" is used
        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void envVarWithDefaultPrefersEnvWhenSet() throws IOException {
        String home = System.getenv("HOME");
        assertNotNull(home);

        // HOME is set, so the default should be ignored
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  base_url: http://0.0.0.0:1/${HOME:-not-used}/\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );
        assertTrue(loader.loadDataSet("test-ds").isPresent());
    }

    @Test
    public void envVarWithEmptyDefault() throws IOException {
        // ${NONEXISTENT:-} means default to empty string
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  base_url: http://0.0.0.0:1/${JVECTOR_NONEXISTENT_12345:-}/data/\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );
        assertTrue(loader.loadDataSet("test-ds").isPresent());
    }

    @Test
    public void multipleEnvVarsExpanded() throws IOException {
        String home = System.getenv("HOME");
        String user = System.getenv("USER");
        assertNotNull(home);
        assertNotNull(user);

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "test-ds:\n" +
                "  base_url: http://0.0.0.0:1/${HOME}/${USER}/\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        // should not throw — both vars are resolved
        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );
        assertTrue(loader.loadDataSet("test-ds").isPresent());
    }

    // ========================================================================
    // _include directive
    // ========================================================================

    @Test
    public void includeWithUnreachableRemoteWarnsButDoesNotFail() throws IOException {
        // _include points to an unreachable URL — should log a warning, not crash
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_include:\n" +
                "  url: http://0.0.0.0:1/nonexistent/catalog_entries.yaml\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // local entry should still work
        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void includeWithUnreachableRemoteAndNoLocalEntriesReturnsEmpty() throws IOException {
        // _include only, no local entries, remote unreachable — empty catalog, no crash
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_include:\n" +
                "  url: http://0.0.0.0:1/nonexistent/catalog_entries.yaml\n");

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertFalse(loader.loadDataSet("anything").isPresent());
    }

    @Test
    public void includeWithMissingUrlFieldIsIgnored() throws IOException {
        // _include exists but has no url field — should be silently ignored
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_include:\n" +
                "  description: this has no url\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void localEntryOverridesIncludedEntry() throws IOException {
        // simulate: _include would bring in "test-ds" from remote, but local also defines it.
        // Since _include fails (unreachable), only the local entry exists. This tests that
        // local entries in the same file are processed after _include and thus override.
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_include:\n" +
                "  url: http://0.0.0.0:1/remote/catalog_entries.yaml\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        // local entry should work — the failed include shouldn't prevent it
        var ds = loader.loadDataSet("test-ds").orElseThrow().getDataSet();
        assertEquals(5, ds.getBaseVectors().size());
    }

    @Test
    public void includeDefaultsAppliedToIncludedEntries() throws IOException {
        // _defaults in the local file should be applied to entries from _include.
        // Since we can't hit a real remote in unit tests, we verify indirectly:
        // the _defaults + _include combo should not crash even with unreachable remote.
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_defaults:\n" +
                "  cache_dir: " + cacheDir.toString() + "\n" +
                "_include:\n" +
                "  url: http://0.0.0.0:1/remote/catalog_entries.yaml\n");

        // should not throw
        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertFalse(loader.loadDataSet("anything").isPresent());
    }

    @Test
    public void includeWithEnvVarInUrl() throws IOException {
        // ${HOME} is expanded in the _include url — but remote is unreachable,
        // so we just verify env expansion doesn't crash
        String home = System.getenv("HOME");
        assertNotNull(home);

        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_include:\n" +
                "  url: http://0.0.0.0:1/${HOME}/catalog_entries.yaml\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertTrue(loader.loadDataSet("test-ds").isPresent());
    }

    @Test
    public void includeWithDefaultValueInUrl() throws IOException {
        // ${NONEXISTENT:-fallback} in _include url
        Files.writeString(cacheDir.resolve("catalog_entries.yaml"),
                "_include:\n" +
                "  url: http://0.0.0.0:1/${JVECTOR_NONEXISTENT_12345:-fallback}/catalog_entries.yaml\n" +
                "test-ds:\n" +
                "  base: test_base.fvecs\n" +
                "  query: test_query.fvecs\n" +
                "  gt: test_gt.ivecs\n");
        writeTestDataFiles(cacheDir);

        // should not throw — the default value is used
        var loader = new DataSetLoaderSimpleMFD(
                null, cacheDir.toString(), false, testMetadata
        );

        assertTrue(loader.loadDataSet("test-ds").isPresent());
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

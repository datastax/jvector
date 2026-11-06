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

package io.github.jbellis.jvector.fastervector.bench;

import io.github.jbellis.jvector.example.yaml.DatasetCollection;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.ConstructionPoint;
import io.github.jbellis.jvector.fastervector.bench.IvfConfig.SearchPoint;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

// Surefire runs from the repository root, so the default directory resolves as it does for the runners
class IvfConfigTest {
    private static final String VALID = """
            yamlSchemaVersion: 1
            dataset: cap-1M
            construction:
              nlist: [256, 1024]
              storageFactor: [1.0, 2.0]
              buildThreads: 8
              ssdPaths: [/mnt/ssd0, /mnt/ssd1]
              useSavedIndexIfExists: Yes
            search:
              topKNprobe:
                10: [8, 16]
                100: [32]
              searchThreads: [1, 4]
            """;

    @TempDir
    Path dir;

    @Test
    void defaultConfigFallsBackToDefaultYml() throws FileNotFoundException {
        IvfConfig config = IvfConfig.getDefaultConfig("no-such-dataset");
        assertEquals("no-such-dataset", config.dataset);
        assertEquals(List.of(1024), config.construction.nlist);
        assertEquals(List.of(1.0f), config.construction.storageFactor);
        assertEquals(0, config.construction.buildThreads);
        assertEquals(List.of(Path.of("ivf_cache")), config.ssdPaths());
        assertFalse(config.construction.useSavedIndexIfExists);
        assertEquals(Map.of(10, List.of(8, 16, 32, 64), 100, List.of(16, 64)), config.search.topKNprobe);
        assertEquals(List.of(1, 8), config.search.searchThreads);
    }

    @Test
    void shippedConfigsAreValid() throws IOException {
        for (String dir : List.of(IvfConfig.DEFAULT_DIRECTORY, "jvector-fastervector-bench/yaml-configs/ivf-experiments/")) {
            try (var files = Files.list(Path.of(dir))) {
                for (Path file : files.filter(f -> f.toString().endsWith(".yml")).toList()) {
                    assertDoesNotThrow(() -> IvfConfig.getConfig(file.toString()), file.toString());
                }
            }
        }
    }

    // Matched by exact file name, as on Linux: a name differing only in case falls back to default.yml there
    @Test
    void everyRegressionDatasetHasMatchingIvfAndGraphConfigs() throws IOException {
        List<String> ivfFiles = fileNames(IvfConfig.DEFAULT_DIRECTORY);
        List<String> graphFiles = fileNames("jvector-examples/yaml-configs/index-parameters/");
        for (String dataset : DatasetCollection.load().getSection("regression-tests")) {
            assertTrue(ivfFiles.contains(dataset + ".yml"), "no IVF config for " + dataset);
            assertTrue(graphFiles.contains(dataset + ".yml"), "no graph config for " + dataset);
            IvfConfig config = IvfConfig.getDefaultConfig(dataset);
            assertEquals(dataset, config.dataset);
            // One topK-10 point on all CPUs, like the graph's topKOverquery 10: [1.0]
            assertEquals(Set.of(10), config.search.topKNprobe.keySet(), dataset);
            assertEquals(1, config.searchPoints().size(), dataset);
            assertEquals(List.of(0), config.search.searchThreads, dataset);
            assertEquals(List.of(1.0f), config.construction.storageFactor, dataset);
        }
    }

    private static List<String> fileNames(String dir) throws IOException {
        try (var files = Files.list(Path.of(dir))) {
            return files.map(f -> f.getFileName().toString()).toList();
        }
    }

    @Test
    void parsesAllFields() throws IOException {
        IvfConfig config = IvfConfig.getConfig(write(VALID).toString());
        assertEquals("cap-1M", config.dataset);
        assertEquals(List.of(256, 1024), config.construction.nlist);
        assertEquals(List.of(1.0f, 2.0f), config.construction.storageFactor);
        assertEquals(8, config.construction.buildThreads);
        assertEquals(List.of(Path.of("/mnt/ssd0"), Path.of("/mnt/ssd1")), config.ssdPaths());
        assertTrue(config.construction.useSavedIndexIfExists);
    }

    @Test
    void constructionPointsCoverEveryCombination() throws IOException {
        IvfConfig config = IvfConfig.getConfig(write(VALID).toString());
        assertEquals(List.of(new ConstructionPoint(256, 1.0f), new ConstructionPoint(256, 2.0f),
                             new ConstructionPoint(1024, 1.0f), new ConstructionPoint(1024, 2.0f)),
                     config.constructionPoints());
    }

    @Test
    void searchPointsCoverEveryCombinationInConfigOrder() throws IOException {
        IvfConfig config = IvfConfig.getConfig(write(VALID).toString());
        assertEquals(List.of(new SearchPoint(10, 8, 1, 1), new SearchPoint(10, 8, 4, 1),
                             new SearchPoint(10, 16, 1, 1), new SearchPoint(10, 16, 4, 1),
                             new SearchPoint(100, 32, 1, 1), new SearchPoint(100, 32, 4, 1)),
                     config.searchPoints());
    }

    @Test
    void subfolderSeparatesConstructionPoints() {
        assertEquals("cap-1M/nlist1024-sf1.0", new ConstructionPoint(1024, 1.0f).subfolder("cap-1M"));
        assertNotEquals(new ConstructionPoint(1024, 1.0f).subfolder("cap-1M"),
                        new ConstructionPoint(1024, 2.0f).subfolder("cap-1M"));
    }

    @Test
    void missingFileThrows() {
        assertThrows(FileNotFoundException.class, () -> IvfConfig.getConfig(dir.resolve("absent.yml").toString()));
    }

    @Test
    void rejectsMissingOrWrongSchemaVersion() throws IOException {
        assertInvalid(VALID.replace("yamlSchemaVersion: 1\n", ""), "yamlSchemaVersion");
        assertInvalid(VALID.replace("yamlSchemaVersion: 1", "yamlSchemaVersion: 2"), "yamlSchemaVersion");
    }

    @Test
    void rejectsUnknownFields() throws IOException {
        assertInvalid(VALID.replace("buildThreads: 8", "buildThread: 8"), "buildThread");
    }

    @Test
    void rejectsBadValues() throws IOException {
        assertInvalid(VALID.replace("nlist: [256, 1024]", "nlist: []"), "construction.nlist");
        assertInvalid(VALID.replace("nlist: [256, 1024]", "nlist: [0]"), "construction.nlist");
        assertInvalid(VALID.replace("storageFactor: [1.0, 2.0]", "storageFactor: [-1.0]"), "storageFactor");
        assertInvalid(VALID.replace("buildThreads: 8", "buildThreads: -1"), "buildThreads");
        assertInvalid(VALID.replace("ssdPaths: [/mnt/ssd0, /mnt/ssd1]", "ssdPaths: []"), "ssdPaths");
        assertInvalid(VALID.replace("10: [8, 16]", "10: [0]"), "nprobe");
        assertInvalid(VALID.replace("10: [8, 16]", "10: []"), "topKNprobe[10]");
        assertInvalid(VALID.replace("searchThreads: [1, 4]", "searchThreads: [-1]"), "searchThreads");
        assertInvalid(VALID.replace("searchThreads: [1, 4]", "searchThreads: [1, 4]\n  queryThreads: [-1]"), "queryThreads");
        assertInvalid(VALID.replace("searchThreads: [1, 4]", "searchThreads: [1, 4]\n  queryThreads: []"), "queryThreads");
    }

    @Test
    void queryThreadsDefaultToOneAndMultiplyTheSearchPoints() throws IOException {
        assertEquals(List.of(1), IvfConfig.getConfig(write(VALID).toString()).search.queryThreads);
        IvfConfig config = IvfConfig.getConfig(
                write(VALID.replace("searchThreads: [1, 4]", "searchThreads: [4]\n  queryThreads: [2, 0]")).toString());
        int cpus = Runtime.getRuntime().availableProcessors();
        assertEquals(List.of(new SearchPoint(10, 8, 4, 2), new SearchPoint(10, 8, 4, cpus)),
                     config.searchPoints().subList(0, 2));
        assertEquals(6, config.searchPoints().size());
    }

    @Test
    void zeroSearchThreadsMeansAllCpus() throws IOException {
        IvfConfig config = IvfConfig.getConfig(write(VALID.replace("searchThreads: [1, 4]", "searchThreads: [0]")).toString());
        assertEquals(List.of(0), config.search.searchThreads);
        int cpus = Runtime.getRuntime().availableProcessors();
        assertTrue(config.searchPoints().stream().allMatch(p -> p.searchThreads() == cpus));
    }

    @Test
    void rejectsMissingSections() throws IOException {
        assertInvalid(VALID.substring(0, VALID.indexOf("search:")), "search");
    }

    @Test
    void machineSettingsCanBeOverridden() throws IOException {
        IvfConfig config = IvfConfig.getConfig(write(VALID).toString());
        var props = new java.util.Properties();
        props.setProperty("ivf.ssdPaths", " /nvme/a , /nvme/b ");
        props.setProperty("ivf.buildThreads", "48");
        props.setProperty("ivf.searchThreads", "1,16, 48");
        props.setProperty("ivf.queryThreads", "1,4");
        config.applyOverrides(props);
        assertEquals(List.of(Path.of("/nvme/a"), Path.of("/nvme/b")), config.ssdPaths());
        assertEquals(48, config.construction.buildThreads);
        assertEquals(List.of(1, 16, 48), config.search.searchThreads);
        assertEquals(List.of(1, 4), config.search.queryThreads);
        // Dataset settings are untouched
        assertEquals(List.of(256, 1024), config.construction.nlist);
    }

    @Test
    void blankOverridesAreIgnoredAndBadOnesRejected() throws IOException {
        IvfConfig config = IvfConfig.getConfig(write(VALID).toString());
        var blank = new java.util.Properties();
        blank.setProperty("ivf.searchThreads", " ");
        config.applyOverrides(blank);
        assertEquals(List.of(1, 4), config.search.searchThreads);

        var bad = new java.util.Properties();
        bad.setProperty("ivf.buildThreads", "lots");
        var e = assertThrows(IllegalArgumentException.class, () -> config.applyOverrides(bad));
        assertTrue(e.getMessage().contains("ivf.buildThreads"), e.getMessage());
    }

    @Test
    void overridesApplyWhenLoadingAndAreValidated() throws IOException {
        Path file = write(VALID);
        System.setProperty("ivf.searchThreads", "-1");
        try {
            assertThrows(IllegalArgumentException.class, () -> IvfConfig.getConfig(file.toString()));
            System.setProperty("ivf.searchThreads", "32");
            assertEquals(List.of(32), IvfConfig.getConfig(file.toString()).search.searchThreads);
        } finally {
            System.clearProperty("ivf.searchThreads");
        }
    }

    private void assertInvalid(String yaml, String expectedInMessage) throws IOException {
        Path file = write(yaml);
        var e = assertThrows(IllegalArgumentException.class, () -> IvfConfig.getConfig(file.toString()));
        assertTrue(e.getMessage().contains(expectedInMessage),
                   "message should mention " + expectedInMessage + ": " + e.getMessage());
    }

    private Path write(String yaml) throws IOException {
        Path file = Files.createTempFile(dir, "ivf", ".yml");
        Files.writeString(file, yaml);
        return file;
    }
}

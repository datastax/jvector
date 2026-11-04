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
        assertEquals(List.of(new SearchPoint(10, 8, 1), new SearchPoint(10, 8, 4),
                             new SearchPoint(10, 16, 1), new SearchPoint(10, 16, 4),
                             new SearchPoint(100, 32, 1), new SearchPoint(100, 32, 4)),
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
        assertInvalid(VALID.replace("searchThreads: [1, 4]", "searchThreads: [0]"), "searchThreads");
    }

    @Test
    void rejectsMissingSections() throws IOException {
        assertInvalid(VALID.substring(0, VALID.indexOf("search:")), "search");
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

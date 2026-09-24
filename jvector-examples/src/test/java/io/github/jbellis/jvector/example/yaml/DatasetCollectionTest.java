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

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/// Tests for {@link DatasetCollection} handling of plain and structured dataset entries.
public class DatasetCollectionTest {

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test
    public void mixedEntriesCanonicalizeToSugaredSpecs() throws IOException {
        Path file = tempFolder.newFile("datasets.yml").toPath();
        Files.writeString(file,
                "small:\n" +
                "  - ada002-100k\n" +
                "  - cohere-english-v3-100k(mmap)\n" +
                "  - name: gecko-100k\n" +
                "    profile: default\n" +
                "    wrappers: [ mmap ]\n" +
                "  - name: e5-small-v2-100k\n" +
                "    profile: fast\n" +
                "  - name: e5-base-v2-100k\n" +
                "empty:\n" +
                "large:\n" +
                "  - name: cap-6M\n" +
                "    wrappers: mmap, memory\n" +
                "  - name: cohere-english-v3-10M\n" +
                "    wrappers:\n" +
                "      - mmap\n" +
                "      - lru: { grain: 4096, capacityMb: 512 }\n" +
                "  - dpr-gemma-10m(mmap,lru[grain=2048])\n");

        var collection = DatasetCollection.load(file.toString());
        assertEquals(List.of("ada002-100k", "cohere-english-v3-100k(mmap)", "gecko-100k(mmap)",
                        "e5-small-v2-100k:fast", "e5-base-v2-100k"),
                collection.getSection("small"));
        assertTrue(collection.getSection("empty").isEmpty());
        assertEquals(List.of("cap-6M(mmap,memory)",
                        "cohere-english-v3-10M(mmap,lru[grain=4096,capacityMb=512])",
                        "dpr-gemma-10m(mmap,lru[grain=2048])"),
                collection.getSection("large"));
        assertEquals(8, collection.getAll().size());
        assertTrue(collection.datasetNames.containsKey("empty"));
    }

    @Test
    public void invalidEntriesNameTheirSection() throws IOException {
        Path file = tempFolder.newFile("bad.yml").toPath();
        Files.writeString(file,
                "ok:\n" +
                "  - ada002-100k\n" +
                "broken:\n" +
                "  - profile: fast\n");
        var e = assertThrows(IllegalArgumentException.class, () -> DatasetCollection.load(file.toString()));
        assertTrue(e.getMessage().contains("'broken'"), e.getMessage());
    }

    @Test
    public void defaultCollectionStillLoads() throws IOException {
        var collection = DatasetCollection.load();
        assertFalse(collection.getAll().isEmpty());
        assertTrue(collection.getSection("regression-tests").contains("cap-1M"));
    }
}

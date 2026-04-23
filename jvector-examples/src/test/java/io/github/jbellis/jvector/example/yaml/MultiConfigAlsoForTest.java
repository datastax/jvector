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

import org.junit.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/// Verifies that a yaml index-parameters config declaring an {@code also_for} list
/// has that field populated correctly. Full directory-scan resolution via
/// {@link MultiConfig#getDefaultConfig(String)} depends on the repo-relative
/// index-parameters directory and is covered by running BenchYAML end-to-end.
public class MultiConfigAlsoForTest {

    @Test
    public void alsoForFieldParsesFromYaml() throws Exception {
        String yaml = "yamlSchemaVersion: 1\n"
                + "onDiskIndexVersion: 6\n"
                + "dataset: some-dataset\n"
                + "also_for:\n"
                + "  - \"other-dataset:variant\"\n"
                + "  - plain-alias\n";

        Path tmp = Files.createTempFile("multiconfig-alsofor-", ".yml");
        try {
            Files.writeString(tmp, yaml);
            MultiConfig cfg = MultiConfig.getConfig(tmp.toFile());
            assertNotNull(cfg.also_for, "also_for should be parsed");
            assertEquals(2, cfg.also_for.size());
            assertEquals("other-dataset:variant", cfg.also_for.get(0));
            assertEquals("plain-alias", cfg.also_for.get(1));
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void globToPattern_starMatchesAcrossSegments() {
        var p = MultiConfig.globToPattern("sift1m:label_*");
        assertTrue(p.matcher("sift1m:label_00").matches());
        assertTrue(p.matcher("sift1m:label_11").matches());
        assertTrue(p.matcher("sift1m:label_").matches()); // * allows empty
        assertFalse(p.matcher("sift1m-label_00").matches());
        assertFalse(p.matcher("other:label_00").matches());
    }

    @Test
    public void globToPattern_questionMarkMatchesSingleChar() {
        var p = MultiConfig.globToPattern("ds?-v1");
        assertTrue(p.matcher("dsA-v1").matches());
        assertTrue(p.matcher("ds1-v1").matches());
        assertFalse(p.matcher("ds-v1").matches()); // ? requires one char
        assertFalse(p.matcher("dsAB-v1").matches());
    }

    @Test
    public void globToPattern_regexMetacharsAreEscaped() {
        var p = MultiConfig.globToPattern("a.b+c(d)");
        assertTrue(p.matcher("a.b+c(d)").matches());
        assertFalse(p.matcher("aXb+c(d)").matches()); // '.' must be literal, not wildcard
    }

    @Test
    public void alsoForIsNullWhenAbsent() throws Exception {
        String yaml = "yamlSchemaVersion: 1\n"
                + "onDiskIndexVersion: 6\n"
                + "dataset: some-dataset\n";

        Path tmp = Files.createTempFile("multiconfig-noalso-", ".yml");
        try {
            Files.writeString(tmp, yaml);
            MultiConfig cfg = MultiConfig.getConfig(tmp.toFile());
            assertNull(cfg.also_for);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}

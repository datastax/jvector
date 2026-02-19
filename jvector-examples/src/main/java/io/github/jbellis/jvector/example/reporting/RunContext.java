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

package io.github.jbellis.jvector.example.reporting;

import java.nio.file.Path;
import java.time.Instant;
import java.util.Objects;
import java.util.UUID;

/**
 * Immutable run-scoped context for logging artifacts.
 *
 * - runId: human-friendly directory name (rendered from logging.runId template; e.g., "20260209-222308Z" or "my-exp_20260209-222308Z")
 * - runUuid: collision-proof identity for the run
 * - runDir: logDir/runId
 * - schemaVersion: version of the logging artifact schema
 * - systemId: stable join key derived from sys_info (hash of OS/JVM/SIMD/threads/etc.)
 */
public final class RunContext {
    public static final int SCHEMA_VERSION = 1;

    private final String runId;
    private final UUID runUuid;
    private final Instant createdAt;
    private final Path runDir;

    // NEW: join key for experiments.csv
    private final String systemId;

    public RunContext(String runId, UUID runUuid, Instant createdAt, Path runDir, String systemId) {
        this.runId = Objects.requireNonNull(runId, "runId");
        this.runUuid = Objects.requireNonNull(runUuid, "runUuid");
        this.createdAt = Objects.requireNonNull(createdAt, "createdAt");
        this.runDir = Objects.requireNonNull(runDir, "runDir");
        this.systemId = Objects.requireNonNull(systemId, "systemId");
    }

    public String runId() { return runId; }
    public UUID runUuid() { return runUuid; }
    public Instant createdAt() { return createdAt; }
    public Path runDir() { return runDir; }
    public int schemaVersion() { return SCHEMA_VERSION; }

    public String systemId() { return systemId; }
}

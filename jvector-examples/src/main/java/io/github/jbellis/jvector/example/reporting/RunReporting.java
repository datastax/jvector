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

import io.github.jbellis.jvector.example.yaml.RunConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

/**
 * Bootstraps a benchmark run directory and writes sys_info.json.
 *
 * This class creates a run_id/run_uuid, selects the logging directory from run.yml, captures basic
 * environment metadata (OS/JVM/CPU/SIMD/threads/memory), computes a stable system_id, and returns a
 * {@link RunContext} for downstream writers (dataset_info.csv, experiments.csv).
 */
public final class RunReporting {
    private static final DateTimeFormatter RUN_ID_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");

    private final RunContext run;
    private final String jvectorRef;

    private RunReporting(RunContext run, String jvectorRef) {
        this.run = run;
        this.jvectorRef = jvectorRef;
    }

    public RunContext run() {
        return run;
    }

    public String jvectorRef() {
        return jvectorRef;
    }

    public static RunReporting open(RunConfig runCfg) throws IOException {
        String logDir = (runCfg != null && runCfg.logging != null && runCfg.logging.dir != null && !runCfg.logging.dir.isBlank())
                ? runCfg.logging.dir
                : "logging";

        String runId = (runCfg != null && runCfg.logging != null && runCfg.logging.runId != null && !runCfg.logging.runId.isBlank())
                ? runCfg.logging.runId
                : LocalDateTime.now().format(RUN_ID_FMT);

        String jvectorRef = (runCfg != null && runCfg.logging != null && runCfg.logging.jvectorRef != null)
                ? runCfg.logging.jvectorRef
                : "";

        UUID runUuid = UUID.randomUUID();
        Instant createdAt = Instant.now();
        Path runDir = Paths.get(logDir).resolve(runId);

        Integer buildThreads = detectBuildThreads();
        Integer queryThreads = java.util.concurrent.ForkJoinPool.getCommonPoolParallelism();

        // Threads: fill later if you want; safe to start null
        String systemId = SysInfoWriter.writeSysInfo(
                runDir,
                RunContext.SCHEMA_VERSION,
                runId,
                runUuid,
                createdAt,
                jvectorRef,
                buildThreads,
                queryThreads
        );

        RunContext run = new RunContext(runId, runUuid, createdAt, runDir, systemId);
        return new RunReporting(run, jvectorRef);
    }

    private static Integer detectBuildThreads() {
        try {
            Object exec = io.github.jbellis.jvector.util.PhysicalCoreExecutor.pool();

            if (exec instanceof java.util.concurrent.ForkJoinPool) {
                return ((java.util.concurrent.ForkJoinPool) exec).getParallelism();
            }
            if (exec instanceof java.util.concurrent.ThreadPoolExecutor) {
                return ((java.util.concurrent.ThreadPoolExecutor) exec).getMaximumPoolSize();
            }

            return Runtime.getRuntime().availableProcessors();
        } catch (Throwable t) {
            return Runtime.getRuntime().availableProcessors();
        }
    }
}

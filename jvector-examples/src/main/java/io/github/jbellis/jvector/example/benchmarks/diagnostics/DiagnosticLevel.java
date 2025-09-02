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

package io.github.jbellis.jvector.example.benchmarks.diagnostics;

/**
 * Defines the level of diagnostic information to collect and log during benchmarks.
 */
public enum DiagnosticLevel {
    /** No diagnostic information collected */
    NONE,

    /** Basic system stats (GC, memory, CPU) before and after each run */
    BASIC,

    /** Detailed timing analysis and per-run statistics */
    DETAILED,

    /** Verbose logging including individual query times and thread statistics */
    VERBOSE
}

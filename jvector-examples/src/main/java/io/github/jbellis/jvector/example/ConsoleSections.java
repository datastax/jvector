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

package io.github.jbellis.jvector.example;

/** Console structure for BenchYAML runs: a banner per experiment and a header per stage. */
public final class ConsoleSections {
    private static final String RULE = "=".repeat(80);

    private ConsoleSections() {}

    /** The experiment's pipeline, framed so it stands out at the top of its output. */
    public static void pipeline(String dataset, String pipeline) {
        System.out.println();
        System.out.println(RULE);
        System.out.println(" PIPELINE  " + dataset + ": " + pipeline);
        System.out.println(RULE);
    }

    /** Starts a stage of the run, for example Dataset, Construction or Search. */
    public static void stage(String title) {
        System.out.println();
        System.out.println("== " + title + " ==");
    }
}

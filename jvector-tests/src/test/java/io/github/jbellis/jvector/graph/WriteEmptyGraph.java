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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Run on PR #685 branch to write an empty graph to disk via OnDiskGraphIndexWriter.
 * Then switch to main and run LoadEmptyGraph to check backward compatibility.
 *
 * Uses the production write path (OnDiskGraphIndex.write → OnDiskGraphIndexWriter →
 * AbstractGraphIndexWriter.writeFooter), which is what Cassandra calls in production.
 * PR #685 patches writeFooter/writeHeader to write ENTRY_NODE_ABSENT (-1) instead of
 * NPE-ing when view.entryNode() is null for an empty graph.
 *
 * Usage (on PR #685 branch):
 *   mvn test -pl jvector-tests -Dtest=WriteEmptyGraph -DfailIfNoTests=false
 */
public class WriteEmptyGraph {

    static final Path OUTPUT = Path.of("/tmp/jvector_compat_test.bin");
    static final int DIMENSION = 64;

    public static void main(String[] args) throws Exception {
//        var ravv = MockVectorValues.empty(DIMENSION);
//        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.2f, true);
//        builder.cleanup();
//        var graph = builder.getGraph();
//
//        System.out.println("Before write (on PR #685 branch):");
//        System.out.println("  size(0)    = " + graph.size(0));
//        System.out.println("  maxLevel   = " + graph.getMaxLevel());
//        try (var view = graph.getView()) {
//            System.out.println("  entryNode  = " + view.entryNode());
//        }
//
//        // Write via OnDiskGraphIndexWriter (AbstractGraphIndexWriter.writeFooter path).
//        // On main (before the PR) this throws NullPointerException because view.entryNode() is null.
//        // PR #685 patches this to write -1 (ENTRY_NODE_ABSENT) instead.
//        OnDiskGraphIndex.write(graph, ravv, OUTPUT);
//
//        System.out.println("Written to:  " + OUTPUT.toAbsolutePath());
//        System.out.println("File size:   " + Files.size(OUTPUT) + " bytes");
//        System.out.println("Now switch to main and run LoadEmptyGraph.");
    }

    // Allow running as a JUnit test so `mvn test -Dtest=WriteEmptyGraph` works
    @org.junit.Test
    public void main() throws Exception {
        main(new String[0]);
    }
}

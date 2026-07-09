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

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.nio.file.Path;

/**
 * Run on MAIN branch after running WriteEmptyGraph on the PR #685 branch.
 *
 * Tests backward compatibility: what does old code (main) do when it tries to
 * load an OnDisk-format file produced by new code (PR #685) for an empty graph?
 *
 * PR #685 writes entryNode = -1 (ENTRY_NODE_ABSENT) via AbstractGraphIndexWriter.
 * Old code reads that -1 as a real ordinal, constructing NodeAtLevel(0, -1) silently —
 * no exception, just corrupt state. Any search on the loaded graph will misbehave.
 *
 * Usage (on main branch):
 *   mvn test -pl jvector-tests -Dtest=LoadEmptyGraph -DfailIfNoTests=false
 */
public class LoadEmptyGraph {

    static final Path INPUT = Path.of("/tmp/jvector_compat_test.bin");

    public static void main(String[] args) throws Exception {
        if (!INPUT.toFile().exists()) {
            System.out.println("ERROR: " + INPUT + " not found.");
            System.out.println("Run WriteEmptyGraph on the PR #685 branch first.");
            return;
        }

        System.out.println("Loading from: " + INPUT.toAbsolutePath());

        OnDiskGraphIndex graph;
        try (var readerSupplier = new SimpleMappedReader.Supplier(INPUT)) {
            graph = OnDiskGraphIndex.load(readerSupplier);
        } catch (Exception e) {
            System.out.println("Load threw: " + e.getClass().getName() + ": " + e.getMessage());
            e.printStackTrace();
            return;
        }

        System.out.println("Load succeeded (no exception thrown)");
        System.out.println("  size(0)    = " + graph.size(0));
        System.out.println("  maxLevel   = " + graph.getMaxLevel());

        try (var view = graph.getView()) {
            var entry = view.entryNode();
            System.out.println("  entryNode  = " + entry);
            System.out.println();

            if (entry != null && entry.node == -1) {
                System.out.println("RESULT: SILENT CORRUPTION");
                System.out.println("  entryNode.node = -1 was accepted without error.");
                System.out.println("  Any search on this graph will use ordinal -1 as entry, causing incorrect results or a crash.");
                System.out.println("  This confirms a format version bump is needed before merging PR #685.");
            } else if (entry == null) {
                System.out.println("RESULT: Clean empty graph (unexpected for old code).");
            } else {
                System.out.println("RESULT: Unexpected entry node state: " + entry);
            }
        }
        ReaderSupplier graphHandle = new SimpleMappedReader.Supplier(INPUT);
        var rawGraph = OnDiskGraphIndex.load(graphHandle, 0, false);
        System.out.println("rawGraph.size(0) = " + rawGraph.size(0));

        GraphSearcher searcher = new GraphSearcher(rawGraph);
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        VectorFloat<?> queryVector = vts.createFloatVector(rawGraph.getDimension());
        VectorSimilarityFunction vsf = VectorSimilarityFunction.COSINE;
        VectorFloat<?>[] values = new VectorFloat<?>[0];
        var ravv = MockVectorValues.fromValues(values);
        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(queryVector, vsf, rawGraph.getView());
        var result = searcher.search(ssp, 10, Bits.ALL);
        System.out.println("result.getNodes().length = " + result.getNodes().length);
    }

    @org.junit.Test
    public void main() throws Exception {
        main(new String[0]);
    }
}

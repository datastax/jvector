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

/**
 * The previous name of {@link GraphIndex}, kept so that code written against JVector 4.0.x keeps
 * compiling for one more release.
 * <p>
 * Everything that was declared here now lives on {@link GraphIndex}, and this interface only extends
 * it. Nested types and constants are inherited, so references such as
 * {@code ImmutableGraphIndex.View}, {@code ImmutableGraphIndex.ScoringView},
 * {@code ImmutableGraphIndex.NodeAtLevel} and {@code ImmutableGraphIndex.ENTRY_NODE_ABSENT} still
 * resolve (to the same types and values as their {@code GraphIndex} spellings). The graphs JVector
 * creates ({@link OnHeapGraphIndex}, {@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex})
 * implement it, and the {@link GraphIndexBuilder} methods that returned it before still do.
 * <p>
 * Source compatibility only: classes compiled against 4.0.x that use the nested types must be
 * recompiled, since those types are now members of {@code GraphIndex}.
 *
 * @deprecated Use {@link GraphIndex}. This interface will be removed in the release after the one
 *             that introduced {@code GraphIndex}.
 */
@Deprecated(forRemoval = true)
public interface ImmutableGraphIndex extends GraphIndex {

    /**
     * @deprecated Use {@link GraphIndex#prettyPrint(GraphIndex)}.
     */
    @Deprecated(forRemoval = true)
    static String prettyPrint(ImmutableGraphIndex graph) {
        return GraphIndex.prettyPrint(graph);
    }
}

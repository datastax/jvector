/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.disk.IndexWriter;

import java.io.FileNotFoundException;

/**
 * A graph index that can be written to disk.
 * <p>
 * Both {@code OnHeapGraphIndex} (in-memory, potentially still under construction) and
 * {@code OnDiskGraphIndex} (already on disk) implement this interface. Writing is supported
 * via {@link #writer(java.nio.file.Path)} (parallel/random-access) and
 * {@link #writer(IndexWriter)} (sequential, e.g. for Cassandra/Lucene integration).
 * <p>
 * {@link MutableGraphIndex} extends this interface, so an on-heap graph is persistable
 * at any point during construction.
 */
public interface PersistableGraphIndex extends GraphIndex {

    /**
     * Returns a {@link WriteBuilder} that writes this graph sequentially to {@code out}.
     * <p>
     * Sequential writing is suitable for cloud object storage and frameworks such as
     * Lucene or Cassandra that require or prefer sequential I/O. The header is written
     * as a footer; the caller is responsible for flushing {@code out}.
     */
    default WriteBuilder writer(IndexWriter out) throws FileNotFoundException {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support sequential writing");
    }

    /** @deprecated use {@link GraphIndex#prettyPrint(GraphIndex)} */
    @Deprecated
    static String prettyPrint(PersistableGraphIndex graph) {
        return GraphIndex.prettyPrint(graph);
    }
}
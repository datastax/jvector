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

package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.annotations.Experimental;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

/**
 * Where a host lets jvector put index bytes. Stateless configuration: a host builds one, holds
 * it, and jvector calls {@link #open()} once per write operation. Nothing here or in anything it
 * vends names a path, file, channel or host stream type; the host keeps those.
 * <p>
 * The contract in one picture:
 * <pre>
 *   IndexDestination                    stateless "where", built once by the host
 *      open() ---------> OutputSession               one write operation; the transactional unit
 *                          reserve(artifact) --> OutputReservation   one artifact's output
 *                                                  stream()    IndexWriter, append-only
 *                                                  complete() / close()
 *                          commit() / close()
 * </pre>
 * jvector writes every artifact strictly sequentially through its reservation's
 * {@link OutputReservation#stream() stream}. Two levels of "done" mirror what every host already
 * has: each artifact is {@link OutputReservation#complete completed} (where a host writes its
 * per-file footer and checksum), and the session is {@link OutputSession#commit committed} once
 * as a set (where a host publishes). A close without the matching commit is an abort at either
 * level, so try-with-resources is the whole error-handling story.
 * <p>
 * The file-backed factories below are the normative reference implementation and the fast path
 * for standalone use; every other host implements this interface over its own storage.
 */
@Experimental
@FunctionalInterface
public interface IndexDestination {

    /**
     * Begins one write operation. The returned session is the transactional unit: it is
     * committed at most once and closed exactly once, normally with try-with-resources.
     */
    OutputSession open() throws IOException;

    /**
     * Standalone, jvector-owned file. The session writes a sibling temporary file and renames it
     * over {@code path} on commit; an abort deletes the temporary file and leaves any pre-existing
     * {@code path} untouched. Places the {@link OutputArtifact#GRAPH} artifact only.
     */
    static IndexDestination toFile(Path path) {
        return FileIndexDestination.standalone(Collections.singletonMap(OutputArtifact.GRAPH, path));
    }

    /**
     * Standalone files, one per artifact (for example the graph plus a compressed-vectors
     * sidecar), each with the same temporary-file-and-rename lifecycle applied at commit.
     */
    static IndexDestination toFiles(Map<OutputArtifact, Path> paths) {
        return FileIndexDestination.standalone(paths);
    }

    /**
     * A region inside a caller-owned file, starting at {@code offset}. The file is never
     * truncated, deleted or renamed, and nothing before {@code offset} is written; the caller
     * owns whatever precedes and follows the region (its own header and footer). Intended for
     * tests and simple embedders; real hosts implement the interface directly so that
     * {@link OutputReservation#complete} can write their footer.
     */
    static IndexDestination inFile(Path path, long offset) {
        return FileIndexDestination.region(path, offset);
    }
}

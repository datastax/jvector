package io.github.jbellis.jvector.graph.disk.representations;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.NodeArray;

import java.io.IOException;

public interface InGraphRepresentationWriter {
    /**
     * The number of bytes that will be written by the {@link #writeFor}
     * @return
     */
    long representationFootprint();

    /**
     * Writes the vector representation for the given node and its neighborhood. Depending on the persistence type,
     * it will inline, fuse, or be a no-op (for separated features).
     * @param nodeId the
     * @param nodes
     * @param out
     * @throws IOException
     */
    void writeFor(long nodeId, NodeArray nodes, IndexWriter out) throws IOException;
}

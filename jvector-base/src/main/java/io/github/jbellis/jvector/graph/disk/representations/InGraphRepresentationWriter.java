package io.github.jbellis.jvector.graph.disk.representations;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.NodeArray;

import java.io.IOException;

public interface InGraphRepresentationWriter {
    void writeFor(int nodeId, NodeArray nodes, IndexWriter out) throws IOException;
}

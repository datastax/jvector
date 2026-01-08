package io.github.jbellis.jvector.graph.disk.representations;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.graph.NodeArray;

import java.io.IOException;

public class EmptyInGraphRepresentationWriter implements InGraphRepresentationWriter {
    @Override
    public void writeFor(int nodeId, NodeArray nodes, IndexWriter out) throws IOException {
        // This is a no-op, meant to be used with separable representations
    }
}

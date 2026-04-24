package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public interface InMemoryGraphIndex extends GraphIndex , AutoCloseable, Accountable {

    public long addGraphNode(int node, VectorFloat<?> vector);

    public long addGraphNodes(VectorFloat<?> vector);

}

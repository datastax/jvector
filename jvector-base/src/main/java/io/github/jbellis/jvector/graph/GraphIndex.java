package io.github.jbellis.jvector.graph;

import java.nio.file.Path;

public interface GraphIndex {

    default InMemoryGraphIndexBuilder builder() {
        return new InMemoryGraphIndexBuilder();
    }

    PersistedGraphIndex persist(Path path);

    InMemoryGraphIndex load(GraphIndex index);

    InMemoryGraphIndex load(Path path);
}

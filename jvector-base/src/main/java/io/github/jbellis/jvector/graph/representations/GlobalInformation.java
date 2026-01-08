package io.github.jbellis.jvector.graph.representations;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorRepresentation;

import java.io.IOException;

public interface GlobalInformation<Vec extends VectorRepresentation> extends Accountable {
    void write(IndexWriter out) throws IOException;

    void loadFrom(RandomAccessReader reader) throws IOException;
}

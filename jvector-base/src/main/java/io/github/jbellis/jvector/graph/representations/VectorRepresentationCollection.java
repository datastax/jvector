package io.github.jbellis.jvector.graph.representations;

import io.github.jbellis.jvector.vector.VectorRepresentation;


public interface VectorRepresentationCollection<Vec extends VectorRepresentation> {
    /**
     * Return the number of vector values.
     * <p>
     * All copies of a given RAVV should have the same size.  Typically this is achieved by either
     * (1) implementing a threadsafe, un-shared RAVV, where `copy` returns `this`, or
     * (2) implementing a fixed-size RAVV.
     */
    int size();

    /** Return the dimension of the returned vector values */
    int dimension();

    GlobalInformation<Vec> getGlobalInformation();

    PersistenceType getPersistenceType();
}

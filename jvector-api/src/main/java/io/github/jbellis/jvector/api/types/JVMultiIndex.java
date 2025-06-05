package io.github.jbellis.jvector.api.types;

/**
 This is the core index type for JVector.
 It is indirect here to support modularity and segmented testing of
 the API itself. */
public interface JVMultiIndex {
  JVIndexer getIndexer(int index);
}

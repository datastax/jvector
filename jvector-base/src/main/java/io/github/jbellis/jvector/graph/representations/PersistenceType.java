package io.github.jbellis.jvector.graph.representations;

public enum PersistenceType {
    INLINE,
    FUSED,
    SEPARATE,
    NONE // TODO is there is no intention to persist, we may use a none type
}

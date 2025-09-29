package io.github.jbellis.jvector.graph.disk.feature;

public abstract class AbstractFeature implements Feature {
    public int compareTo(Feature f) {
        if (this.isFused() != f.isFused()) {
            return Boolean.compare(this.isFused(), f.isFused());
        }
        return this.id().compareTo(f.id());
    }
}

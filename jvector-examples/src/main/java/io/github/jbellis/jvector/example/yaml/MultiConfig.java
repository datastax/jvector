package io.github.jbellis.jvector.example.yaml;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;

public class MultiConfig {
    private int version;
    public String dataset;
    public ConstructionParameters construction;
    public SearchParameters search;

    public int getVersion() {
        return version;
    }

    public void setVersion(int version) {
        if (version != OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("Invalid version: " + version);
        }
        this.version = version;
    }
}
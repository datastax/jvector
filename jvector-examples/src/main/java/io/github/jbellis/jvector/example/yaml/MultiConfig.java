package io.github.jbellis.jvector.example.yaml;

public class MultiConfig {
    private static int CURRENT_VERSION = 4;

    private int version;
    public String dataset;
    public ConstructionParameters construction;
    public SearchParameters search;

    public int getVersion() {
        return version;
    }

    public void setVersion(int version) {
        if (version != CURRENT_VERSION) {
            throw new IllegalArgumentException("Invalid version: " + version);
        }
        this.version = version;
    }
}
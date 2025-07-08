package io.github.jbellis.jvector.example.dynamic;

import io.github.jbellis.jvector.example.util.AbstractDataset;
import io.github.jbellis.jvector.example.util.QueryBundle;

import java.util.List;

public interface DynamicDataset extends AbstractDataset {
    int epochs();

    List<Integer> insertions(int epoch);

    List<Integer> deletions(int epoch);

    QueryBundle getQueryBundle(int epoch);
}

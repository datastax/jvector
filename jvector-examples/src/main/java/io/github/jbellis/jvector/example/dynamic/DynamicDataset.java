package io.github.jbellis.jvector.example.dynamic;

import io.github.jbellis.jvector.example.util.AbstractDataSet;
import io.github.jbellis.jvector.example.util.QueryBundle;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

public interface DynamicDataset extends AbstractDataSet {
    int epochs();

    List<Integer> insertions(int epoch);

    List<Integer> deletions(int epoch);

    QueryBundle getQueryBundle(int epoch);
}

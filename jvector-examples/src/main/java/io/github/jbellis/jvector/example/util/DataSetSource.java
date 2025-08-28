package io.github.jbellis.jvector.example.util;

import java.util.Optional;
import java.util.function.Function;

public interface DataSetSource extends Function<String, Optional<DataSet>> {
  public DataSetSource DEFAULT = new DataSetLoader(DataSetLoader.HDF5Loader, DataSetLoader.FVecsDownloader);

  public default DataSetSource and(DataSetSource... loaders) {
    return new DataSetSource() {
      @Override
      public Optional<DataSet> apply(String name) {
        for (var loader : loaders) {
          var ds = loader.apply(name);
          if (ds.isPresent()) {
            return ds;
          }
        }
        return Optional.empty();
      }
    };
  }
}

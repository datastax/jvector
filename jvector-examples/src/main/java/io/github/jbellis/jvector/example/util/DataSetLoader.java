/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.example.util;

import io.jhdf.api.Dataset;

import java.io.IOException;
import java.util.Optional;
import java.util.function.Function;

public class DataSetLoader implements DataSetSource {

  private final Function<String, Optional<DataSet>>[] loaders;

  public DataSetLoader(DataSetSource... loaders) {
    this.loaders = loaders;
  }

  @Override
  public Optional<DataSet> apply(String name) {
    return Optional.empty();
  }

  public final static DataSetSource FVecsDownloader = new DataSetSource() {
    @Override
    public Optional<DataSet> apply(String name) {
      var mfd = DownloadHelper.maybeDownloadFvecs(name);
      try {
        var ds = mfd.load();
        return Optional.of(ds);
      } catch (IOException e) {
        System.err.println("error while trying to load dataset: " + e + ", this error handling "
                           + "path needs to be updated");
        return Optional.empty();
      }
    }
  };

  public final static DataSetSource HDF5Loader = new DataSetSource() {

    @Override
    public Optional<DataSet> apply(String name) {
      if (name.endsWith(".hdf5")) {
        DownloadHelper.maybeDownloadHdf5(name);
        return Optional.of(Hdf5Loader.load(name));
      }
      return Optional.empty();
    }
  };
}

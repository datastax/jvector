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

import io.github.jbellis.jvector.benchframe.TestDataViewWrapper;
import io.nosqlbench.nbdatatools.api.concurrent.ProgressIndicator;
import io.nosqlbench.vectordata.discovery.TestDataSources;
import io.nosqlbench.vectordata.discovery.TestDataView;
import io.nosqlbench.vectordata.downloader.Catalog;
import io.nosqlbench.vectordata.downloader.DatasetEntry;

import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

public class DataSetLoader implements DataSetSource {

  private final DataSetSource[] loaders;

  public DataSetLoader(DataSetSource... loaders) {
    this.loaders = loaders;
  }

  @Override
  public Optional<DataSet> apply(String name) {
    for (DataSetSource loader : loaders) {
      Optional<DataSet> result = loader.apply(name);
      if (result.isPresent()) {
        return result;
      }
    }
    return Optional.empty();
  }

  @Override
  public String toString() {
    return "DataSetLoader{loaders=" + loaders.length + "}";
  }

  public final static DataSetSource FVecsDownloader = new DataSetSource() {
    @Override
    public Optional<DataSet> apply(String name) {
      Optional<MultiFileDatasource> mfdOpt = DownloadHelper.maybeDownloadFvecs(name);
      if (mfdOpt.isEmpty()) {
        return Optional.empty();
      }

      try {
        var ds = mfdOpt.get().load();
        return Optional.of(ds);
      } catch (IOException e) {
        System.err.println("error while trying to load dataset: " + e + ", this error handling "
                           + "path needs to be updated");
        return Optional.empty();
      }
    }

    @Override
    public String toString() {
      return "FVecsDownloader";
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

    @Override
    public String toString() {
      return "HDF5Loader";
    }
  };

  /**
   * VectorData downloader that loads datasets from the vectordata catalog system.
   * Supports optional additional catalogs via VECTORDATA_CATALOGS environment variable.
   *
   * Environment variable format:
   * VECTORDATA_CATALOGS=~/.config/custom1/catalogs.yaml,~/.config/custom2/catalogs.yaml
   */
  public static final DataSetSource vectorDataDownloader = new DataSetSource() {
    private final Catalog catalog = initializeCatalog();

    private Catalog initializeCatalog() {
      TestDataSources sources = new TestDataSources().configure();

      // Add additional catalogs from environment variable
      String envCatalogs = System.getenv("VECTORDATA_CATALOGS");
      if (envCatalogs != null && !envCatalogs.trim().isEmpty()) {
        String[] catalogPaths = envCatalogs.split(",");
        for (String catalogPath : catalogPaths) {
          String trimmedPath = catalogPath.trim();
          if (!trimmedPath.isEmpty()) {
            System.out.println("Adding optional catalog from VECTORDATA_CATALOGS: " + trimmedPath);
            sources.addOptionalCatalogs(trimmedPath);
          }
        }
      }

      return sources.catalog();
    }

    @Override
    public Optional<DataSet> apply(String name) {
      name = name.contains(":") ? name : name + ":default";

      TestDataView tdv = catalog.profile(name);
      System.out.println("prebuffering dataset '" + name + "' (assumed performance oriented testing)");

      CompletableFuture<Void> statusFuture = tdv.getBaseVectors().orElseThrow().prebuffer();
      if (statusFuture instanceof ProgressIndicator<?>) {
        ((ProgressIndicator<?>) statusFuture).monitorProgress(1000);
      }

      TestDataViewWrapper tdw = new TestDataViewWrapper(tdv);
      System.out.println("Loaded " + tdw.getName() + " from streaming source");
      return Optional.of(tdw);
    }

    @Override
    public String toString() {
      String envCatalogs = System.getenv("VECTORDATA_CATALOGS");
      return "VectorDataDownloader{defaultCatalog=~/.config/vectordata/catalogs.yaml" +
             (envCatalogs != null ? ", additionalCatalogs=" + envCatalogs : "") + "}";
    }
  };

  /**
   * Creates a VectorDataDownloader with a specific catalog path.
   * Use this when you need a custom catalog location programmatically.
   * For most use cases, prefer using the VECTORDATA_CATALOGS environment variable instead.
   *
   * @param catalogPath path to the catalog YAML file (e.g., "~/.config/vectordata/catalogs.yaml")
   * @return a DataSetSource that can load from the specified catalog
   */
  public static DataSetSource createVectorDataDownloader(String catalogPath) {
    Catalog catalog = new TestDataSources()
        .configure()
        .addOptionalCatalogs(catalogPath)
        .catalog();

    return name -> {
      Optional<DatasetEntry> dsentryOption = catalog.matchOne(name);
      if (dsentryOption.isEmpty()) {
        return Optional.empty();
      }

      DatasetEntry dsentry = dsentryOption.get();
      TestDataView tdv = dsentry.select().profile(name);

      System.out.println("prebuffering dataset (assumed performance oriented testing)");
      CompletableFuture<Void> statusFuture = tdv.getBaseVectors().orElseThrow().prebuffer();
      if (statusFuture instanceof ProgressIndicator<?>) {
        ((ProgressIndicator<?>) statusFuture).monitorProgress(1000);
      }

      TestDataViewWrapper tdw = new TestDataViewWrapper(tdv);
      System.out.println("Loaded " + tdw.getName() + " from streaming source");
      return Optional.of(tdw);
    };
  }
}

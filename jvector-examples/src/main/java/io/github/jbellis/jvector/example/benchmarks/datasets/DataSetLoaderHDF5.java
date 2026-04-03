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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import io.jhdf.object.datatype.FloatingPoint;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.IntStream;

/**
 * This dataset loader will get and load hdf5 files from <a href="https://ann-benchmarks.com/">ann-benchmarks</a>.
 *
 * <p>For curated benchmark datasets, properties are provided by
 * {@code dataset_metadata.yml} via {@link DataSetMetadataReader}. If the metadata
 * does not provide a similarity function, an error is thrown.
 */
public class DataSetLoaderHDF5 implements DataSetLoader {
    public static final Path HDF5_DIR = Path.of("hdf5");
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    public static final String HDF5_EXTN = ".hdf5";
    private static final DataSetMetadataReader metadata = DataSetMetadataReader.load();

    @Override
    public String getName() {
        return "HDF5";
    }

    /**
     * {@inheritDoc}
     */
    public Optional<DataSetInfo> loadDataSet(String datasetName) {
        return maybeDownloadHdf5(datasetName).map(path -> {
            var props = getProperties(datasetName);
            props.similarityFunction()
                    .orElseThrow(() -> new IllegalArgumentException(
                            "No similarity function configured in dataset_metadata.yml for HDF5 dataset: " + datasetName));
            return new DataSetInfo(props, () -> readHdf5Data(path, props));
        });
    }

    /// Reads base vectors, query vectors, and ground truth from an HDF5 file
    /// and returns a {@link DataSet} using the configured dataset properties.
    private DataSet readHdf5Data(Path path, DataSetProperties props) {
        VectorFloat<?>[] baseVectors;
        VectorFloat<?>[] queryVectors;
        var gtSets = new ArrayList<List<Integer>>();
        try (HdfFile hdf = new HdfFile(path)) {
            var baseVectorsArray =
                    (float[][]) hdf.getDatasetByPath("train").getData();
            baseVectors = IntStream.range(0, baseVectorsArray.length).parallel().mapToObj(i -> vectorTypeSupport.createFloatVector(baseVectorsArray[i])).toArray(VectorFloat<?>[]::new);
            Dataset queryDataset = hdf.getDatasetByPath("test");
            if (((FloatingPoint) queryDataset.getDataType()).getBitPrecision() == 64) {
                // lastfm dataset contains f64 queries but f32 everything else
                var doubles = ((double[][]) queryDataset.getData());
                queryVectors = IntStream.range(0, doubles.length).parallel().mapToObj(i -> {
                    var a = new float[doubles[i].length];
                    for (int j = 0; j < doubles[i].length; j++) {
                        a[j] = (float) doubles[i][j];
                    }
                    return vectorTypeSupport.createFloatVector(a);
                }).toArray(VectorFloat<?>[]::new);
            } else {
                var queryVectorsArray = (float[][]) queryDataset.getData();
                queryVectors = IntStream.range(0, queryVectorsArray.length).parallel().mapToObj(i -> vectorTypeSupport.createFloatVector(queryVectorsArray[i])).toArray(VectorFloat<?>[]::new);
            }
            int[][] groundTruth = (int[][]) hdf.getDatasetByPath("neighbors").getData();
            gtSets = new ArrayList<>(groundTruth.length);
            for (int[] i : groundTruth) {
                var gtSet = new ArrayList<Integer>(i.length);
                for (int j : i) {
                    gtSet.add(j);
                }
                gtSets.add(gtSet);
            }
        }

        return DataSetUtils.processDataSet(
                path.getFileName().toString(),
                props,
                Arrays.asList(baseVectors),
                Arrays.asList(queryVectors),
                gtSets);
    }

    /// Looks up dataset properties in {@code dataset_metadata.yml}.
    ///
    /// @param datasetName the logical dataset name (without {@code .hdf5} extension)
    /// @return the dataset properties, or a minimal name-only property set if no entry exists
    private static DataSetProperties getProperties(String datasetName) {
        return metadata.getProperties(datasetName)
                .orElse(new DataSetProperties.PropertyMap(Map.of(DataSetProperties.KEY_NAME, datasetName)));
    }

    /// Downloads the HDF5 file for the given dataset if it is not already present locally.
    ///
    /// @param datasetName the logical dataset name (without {@code .hdf5} extension)
    /// @return the local path to the HDF5 file, or empty if the remote file was not found
    private Optional<Path> maybeDownloadHdf5(String datasetName) {
        var dsFilePath = HDF5_DIR.resolve(datasetName+HDF5_EXTN);

        if (Files.exists(dsFilePath)) {
            return Optional.of(dsFilePath);
        }

        // Download from https://ann-benchmarks.com/datasetName
        var url = "https://ann-benchmarks.com/" + datasetName + HDF5_EXTN;

        HttpURLConnection connection;
        while (true) {
            int responseCode;
            try {
                connection = (HttpURLConnection) new URL(url).openConnection();
                responseCode = connection.getResponseCode();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            if (responseCode == HttpURLConnection.HTTP_NOT_FOUND) {
                return Optional.empty();
            }
            if (responseCode == HttpURLConnection.HTTP_MOVED_PERM || responseCode == HttpURLConnection.HTTP_MOVED_TEMP) {
                String newUrl = connection.getHeaderField("Location");
                System.out.println("Redirect detected to URL: " + newUrl);
                url = newUrl;
            } else {
                break;
            }
        }

        try (InputStream in = connection.getInputStream()) {
            Files.createDirectories(dsFilePath.getParent());
            System.out.println("Downloading: " + url);
            Files.copy(in, dsFilePath, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException("Error downloading data:" + e.getMessage(),e);
        }
        return Optional.of(dsFilePath);
    }

}

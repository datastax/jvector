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

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
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
import java.util.Optional;
import java.util.stream.IntStream;

/**
 * This dataset loader will get and load hdf5 files from <a href="https://ann-benchmarks.com/">ann-benchmarks</a>.
 */
public class DataSetLoaderHDF5 implements DataSetLoader {
    public static final String HDF5_DIR = "hdf5/";
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * {@inheritDoc}
     */
    public Optional<DataSet> loadDataSet(String filename) {
        return maybeDownloadHdf5(filename).map(this::readHdf5Data);
    }

    private DataSet readHdf5Data(Path filename) {

        // infer the similarity
        VectorSimilarityFunction similarityFunction = getVectorSimilarityFunction(filename);

        // read the data
        VectorFloat<?>[] baseVectors;
        VectorFloat<?>[] queryVectors;
        Path path = Path.of(HDF5_DIR).resolve(filename);
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

        return DataSetUtils.getScrubbedDataSet(path.getFileName().toString(), similarityFunction, Arrays.asList(baseVectors), Arrays.asList(queryVectors), gtSets);
    }

    /**
     * Derive the similarity function from the dataset name.
     * @param filename filename of the dataset AKA "name"
     * @return The matching similarity function, or throw an error
     */
    private static VectorSimilarityFunction getVectorSimilarityFunction(Path filename) {
        VectorSimilarityFunction similarityFunction;
        if (filename.toString().contains("-angular") || filename.toString().contains("-dot")) {
            similarityFunction = VectorSimilarityFunction.COSINE;
        }
        else if (filename.toString().contains("-euclidean")) {
            similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        }
        else {
            throw new IllegalArgumentException("Unknown similarity function -- expected angular or euclidean for " + filename);
        }
        return similarityFunction;
    }

    private Optional<Path> maybeDownloadHdf5(String datasetName) {

        Path hdf5DirPath = Path.of(DataSetLoaderHDF5.HDF5_DIR);
        var localPath = hdf5DirPath.resolve(datasetName);
        if (Files.exists(localPath)) {
            return Optional.of(localPath);
        }

        // Download from https://ann-benchmarks.com/datasetName
        var url = "https://ann-benchmarks.com/" + datasetName;
        System.out.println("Downloading: " + url);

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
            Files.createDirectories(hdf5DirPath);
            Files.copy(in, localPath, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException("Error downloading data:" + e.getMessage(),e);
        }
        return Optional.of(localPath);
    }

}

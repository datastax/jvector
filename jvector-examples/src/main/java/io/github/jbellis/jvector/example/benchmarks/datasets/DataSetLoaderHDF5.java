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
    private static final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(DataSetLoaderHDF5.class);
    public static final Path HDF5_DIR = Path.of("hdf5");
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    public static final String HDF5_EXTN = ".hdf5";

    public static final String NAME = "HDF5";
    public String getName() {
        return NAME;
    }

    private static final java.util.Set<String> KNOWN_DATASETS = java.util.Set.of(
            "deep-image-96-angular",
            "fashion-mnist-784-euclidean",
            "gist-960-euclidean",
            "glove-25-angular",
            "glove-50-angular",
            "glove-100-angular",
            "glove-200-angular",
            "kosarak-jaccard",
            "mnist-784-euclidean",
            "movielens10m-jaccard",
            "nytimes-256-angular",
            "sift-128-euclidean",
            "lastfm-64-dot",
            "coco-i2i-512-angular",
            "coco-t2i-512-angular"
    );


    /**
     * {@inheritDoc}
     */
    public Optional<DataSet> loadDataSet(String datasetName) {

        // HDF5 loader does not support profiles
        if (datasetName.contains(":")) {
            logger.trace("Dataset '{}' has a profile, which is not supported by the HDF5 loader.", datasetName);
            return Optional.empty();
        }

        // If not local, only download if it's explicitly known to be on ann-benchmarks.com
        if (!KNOWN_DATASETS.contains(datasetName)) {
            logger.trace("Dataset '{}' not in known list, skipping HDF5 download.", datasetName);
            return Optional.empty();
        }

        // If it exists locally, we're good
        var dsFilePath = HDF5_DIR.resolve(datasetName + HDF5_EXTN);
        if (Files.exists(dsFilePath)) {
            logger.trace("Dataset '{}' already downloaded.", datasetName);
            return Optional.of(readHdf5Data(dsFilePath));
        }

        return maybeDownloadHdf5(datasetName).map(this::readHdf5Data);
    }


    private DataSet readHdf5Data(Path path) {

        // infer the similarity
        VectorSimilarityFunction similarityFunction = getVectorSimilarityFunction(path);

        // read the data
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
        var dsFilePath = HDF5_DIR.resolve(datasetName + HDF5_EXTN);

        // Download from https://ann-benchmarks.com/datasetName
        var url = "https://ann-benchmarks.com/" + datasetName + HDF5_EXTN;
        logger.info("Downloading: {}", url);


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
                logger.info("Redirect detected to URL: {}", newUrl);
                url = newUrl;
            } else {
                break;
            }
        }

        try (InputStream in = connection.getInputStream()) {
            Files.createDirectories(dsFilePath.getParent());
            Files.copy(in, dsFilePath, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException("Error downloading data:" + e.getMessage(),e);
        }
        return Optional.of(dsFilePath);
    }

}

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

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import io.jhdf.object.datatype.DataType;
import io.jhdf.object.datatype.FloatingPoint;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.IntStream;

public class Hdf5Loader {
    public static final String HDF5_DIR = "hdf5/";

    public static DataSet load(String filename) {
        // infer the similarity
        VectorSimilarityFunction similarityFunction;
        if (filename.contains("-angular") || filename.contains("-dot")) {
            similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        }
        else if (filename.contains("-euclidean")) {
            similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        }
        else {
            throw new IllegalArgumentException("Unknown similarity function -- expected angular or euclidean for " + filename);
        }

        // read the data
        float[][] baseVectors;
        float[][] queryVectors;
        int[][] groundTruth;
        Path path = Path.of(HDF5_DIR).resolve(filename);
        try (HdfFile hdf = new HdfFile(path)) {
            baseVectors = (float[][]) hdf.getDatasetByPath("train").getData();
            Dataset queryDataset = hdf.getDatasetByPath("test");
            if (((FloatingPoint) queryDataset.getDataType()).getBitPrecision() == 64) {
                // lastfm dataset contains f64 queries but f32 everything else
                var doubles = ((double[][]) queryDataset.getData());
                queryVectors = IntStream.range(0, doubles.length).parallel().mapToObj(i -> {
                    var a = new float[doubles[i].length];
                    for (int j = 0; j < doubles[i].length; j++) {
                        a[j] = (float) doubles[i][j];
                    }
                    return a;
                }).toArray(float[][]::new);
            } else {
                queryVectors = (float[][]) queryDataset.getData();
            }
            groundTruth = (int[][]) hdf.getDatasetByPath("neighbors").getData();
        }

        List<float[]> scrubbedBaseVectors;
        List<float[]> scrubbedQueryVectors;
        List<Set<Integer>> gtSet;
        if (similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
            // verify that vectors are normalized and sane.
            // this is necessary b/c NYT dataset contains zero vectors (!)
            scrubbedBaseVectors = new ArrayList<>(baseVectors.length);
            scrubbedQueryVectors = new ArrayList<>(queryVectors.length);
            gtSet = new ArrayList<>(groundTruth.length);
            // remove zero vectors, noting that this will change the indexes of the ground truth answers
            Map<Integer, Integer> rawToScrubbed = new HashMap<>();
            {
                int j = 0;
                for (int i = 0; i < baseVectors.length; i++) {
                    float[] v = baseVectors[i];
                    if (Math.abs(normOf(v)) > 1e-5) {
                        scrubbedBaseVectors.add(v);
                        rawToScrubbed.put(i, j++);
                    }
                }
            }
            for (int i = 0; i < queryVectors.length; i++) {
                float[] v = queryVectors[i];
                if (Math.abs(normOf(v)) > 1e-5) {
                    scrubbedQueryVectors.add(v);
                    var gt = new HashSet<Integer>();
                    for (int j = 0; j < groundTruth[i].length; j++) {
                        gt.add(rawToScrubbed.get(groundTruth[i][j]));
                    }
                    gtSet.add(gt);
                }
            }
            // now that the zero vectors are removed, we can normalize
            if (Math.abs(normOf(baseVectors[0]) - 1.0) > 1e-5) {
                normalizeAll(scrubbedBaseVectors);
                normalizeAll(scrubbedQueryVectors);
            }
            assert scrubbedQueryVectors.size() == gtSet.size();
        }
        else {
            scrubbedBaseVectors = Arrays.asList(baseVectors);
            scrubbedQueryVectors = Arrays.asList(queryVectors);
            gtSet = new ArrayList<>(groundTruth.length);
            for (int[] gt : groundTruth) {
                var gtSetForQuery = new HashSet<Integer>();
                for (int i : gt) {
                    gtSetForQuery.add(i);
                }
                gtSet.add(gtSetForQuery);
            }
        }

        System.out.format("%n%s: %d base and %d query vectors loaded, dimensions %d%n",
                          filename, scrubbedBaseVectors.size(), scrubbedQueryVectors.size(), scrubbedBaseVectors.get(0).length);

        return new DataSet(path.getFileName().toString(), similarityFunction, scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static void normalizeAll(Iterable<float[]> vectors) {
        for (float[] v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }
}

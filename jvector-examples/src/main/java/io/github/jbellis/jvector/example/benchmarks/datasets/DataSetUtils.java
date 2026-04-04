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
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class DataSetUtils {

    private static final Logger logger = LoggerFactory.getLogger(DataSetUtils.class);

    /**
     * Return a dataset containing the given vectors, scrubbed free from zero vectors and normalized to unit length.
     * Note: This only scrubs and normalizes for dot product similarity.
     */
    private static final Comparator<VectorFloat<?>> VECTOR_COMPARATOR = (a, b) -> {
        assert a.length() == b.length();
        for (int i = 0; i < a.length(); i++) {
            if (a.get(i) < b.get(i)) {
                return -1;
            }
            if (a.get(i) > b.get(i)) {
                return 1;
            }
        }
        return 0;
    };

    /**
     * Processes a dataset using the configured load behavior from the dataset metadata.
     */
    public static DataSet processDataSet(String pathStr,
                                         DataSetProperties props,
                                         List<VectorFloat<?>> baseVectors,
                                         List<VectorFloat<?>> queryVectors,
                                         List<List<Integer>> groundTruth) {
        var vsf = props.similarityFunction()
                .orElseThrow(() -> new IllegalArgumentException(
                        "No similarity function configured for dataset: " + props.getName()));

        logger.info("load_behavior: " + props.loadBehavior());

        switch (props.loadBehavior()) {
            case NO_SCRUB:
                return new SimpleDataSet(pathStr, vsf, baseVectors, queryVectors, groundTruth);
            case LEGACY_SCRUB:
                return legacyScrubDataSet(pathStr, vsf, baseVectors, queryVectors, groundTruth);
            default:
                throw new IllegalArgumentException("Unsupported load behavior: " + props.loadBehavior());
        }
    }

    /**
     * @deprecated Benchmark loaders should use
     * {@link #processDataSet(String, DataSetProperties, List, List, List)}
     * so that load behavior is controlled explicitly by dataset metadata.
     */
    @Deprecated(forRemoval = true)
    public static DataSet getScrubbedDataSet(String pathStr,
                                             VectorSimilarityFunction vsf,
                                             List<VectorFloat<?>> baseVectors,
                                             List<VectorFloat<?>> queryVectors,
                                             List<List<Integer>> groundTruth) {
        return legacyScrubDataSet(pathStr, vsf, baseVectors, queryVectors, groundTruth);
    }

    private static DataSet legacyScrubDataSet(String pathStr,
                                              VectorSimilarityFunction vsf,
                                              List<VectorFloat<?>> baseVectors,
                                              List<VectorFloat<?>> queryVectors,
                                              List<List<Integer>> groundTruth) {
        List<VectorFloat<?>> scrubbedBaseVectors = new ArrayList<>(baseVectors.size());
        List<VectorFloat<?>> scrubbedQueryVectors = new ArrayList<>(queryVectors.size());
        List<ArrayList<Integer>> gtSet = new ArrayList<>(groundTruth.size());

        var uniqueVectors = new TreeSet<VectorFloat<?>>(VECTOR_COMPARATOR);
        Map<Integer, Integer> rawToScrubbed = new HashMap<>();

        int nextOrdinal = 0;
        for (int i = 0; i < baseVectors.size(); i++) {
            VectorFloat<?> v = baseVectors.get(i);
            boolean valid = isValidLegacyVector(v, vsf);
            if (valid && uniqueVectors.add(v)) {
                scrubbedBaseVectors.add(v);
                rawToScrubbed.put(i, nextOrdinal++);
            }
        }

        // Also remove zero query vectors and query vectors that are present in the base set.
        for (int i = 0; i < queryVectors.size(); i++) {
            VectorFloat<?> v = queryVectors.get(i);
            boolean valid = isValidLegacyVector(v, vsf);
            boolean dupe = uniqueVectors.contains(v);
            if (valid && !dupe) {
                scrubbedQueryVectors.add(v);
                var gt = new ArrayList<Integer>(groundTruth.get(i).size());
                for (int ordinal : groundTruth.get(i)) {
                    gt.add(rawToScrubbed.get(ordinal));
                }
                gtSet.add(gt);
            }
        }

        if (shouldNormalizeLegacy(vsf, baseVectors)) {
            normalizeAll(scrubbedBaseVectors);
            normalizeAll(scrubbedQueryVectors);
        }

        assert scrubbedQueryVectors.size() == gtSet.size();
        return new SimpleDataSet(pathStr, vsf, scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static boolean isValidLegacyVector(VectorFloat<?> vector, VectorSimilarityFunction vsf) {
        return vsf == VectorSimilarityFunction.EUCLIDEAN || Math.abs(normOf(vector)) > 1e-5;
    }

    private static boolean shouldNormalizeLegacy(VectorSimilarityFunction vsf, List<VectorFloat<?>> baseVectors) {
        return vsf == VectorSimilarityFunction.DOT_PRODUCT
                && !baseVectors.isEmpty()
                && Math.abs(normOf(baseVectors.get(0)) - 1.0) > 1e-5;
    }

    public static void normalizeAll(Iterable<VectorFloat<?>> vectors) {
        for (VectorFloat<?> v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    public static float normOf(VectorFloat<?> baseVector) {
        float norm = 0;
        for (int i = 0; i < baseVector.length(); i++) {
            norm += baseVector.get(i) * baseVector.get(i);
        }
        return (float) Math.sqrt(norm);
    }
}

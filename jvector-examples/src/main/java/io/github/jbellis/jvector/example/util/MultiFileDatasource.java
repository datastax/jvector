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

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MultiFileDatasource {
    public final String name;
    public final Path basePath;
    public final Path queriesPath;
    public final Path groundTruthPath;

    public MultiFileDatasource(String name, String basePath, String queriesPath, String groundTruthPath) {
        this.name = name;
        this.basePath = Paths.get(basePath);
        this.queriesPath = Paths.get(queriesPath);
        this.groundTruthPath = Paths.get(groundTruthPath);
    }

    public Path directory() {
        return basePath.getParent();
    }

    public Iterable<Path> paths() {
        return List.of(basePath, queriesPath, groundTruthPath);
    }

    public DataSet load() throws IOException {
        var baseVectors = SiftLoader.readFvecs("fvec/" + basePath);
        var queryVectors = SiftLoader.readFvecs("fvec/" + queriesPath);
        var gtVectors = SiftLoader.readIvecs("fvec/" + groundTruthPath);
        return DataSet.getScrubbedDataSet(name, VectorSimilarityFunction.COSINE, baseVectors, queryVectors, gtVectors);
    }

    public static Map<String, MultiFileDatasource> byName = new HashMap<>() {{
        put("degen-200k", new MultiFileDatasource("degen-200k",
                                                   "ada-degen/degen_base_vectors.fvec",
                                                   "ada-degen/degen_query_vectors.fvec",
                                                   "ada-degen/degen_ground_truth.ivec"));
        put("cohere-english-v3-100k", new MultiFileDatasource("cohere-english-v3-100k",
                                                              "wikipedia_squad/100k/cohere_embed-english-v3.0_1024_base_vectors_100000.fvec",
                                                              "wikipedia_squad/100k/cohere_embed-english-v3.0_1024_query_vectors_10000.fvec",
                                                              "wikipedia_squad/100k/cohere_embed-english-v3.0_1024_indices_b100000_q10000_k100.ivec"));
        put("colbert-10M", new MultiFileDatasource("colbert-10M",
                                                   "wikipedia_squad/10M/colbertv2.0_128_base_vectors_10000000.fvec",
                                                   "wikipedia_squad/10M/colbertv2.0_128_query_vectors_100000.fvec",
                                                   "wikipedia_squad/10M/colbertv2.0_128_indices_b10000000_q100000_k100.ivec"));
        put("colbert-1M", new MultiFileDatasource("colbert-1M",
                                                   "wikipedia_squad/1M/colbertv2.0_128_base_vectors_1000000.fvec",
                                                   "wikipedia_squad/1M/colbertv2.0_128_query_vectors_100000.fvec",
                                                   "wikipedia_squad/1M/colbertv2.0_128_indices_b1000000_q100000_k100.ivec"));
        put("nv-qa-v4-100k", new MultiFileDatasource("nv-qa-v4-100k",
                                                     "wikipedia_squad/100k/nvidia-nemo_1024_base_vectors_100000.fvec",
                                                     "wikipedia_squad/100k/nvidia-nemo_1024_query_vectors_10000.fvec",
                                                     "wikipedia_squad/100k/nvidia-nemo_1024_indices_b100000_q10000_k100.ivec"));
        put("openai-v3-large-3072-100k", new MultiFileDatasource("openai-v3-large-3072-100k",
                                                                 "wikipedia_squad/100k/text-embedding-3-large_3072_100000_base_vectors.fvec",
                                                                 "wikipedia_squad/100k/text-embedding-3-large_3072_100000_query_vectors_10000.fvec",
                                                                 "wikipedia_squad/100k/text-embedding-3-large_3072_100000_indices_query_10000.ivec"));
        put("openai-v3-large-1536-100k", new MultiFileDatasource("openai-v3-large-1536-100k",
                                                                 "wikipedia_squad/100k/text-embedding-3-large_1536_100000_base_vectors.fvec",
                                                                 "wikipedia_squad/100k/text-embedding-3-large_1536_100000_query_vectors_10000.fvec",
                                                                 "wikipedia_squad/100k/text-embedding-3-large_1536_100000_indices_query_10000.ivec"));
        put("openai-v3-small-100k", new MultiFileDatasource("openai-v3-small-100k",
                                                            "wikipedia_squad/100k/text-embedding-3-small_1536_100000_base_vectors.fvec",
                                                            "wikipedia_squad/100k/text-embedding-3-small_1536_100000_query_vectors_10000.fvec",
                                                            "wikipedia_squad/100k/text-embedding-3-small_1536_100000_indices_query_10000.ivec"));
        put("ada002-100k", new MultiFileDatasource("ada002-100k",
                                                   "wikipedia_squad/100k/ada_002_100000_base_vectors.fvec",
                                                   "wikipedia_squad/100k/ada_002_100000_query_vectors_10000.fvec",
                                                   "wikipedia_squad/100k/ada_002_100000_indices_query_10000.ivec"));
        put("ada002-1M", new MultiFileDatasource("ada002-1M",
                                                 "wikipedia_squad/1M/ada_002_1000000_base_vectors.fvec",
                                                 "wikipedia_squad/1M/ada_002_1000000_query_vectors_10000.fvec",
                                                 "wikipedia_squad/1M/ada_002_1000000_indices_query_10000.ivec"));
        put("e5-small-v2-100k", new MultiFileDatasource("e5-small-v2-100k",
                                                        "wikipedia_squad/100k/intfloat_e5-small-v2_100000_base_vectors.fvec",
                                                        "wikipedia_squad/100k/intfloat_e5-small-v2_100000_query_vectors_10000.fvec",
                                                        "wikipedia_squad/100k/intfloat_e5-small-v2_100000_indices_query_10000.ivec"));
        put("e5-base-v2-100k", new MultiFileDatasource("e5-base-v2-100k",
                                                       "wikipedia_squad/100k/intfloat_e5-base-v2_100000_base_vectors.fvec",
                                                       "wikipedia_squad/100k/intfloat_e5-base-v2_100000_query_vectors_10000.fvec",
                                                       "wikipedia_squad/100k/intfloat_e5-base-v2_100000_indices_query_10000.ivec"));
        put("e5-large-v2-100k", new MultiFileDatasource("e5-large-v2-100k",
                                                        "wikipedia_squad/100k/intfloat_e5-large-v2_100000_base_vectors.fvec",
                                                        "wikipedia_squad/100k/intfloat_e5-large-v2_100000_query_vectors_10000.fvec",
                                                        "wikipedia_squad/100k/intfloat_e5-large-v2_100000_indices_query_10000.ivec"));
        put("gecko-100k", new MultiFileDatasource("gecko-100k",
                                                  "wikipedia_squad/100k/textembedding-gecko_100000_base_vectors.fvec",
                                                  "wikipedia_squad/100k/textembedding-gecko_100000_query_vectors_10000.fvec",
                                                  "wikipedia_squad/100k/textembedding-gecko_100000_indices_query_10000.ivec"));
    }};
}

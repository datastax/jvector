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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedASH;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;

/**
 * The similarity function a graph records in its header from format version 8: a float32 function, an int8
 * function, or none.
 * <p>
 * On disk it is two ints: the element type ({@code 0} none, {@code 1} float32, {@code 2} int8), then the
 * function ({@code 0} with no element type; otherwise {@code 1} DOT_PRODUCT, {@code 2} EUCLIDEAN, {@code 3}
 * COSINE, the codes ASH uses for its model). The codes are fixed, independent of the enums' declaration order.
 */
final class RecordedSimilarity {
    private static final Logger logger = LoggerFactory.getLogger(RecordedSimilarity.class);

    /** Bytes this occupies in a header. */
    static final int SIZE = 2 * Integer.BYTES;

    static final RecordedSimilarity NONE = new RecordedSimilarity(null, null);

    private static final int NO_ELEMENT_TYPE = 0;
    private static final int FLOAT32 = 1;
    private static final int INT8 = 2;

    private final VectorSimilarityFunction floatFunction;
    private final ByteVectorSimilarityFunction byteFunction;

    private RecordedSimilarity(VectorSimilarityFunction floatFunction, ByteVectorSimilarityFunction byteFunction) {
        this.floatFunction = floatFunction;
        this.byteFunction = byteFunction;
    }

    static RecordedSimilarity of(VectorSimilarityFunction function) {
        return new RecordedSimilarity(function, null);
    }

    static RecordedSimilarity of(ByteVectorSimilarityFunction function) {
        return new RecordedSimilarity(null, function);
    }

    Optional<VectorSimilarityFunction> floatFunction() {
        return Optional.ofNullable(floatFunction);
    }

    Optional<ByteVectorSimilarityFunction> byteFunction() {
        return Optional.ofNullable(byteFunction);
    }

    /**
     * What a writer records for {@code graph} written with {@code features}: the graph's own function, or, for a
     * graph that names none, the function of its {@code FUSED_ASH} model.
     * <p>
     * ASH codes are only meaningful for the function their model was trained with, and ASH scorers reject any
     * other, so a fused-ASH file is searched with the model's function. When the graph names a different one
     * (for example, a graph built with EUCLIDEAN scores over unit vectors, fused with a DOT_PRODUCT model), the
     * model's function is recorded, with a warning, rather than refusing a write that worked before.
     *
     * @param warn whether to log a disagreement: true when the output format records the function
     */
    static RecordedSimilarity forGraph(GraphIndex graph, Map<FeatureId, ? extends Feature> features, boolean warn) {
        Optional<VectorSimilarityFunction> graphFunction = graph.similarityFunction();
        var fusedAsh = (FusedASH) features.get(FeatureId.FUSED_ASH);
        if (fusedAsh != null) {
            VectorSimilarityFunction modelFunction = fusedAsh.getASH().similarityFunction();
            if (warn && graphFunction.isPresent() && graphFunction.get() != modelFunction) {
                logger.warn("The graph was built with {} similarity but its FUSED_ASH model scores {}; recording {}, "
                        + "the function the file can be searched with", graphFunction.get(), modelFunction, modelFunction);
            }
            return of(modelFunction);
        }
        if (graphFunction.isPresent()) {
            return of(graphFunction.get());
        }
        return graph.byteSimilarityFunction().map(RecordedSimilarity::of).orElse(NONE);
    }

    void write(IndexWriter out) throws IOException {
        if (floatFunction != null) {
            out.writeInt(FLOAT32);
            out.writeInt(code(floatFunction.name()));
        } else if (byteFunction != null) {
            out.writeInt(INT8);
            out.writeInt(code(byteFunction.name()));
        } else {
            out.writeInt(NO_ELEMENT_TYPE);
            out.writeInt(0);
        }
    }

    static RecordedSimilarity read(RandomAccessReader in) throws IOException {
        int elementType = in.readInt();
        int code = in.readInt();
        switch (elementType) {
            case NO_ELEMENT_TYPE:
                return NONE;
            case FLOAT32:
                return of(VectorSimilarityFunction.valueOf(name(code)));
            case INT8:
                return of(ByteVectorSimilarityFunction.valueOf(name(code)));
            default:
                throw new IOException("Unknown element type " + elementType + " for the recorded similarity function");
        }
    }

    // Both enums name their functions DOT_PRODUCT, EUCLIDEAN and COSINE.
    private static int code(String functionName) {
        switch (functionName) {
            case "DOT_PRODUCT": return 1;
            case "EUCLIDEAN": return 2;
            case "COSINE": return 3;
            default: throw new IllegalArgumentException("No on-disk code for similarity function " + functionName);
        }
    }

    private static String name(int code) throws IOException {
        switch (code) {
            case 1: return "DOT_PRODUCT";
            case 2: return "EUCLIDEAN";
            case 3: return "COSINE";
            default: throw new IOException("Unknown similarity function code " + code);
        }
    }

    @Override
    public String toString() {
        return floatFunction != null ? floatFunction.toString()
                : byteFunction != null ? "int8 " + byteFunction
                : "none";
    }
}

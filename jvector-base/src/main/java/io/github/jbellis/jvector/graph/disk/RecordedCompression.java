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
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.management.CompressionType;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.BinaryQuantization;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;

/**
 * The compression of the codes a graph is searched with, which a graph records in its header from format version 8:
 * PQ or ASH for fused codes, or, for codes stored beside the graph, the type its writer declared
 * ({@code withCompressionType}). {@link CompressionType#NONE} means the graph is searched with its stored vectors
 * alone; unknown means nothing was declared.
 * <p>
 * On disk it is one int: {@code 0} unknown, {@code 1} NONE, {@code 2} PQ, {@code 3} BQ, {@code 4} ASH. The codes are
 * fixed, independent of the enum's declaration order.
 */
final class RecordedCompression {
    /** Bytes this occupies in a header. */
    static final int SIZE = Integer.BYTES;

    static final RecordedCompression UNKNOWN = new RecordedCompression(null);

    private final CompressionType type;

    private RecordedCompression(CompressionType type) {
        this.type = type;
    }

    static RecordedCompression of(CompressionType type) {
        return type == null ? UNKNOWN : new RecordedCompression(type);
    }

    /** The recorded type; empty when unknown. */
    Optional<CompressionType> type() {
        return Optional.ofNullable(type);
    }

    /**
     * What a writer records: the type of the fused codes among {@code features}, or the {@code declared} type (null
     * when none was declared) for codes stored beside the graph.
     *
     * @throws IllegalArgumentException if a type is declared for a format before version 8, which can't record it,
     *         or one other than that of the fused codes
     */
    static RecordedCompression forWriter(CompressionType declared, Map<FeatureId, ? extends Feature> features,
                                         int version) {
        CompressionType fused = fusedType(features);
        if (declared == null) {
            return of(fused);
        }
        if (version < 8) {
            throw new IllegalArgumentException("Recording the compression type requires format version 8 or later; "
                    + "version " + version + " can't record " + declared);
        }
        if (fused != null && declared != fused) {
            throw new IllegalArgumentException("The graph fuses " + fused + " codes, so its compression type is "
                    + fused + ", not " + declared);
        }
        return of(declared);
    }

    /** The type of the fused codes among {@code features}, or null if there are none. */
    static CompressionType fusedType(Map<FeatureId, ? extends Feature> features) {
        if (features.containsKey(FeatureId.FUSED_PQ)) {
            return CompressionType.PQ;
        }
        if (features.containsKey(FeatureId.FUSED_ASH)) {
            return CompressionType.ASH;
        }
        return null;
    }

    /** The compression type of {@code vectors}, or null for a kind of codes {@link CompressionType} doesn't name. */
    static CompressionType typeOf(CompressedVectors vectors) {
        var compressor = vectors.getCompressor();
        if (compressor instanceof ProductQuantization) {
            return CompressionType.PQ;
        }
        if (compressor instanceof BinaryQuantization) {
            return CompressionType.BQ;
        }
        if (compressor instanceof AsymmetricHashing) {
            return CompressionType.ASH;
        }
        return null;
    }

    void write(IndexWriter out) throws IOException {
        out.writeInt(code(type));
    }

    static RecordedCompression read(RandomAccessReader in) throws IOException {
        int code = in.readInt();
        switch (code) {
            case 0: return UNKNOWN;
            case 1: return of(CompressionType.NONE);
            case 2: return of(CompressionType.PQ);
            case 3: return of(CompressionType.BQ);
            case 4: return of(CompressionType.ASH);
            default: throw new IOException("Unknown compression type code " + code);
        }
    }

    private static int code(CompressionType type) {
        if (type == null) {
            return 0;
        }
        switch (type) {
            case NONE: return 1;
            case PQ: return 2;
            case BQ: return 3;
            case ASH: return 4;
            default: throw new IllegalArgumentException("No on-disk code for compression type " + type);
        }
    }

    @Override
    public String toString() {
        return type == null ? "unknown" : type.toString();
    }
}

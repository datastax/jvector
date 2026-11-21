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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.vector.VectorizationProvider;

import java.util.Locale;

/**
 * Operand precision for projecting ASH queries. Base-vector encoding always uses FP32.
 * {@link #AUTO} picks the fastest precision the vectorization provider supports: INT8 on x86 processors
 * with VNNI or AMX, then FP16 where the CPU supports it, then FP32.
 *
 * <p>Set it per model with {@link AsymmetricHashing#setQueryOptions}; the
 * {@code jvector.ash.queryProjection} system property supplies the default.
 */
@Experimental
public enum ASHQueryProjection {
    AUTO, FP32, FP16, INT8;

    /** The system property that supplies the default precision. */
    public static final String PROPERTY = "jvector.ash.queryProjection";

    /** The precision named by {@code jvector.ash.queryProjection}, or {@link #AUTO} when it is unset. */
    public static ASHQueryProjection fromSystemProperty() {
        return parse(System.getProperty(PROPERTY, "auto"));
    }

    /**
     * Whether this JVM and processor can project queries at this precision. {@link #AUTO} and {@link #FP32}
     * always can; {@link #FP16} and {@link #INT8} need the native vectorization provider and suitable hardware.
     * The answer does not need a trained model, so applications can check before building an index.
     */
    public boolean isSupported() {
        return unsupportedReason() == null;
    }

    /**
     * Throws {@link UnsupportedOperationException}, saying why and what to use instead, when this precision
     * is not supported; see {@link #isSupported()}.
     */
    public void requireSupported() {
        String reason = unsupportedReason();
        if (reason != null) throw new UnsupportedOperationException(reason);
    }

    private String unsupportedReason() {
        if (this == AUTO || this == FP32) return null;
        return VectorizationProvider.getInstance().getVectorUtilSupport().ashQueryProjectionUnsupportedReason(this);
    }

    /** Parses {@code auto}, {@code fp32}, {@code fp16} or {@code int8}, ignoring case. */
    public static ASHQueryProjection parse(String value) {
        switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "auto": return AUTO;
            case "fp32": return FP32;
            case "fp16": return FP16;
            case "int8": return INT8;
            default: throw new IllegalArgumentException(PROPERTY + " must be auto, fp32, fp16 or int8: " + value);
        }
    }
}

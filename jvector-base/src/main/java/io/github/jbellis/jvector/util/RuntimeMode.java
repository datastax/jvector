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

package io.github.jbellis.jvector.util;

import java.util.Locale;
import java.util.logging.Logger;

/**
 * Library-wide runtime mode, from the {@code jvector.mode} system property.
 *
 * <p>Two modes: {@code prod} (or {@code production}) and {@code dev} (or
 * {@code development}), case-insensitive. The mode gates <em>diagnostic
 * walks</em> — computations whose value is re-derived from first principles by
 * traversing a structure, such as {@link
 * io.github.jbellis.jvector.quantization.PQVectors#ramBytesUsed()} summing
 * every compressed chunk. In production mode such walks are categorically
 * replaced by incrementally-maintained values; in development mode they run in
 * full, so a drifted cache or accounting bug is observable.
 *
 * <p><b>Default is production.</b> An embedding host once called the chunk
 * walk once per inserted vector, turning index-build accounting into
 * O(n&sup2;) — a single compaction burned two CPU-hours inside
 * {@code ramBytesUsed} while build workers starved. A diagnostic full-walk is
 * something a developer opts into, not something a production host should
 * have to know to opt out of.
 *
 * <p>Unrecognized values log a warning and resolve to production, not
 * development: falling back to the diagnostic mode on a typo would silently
 * reintroduce exactly the pathology above.
 */
public final class RuntimeMode {
    private static final Logger LOG = Logger.getLogger(RuntimeMode.class.getName());

    public static final String PROPERTY = "jvector.mode";

    private static final boolean DEVELOPMENT =
            parseIsDevelopment(System.getProperty(PROPERTY), LOG);

    private RuntimeMode() {
    }

    /** True when diagnostic walks should run in full. */
    public static boolean isDevelopment() {
        return DEVELOPMENT;
    }

    /** True when diagnostic walks are replaced by maintained values. */
    public static boolean isProduction() {
        return !DEVELOPMENT;
    }

    /**
     * Pure parse, exposed for tests (the static mode is fixed at class load).
     */
    static boolean parseIsDevelopment(String raw, Logger log) {
        if (raw == null) {
            return false;
        }
        switch (raw.trim().toLowerCase(Locale.ROOT)) {
            case "dev":
            case "development":
                return true;
            case "prod":
            case "production":
            case "":
                return false;
            default:
                log.warning(() -> PROPERTY + "=" + raw
                        + " is not recognized (expected prod|production|dev|development); "
                        + "using production. Development mode re-enables full diagnostic "
                        + "walks and must be asked for exactly.");
                return false;
        }
    }
}

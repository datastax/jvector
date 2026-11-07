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

package io.github.jbellis.jvector.example;

/**
 * Receives {@link Grid}'s query measurements as they are produced. Console output, experiment logs and
 * result collection are listeners, so a new report is a new listener and leaves the pipeline alone.
 *
 * <p>Events arrive on one thread in this order, for each index and search compressor:</p>
 * <pre>
 * onSearchStart
 *   for each pruning setting: onPruningStart
 *     for each topK: onTopKStart, onQueryResult per overquery factor, onTopKEnd
 * </pre>
 */
interface GridListener {
    /** Before any query runs for this index and search compressor; throw here to reject the configuration. */
    default void onSearchStart(SearchContext ctx) {}

    default void onPruningStart(SearchContext ctx, boolean usePruning) {}

    default void onTopKStart(SearchContext ctx, boolean usePruning, int topK) {}

    /** One measured row, including the construction metrics for its index. */
    default void onQueryResult(SearchContext ctx, QueryRow row) {}

    default void onTopKEnd(SearchContext ctx, boolean usePruning, int topK) {}
}

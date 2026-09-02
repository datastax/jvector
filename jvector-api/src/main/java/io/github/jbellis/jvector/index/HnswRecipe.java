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

package io.github.jbellis.jvector.index;

/**
 * Named, recommended starting configurations for a graph/HNSW index, applied via a graph index
 * builder's {@code applyRecipe(HnswRecipe)}.
 * <p>
 * Scaffolding only: the constants below name the recipes callers will eventually reach for, but
 * the actual fixed parameter values (what "high recall" means numerically for this backing) have
 * not been decided yet, so {@code applyRecipe} currently refuses at runtime rather than guess.
 */
public enum HnswRecipe {
    HIGH_RECALL,
    HIGH_PERFORMANCE
}

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
 * Named, recommended starting configurations for an IVF index, applied via an IVF index builder's
 * {@code applyRecipe(IvfRecipe)}.
 * <p>
 * Scaffolding only: the constants below name the recipes callers will eventually reach for, but
 * the actual fixed parameter values have not been decided yet &mdash; IVF's construction
 * parameters themselves are still being worked out &mdash; so {@code applyRecipe} currently
 * refuses at runtime rather than guess.
 */
public enum IvfRecipe {
    HIGH_RECALL,
    HIGH_PERFORMANCE
}

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

package io.github.jbellis.jvector.util.work;

import io.github.jbellis.jvector.annotations.Experimental;

/**
 * Identifies a stage of a long-running operation. The consumer defines its own stages; an
 * {@code enum} satisfies this for free via {@link Enum#name()}.
 *
 * <p>Part of the generic progress + work-admission SPI ({@link ProgressTracker},
 * {@link WorkLimiter}, {@link ProgressLimiter}). Neither the stage identity nor the unit of work
 * is fixed by jvector; both are supplied by the consumer.
 */
@Experimental
public interface WorkStage {
    /** The stage's name, stable within a single operation. */
    String name();
}

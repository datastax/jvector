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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.annotations.Experimental;

import java.util.Optional;
import java.util.ServiceLoader;

/**
 * Selection of a {@link GraphBuildAccelerator}. Accelerators are opt-in: {@link #find()} returns one only when the
 * system property {@value #PROPERTY} names an accelerator found on the class path.
 */
@Experimental
public final class GraphBuildAccelerators {

    /** System property naming the accelerator to use. */
    public static final String PROPERTY = "jvector.graph.accelerator";

    private GraphBuildAccelerators() {
    }

    /** @return the accelerator named by {@value #PROPERTY}, if it is set and that accelerator is on the class path */
    public static Optional<GraphBuildAccelerator> find() {
        String name = System.getProperty(PROPERTY);
        if (name == null || name.isEmpty() || name.equalsIgnoreCase("none")) {
            return Optional.empty();
        }
        for (GraphBuildAccelerator accelerator : ServiceLoader.load(GraphBuildAccelerator.class)) {
            if (accelerator.name().equalsIgnoreCase(name)) {
                return Optional.of(accelerator);
            }
        }
        return Optional.empty();
    }
}

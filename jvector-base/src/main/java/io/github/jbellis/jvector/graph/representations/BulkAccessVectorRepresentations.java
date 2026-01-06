/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph.representations;

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorRepresentation;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.function.Supplier;
import java.util.logging.Logger;

/**
 * Provides similarity computations in bulk (i.e., sequentially) to the stored vectors.
 */
public interface BulkAccessVectorRepresentations extends Accountable {
    Logger LOG = Logger.getLogger(BulkAccessVectorRepresentations.class.getName());

    /**
     * Return the number of vector values.
     * <p>
     * All copies of a given BAVR should have the same size.  Typically this is achieved by either
     * (1) implementing a threadsafe, un-shared BAVR, where `copy` returns `this`, or
     * (2) implementing a fixed-size BAVR.
     */
    int size();

    /** Return the dimension of the returned vector values */
    int dimension();

    /**
     * Return the similarities between the query and the stored vectors. The length of the result
     * will match the size() of this.
     *
     * <p>For performance, implementations are free to re-use the same object across invocations.
     * That is, you will get back the same float[] reference (for instance) for every query.
     * If you want to use those values across calls, you should make a copy.
     *
     * @param query the query vector.
     */
    float[] getSimilarities(VectorFloat<?> query);

    /**
     * @return true iff the vector returned by `getSimilarities` is shared.  A shared vector will
     * only be valid until the next call to getVector overwrites it.
     */
    boolean isValueShared();

    /**
     * Creates a new copy of this {@link BulkAccessVectorRepresentations}. This is helpful when you need to
     * access different values at once, to avoid overwriting the underlying float vector returned by
     * a shared {@link BulkAccessVectorRepresentations#getSimilarities}.
     * <p>
     * Un-shared implementations may simply return `this`.
     */
    BulkAccessVectorRepresentations copy();

    /**
     * Returns a supplier of thread-local copies of the RAVV.
     */
    default Supplier<BulkAccessVectorRepresentations> threadLocalSupplier() {
        if (!isValueShared()) {
            return () -> this;
        }

        if (this instanceof AutoCloseable) {
            LOG.warning("RAVV is shared and implements AutoCloseable; threadLocalSupplier() may lead to leaks");
        }
        var tl = ExplicitThreadLocal.withInitial(this::copy);
        return tl::get;
    }
}

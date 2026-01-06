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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.graph.representations.RandomAccessVectorRepresentations;
import io.github.jbellis.jvector.util.ArrayUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

public class MockVectorRepresentations implements RandomAccessVectorRepresentations {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final VectorFloat<?> scratch;
    private final int dimension;
    private final VectorFloat<?>[] denseValues;

    public static MockVectorRepresentations fromValues(VectorFloat<?>[] values) {
        return new MockVectorRepresentations(values[0].length(), values);
    }

    MockVectorRepresentations(int dimension, VectorFloat<?>[] denseValues) {
        this.dimension = dimension;
        this.denseValues = denseValues;
        this.scratch = vectorTypeSupport.createFloatVector(dimension);
    }

    @Override
    public MockVectorRepresentations copy() {
        return new MockVectorRepresentations(dimension, ArrayUtil.copyOfSubArray(denseValues, 0, denseValues.length));
    }

    @Override
    public boolean isValueShared() {
        return true;
    }

    @Override
    public VectorFloat<?> getVector(int targetOrd) {
        VectorFloat<?> original = denseValues[targetOrd];
        // present a single vector reference to callers like the disk-backed RAVV implmentations,
        // to catch cases where they are not making a copy
        scratch.copyFrom(original, 0, 0, dimension);
        return scratch;
    }

    @Override
    public int size() {
        return denseValues.length;
    }

    @Override
    public int dimension() {
        return dimension;
    }
}

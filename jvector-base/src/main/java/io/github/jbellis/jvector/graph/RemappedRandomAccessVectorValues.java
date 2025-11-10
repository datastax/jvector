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

import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Arrays;

public class RemappedRandomAccessVectorValues implements RandomAccessVectorValues {
    private final RandomAccessVectorValues ravv;
    private final int[] graphToRavvOrdMap;

    /**
     * Remaps a RAVV to a different set of ordinals.  This is useful when the ordinals used by the graph
     * do not match the ordinals used by the RAVV.
     *
     * @param ravv the RAVV to remap
     * @param graphToRavvOrdMap a mapping from the graph's ordinals to the RAVV's ordinals where
     *                         graphToRavvOrdMap[i] is the RAVV ordinal corresponding to graph ordinal i.
     */
    public RemappedRandomAccessVectorValues(RandomAccessVectorValues ravv, int[] graphToRavvOrdMap) {
        this.ravv = ravv;
        this.graphToRavvOrdMap = graphToRavvOrdMap;
    }

    @Override
    public int size() {
        return graphToRavvOrdMap.length;
    }

    @Override
    public int dimension() {
        return ravv.dimension();
    }

    @Override
    public VectorFloat<?> getVector(int node) {
        return ravv.getVector(graphToRavvOrdMap[node]);
    }

    @Override
    public boolean isValueShared() {
        return ravv.isValueShared();
    }

    @Override
    public RandomAccessVectorValues copy() {
        return new RemappedRandomAccessVectorValues(ravv.copy(), Arrays.copyOf(graphToRavvOrdMap, graphToRavvOrdMap.length));
    }

    @Override
    public void getVectorInto(int node, VectorFloat<?> result, int offset) {
        ravv.getVectorInto(graphToRavvOrdMap[node], result, offset);
    }
}

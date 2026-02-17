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

package io.github.jbellis.jvector.graph.disk;

import org.agrona.collections.Int2IntHashMap;

import java.util.Arrays;
import java.util.Map;

/**
 * Allows mapping the ordinals that an index was built with to different ordinals while writing to disk.
 * This is necessary for two use cases:
 *  - Filling in "holes" left by deleted nodes in the ordinal range
 *  - Cassandra wants to map ordinals to rowids where possible, which saves a lookup at read time,
 *    but it doesn't know what rowid vectors in the memtable will get until later, when flushed.
 */
public interface OrdinalMapper {
    /**
     * Used by newToOld to indicate that the new ordinal is a "hole" that has no corresponding old ordinal.
     */
    int OMITTED = Integer.MIN_VALUE;

    /**
     * OnDiskGraphIndexWriter will iterate from 0..maxOrdinal(), inclusive.
     */
    int maxOrdinal();

    /**
     * Map old ordinals (in the graph as constructed) to new ordinals (written to disk).
     * Should always return a valid ordinal (between 0 and maxOrdinal).
     */
    int oldToNew(int oldOrdinal);

    /**
     * Map new ordinals (written to disk) to old ordinals (in the graph as constructed).
     * May return OMITTED if there is a "hole" at the new ordinal.
     */
    int newToOld(int newOrdinal);

    /**
     * A mapper that leaves the original ordinals unchanged.
     */
    class IdentityMapper implements OrdinalMapper {
        private final int maxOrdinal;

        public IdentityMapper(int maxOrdinal) {
            this.maxOrdinal = maxOrdinal;
        }

        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        @Override
        public int oldToNew(int oldOrdinal) {
            return oldOrdinal;
        }

        @Override
        public int newToOld(int newOrdinal) {
            return newOrdinal;
        }
    }

    /**
     * Converts a Map of old to new ordinals into an OrdinalMapper.
     */
    class MapMapper implements OrdinalMapper {
        private final int maxOrdinal;
        private final Map<Integer, Integer> oldToNew;
        private final Int2IntHashMap newToOld;

        public MapMapper(Map<Integer, Integer> oldToNew) {
            this.oldToNew = oldToNew;
            this.newToOld = new Int2IntHashMap(oldToNew.size(), 0.65f, OMITTED);
            oldToNew.forEach((old, newOrdinal) -> newToOld.put(newOrdinal, old));
            this.maxOrdinal = oldToNew.values().stream().mapToInt(i -> i).max().orElse(-1);
        }

        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        @Override
        public int oldToNew(int oldOrdinal) {
            return oldToNew.get(oldOrdinal);
        }

        @Override
        public int newToOld(int newOrdinal) {
            return newToOld.get(newOrdinal);
        }
    }

    interface MutableOrdinalMapper extends OrdinalMapper {
        /**
         * Adds a mapping between oldOrdinal <-> newOrdinal
         * neither oldOrdinal or newOrdinal should be {@link OrdinalMapper#OMITTED}
         */
        void set(int oldOrdinal, int newOrdinal);
    }

    /**
     * An OrdinalMapper implementation that works best when the ordinal mappings
     * are sparse.
     * Not thread safe for concurrenct reads and writes.
     */
    class SparseOrdinalMapper implements MutableOrdinalMapper {

        private final Int2IntHashMap oldToNew;
        private final Int2IntHashMap newToOld;
        private int maxOrdinal;

        /**
         * Creates an empty ordinal mapper
         */
        SparseOrdinalMapper() {
            oldToNew = new Int2IntHashMap(OMITTED);
            newToOld = new Int2IntHashMap(OMITTED);
            maxOrdinal = Integer.MIN_VALUE;
        }

        @Override
        public void set(int oldOrdinal, int newOrdinal) {
            oldToNew.put(oldOrdinal, newOrdinal);
            newToOld.put(newOrdinal, oldOrdinal);
            if (newOrdinal > maxOrdinal) {
                maxOrdinal = newOrdinal;
            }
        }

        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        @Override
        public int oldToNew(int oldOrdinal) {
            return oldToNew.get(oldOrdinal);
        }

        @Override
        public int newToOld(int newOrdinal) {
            return newToOld.get(newOrdinal);
        }
    }

    /** 
     * An OrdinalMapper implementation that works best when ordinal mappings
     * are dense.
     * Not thread safe for concurrent reads and writes.
     */
    class DenseOrdinalMapper implements MutableOrdinalMapper {

        private final int[] oldToNew;
        private final int[] newToOld;
        private int maxOrdinal;

        /**
         * Preallocates space based on the provided inputs.
         * The bounds do not need to be precise and will not affect the
         * return value of {@link #maxOrdinal()}.
         * This mapper cannot be resized.
         */
        DenseOrdinalMapper(int oldOrdinalUpperBound, int newOrdinalUpperBound) {
            oldToNew = new int[oldOrdinalUpperBound];
            newToOld = new int[oldOrdinalUpperBound];
            Arrays.fill(oldToNew, OMITTED);
            Arrays.fill(newToOld, OMITTED);
            maxOrdinal = Integer.MIN_VALUE;
        }

        @Override
        public void set(int oldOrdinal, int newOrdinal) {
            oldToNew[oldOrdinal] = newOrdinal;
            newToOld[newOrdinal] = oldOrdinal;
            if (newOrdinal > maxOrdinal) {
                maxOrdinal = newOrdinal;
            }
        }

        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        @Override
        public int oldToNew(int oldOrdinal) {
            return oldToNew[oldOrdinal];
        }

        @Override
        public int newToOld(int newOrdinal) {
            return newToOld[newOrdinal];
        }
    }
}

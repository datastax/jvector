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
     * Returns the maximum ordinal value (inclusive) that OnDiskGraphIndexWriter will iterate over.
     *
     * @return the maximum ordinal value
     */
    int maxOrdinal();

    /**
     * Maps old ordinals (in the graph as constructed) to new ordinals (written to disk).
     * Should always return a valid ordinal (between 0 and maxOrdinal).
     *
     * @param oldOrdinal the original ordinal in the graph
     * @return the new ordinal to use when writing to disk
     */
    int oldToNew(int oldOrdinal);

    /**
     * Maps new ordinals (written to disk) to old ordinals (in the graph as constructed).
     * May return OMITTED if there is a "hole" at the new ordinal.
     *
     * @param newOrdinal the new ordinal written to disk
     * @return the original ordinal in the graph, or OMITTED if there is a hole
     */
    int newToOld(int newOrdinal);

    /**
     * A mapper that leaves the original ordinals unchanged.
     * This is the simplest implementation where old and new ordinals are identical.
     */
    class IdentityMapper implements OrdinalMapper {
        private final int maxOrdinal;

        /**
         * Constructs an IdentityMapper with the specified maximum ordinal.
         *
         * @param maxOrdinal the maximum ordinal value (inclusive)
         */
        public IdentityMapper(int maxOrdinal) {
            this.maxOrdinal = maxOrdinal;
        }

        /**
         * Returns the maximum ordinal value.
         *
         * @return the maximum ordinal
         */
        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        /**
         * Maps an old ordinal to a new ordinal. For IdentityMapper, returns the same value.
         *
         * @param oldOrdinal the original ordinal
         * @return the same ordinal unchanged
         */
        @Override
        public int oldToNew(int oldOrdinal) {
            return oldOrdinal;
        }

        /**
         * Maps a new ordinal to an old ordinal. For IdentityMapper, returns the same value.
         *
         * @param newOrdinal the new ordinal
         * @return the same ordinal unchanged
         */
        @Override
        public int newToOld(int newOrdinal) {
            return newOrdinal;
        }
    }

    /**
     * Converts a Map of old to new ordinals into an OrdinalMapper.
     * This implementation allows for arbitrary remapping and supports gaps (omitted ordinals).
     */
    class MapMapper implements OrdinalMapper {
        private final int maxOrdinal;
        private final Map<Integer, Integer> oldToNew;
        private final Int2IntHashMap newToOld;

        /**
         * Constructs a MapMapper from a map of old to new ordinals.
         * The mapper builds a reverse mapping and determines the maximum new ordinal.
         *
         * @param oldToNew a map from original ordinals to new ordinals
         */
        public MapMapper(Map<Integer, Integer> oldToNew) {
            this.oldToNew = oldToNew;
            this.newToOld = new Int2IntHashMap(oldToNew.size(), 0.65f, OMITTED);
            oldToNew.forEach((old, newOrdinal) -> newToOld.put(newOrdinal, old));
            this.maxOrdinal = oldToNew.values().stream().mapToInt(i -> i).max().orElse(-1);
        }

        /**
         * Returns the maximum new ordinal value.
         *
         * @return the maximum ordinal
         */
        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        /**
         * Maps an old ordinal to its corresponding new ordinal.
         *
         * @param oldOrdinal the original ordinal
         * @return the new ordinal corresponding to the old ordinal
         */
        @Override
        public int oldToNew(int oldOrdinal) {
            return oldToNew.get(oldOrdinal);
        }

        /**
         * Maps a new ordinal back to its original ordinal.
         * Returns OMITTED if there is no mapping for the new ordinal.
         *
         * @param newOrdinal the new ordinal
         * @return the original ordinal, or OMITTED if there is a gap
         */
        @Override
        public int newToOld(int newOrdinal) {
            return newToOld.get(newOrdinal);
        }
    }
}

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

/**
 * Placement of base-layer (L0) node records. Unaligned records are packed back to back. Page-aligned records
 * never cross a page boundary: records that fit in a page are packed as many per page as fit, and each larger
 * record starts a whole number of pages. Offsets are relative to the first record, which follows the header.
 */
final class L0Layout {
    /** Bytes of one record: node id, inline features, neighbor count and neighbors. */
    final int recordBytes;
    /** Bytes of one group of records, including the zero padding at its end. */
    final int groupBytes;
    /** Records in each group. */
    final int recordsPerGroup;
    /** Page size for aligned layouts, or 0 when unaligned. */
    final int pageBytes;

    private L0Layout(int recordBytes, int groupBytes, int recordsPerGroup, int pageBytes) {
        this.recordBytes = recordBytes;
        this.groupBytes = groupBytes;
        this.recordsPerGroup = recordsPerGroup;
        this.pageBytes = pageBytes;
    }

    static int recordBytes(int inlineBytes, int degree) {
        return Math.addExact(Integer.BYTES + inlineBytes, Math.multiplyExact(Integer.BYTES, degree + 1));
    }

    static L0Layout unaligned(int recordBytes) {
        if (recordBytes <= 0) throw new IllegalArgumentException("Record size must be positive");
        return new L0Layout(recordBytes, recordBytes, 1, 0);
    }

    /** The layout a page-aligned index of {@code recordBytes} records uses with {@code pageBytes} pages. */
    static L0Layout aligned(int recordBytes, int pageBytes) {
        if (recordBytes <= 0 || pageBytes <= 0) throw new IllegalArgumentException("Record and page sizes must be positive");
        if (recordBytes <= pageBytes) {
            return new L0Layout(recordBytes, pageBytes, pageBytes / recordBytes, pageBytes);
        }
        int pages = (recordBytes + pageBytes - 1) / pageBytes;
        return new L0Layout(recordBytes, Math.multiplyExact(pages, pageBytes), 1, pageBytes);
    }

    /**
     * Whether page alignment reduces page reads enough to justify its padding: always for records up to a page,
     * and for larger records when their last page is at least half full. Otherwise padding would add most of a
     * page per record while saving only a small fraction of a page read.
     */
    static boolean alignmentPays(int recordBytes, int pageBytes) {
        if (recordBytes <= pageBytes) return true;
        int lastPageBytes = recordBytes - (recordBytes - 1) / pageBytes * pageBytes;
        return 2 * lastPageBytes >= pageBytes;
    }

    boolean isAligned() {
        return pageBytes > 0;
    }

    /** Offset of {@code ordinal}'s record from the first record. */
    long offset(long ordinal) {
        return ordinal / recordsPerGroup * groupBytes + ordinal % recordsPerGroup * recordBytes;
    }

    /** Bytes occupied by {@code count} records, including the padding that completes the last group. */
    long totalBytes(long count) {
        if (!isAligned()) return count * recordBytes;
        return (count + recordsPerGroup - 1) / recordsPerGroup * groupBytes;
    }

    /** Zero bytes that follow {@code ordinal}'s record when {@code lastOrdinal} is the final record. */
    int paddingAfter(long ordinal, long lastOrdinal) {
        if (!isAligned()) return 0;
        long inGroup = ordinal % recordsPerGroup;
        if (ordinal != lastOrdinal && inGroup != recordsPerGroup - 1) return 0;
        return Math.toIntExact(groupBytes - (inGroup + 1) * recordBytes);
    }

    /** Disk bytes per record, padding included, divided by the record's own bytes. */
    double footprintRatio() {
        return (double) groupBytes / recordsPerGroup / recordBytes;
    }

    @Override
    public String toString() {
        if (!isAligned()) return String.format("%,d-byte records, unaligned", recordBytes);
        String placement = groupBytes == pageBytes
                ? String.format("%d per %,d-byte page", recordsPerGroup, pageBytes)
                : String.format("one per %d pages of %,d bytes", groupBytes / pageBytes, pageBytes);
        return String.format("%,d-byte records, %s, %.1f%% padding", recordBytes, placement, (footprintRatio() - 1) * 100);
    }
}

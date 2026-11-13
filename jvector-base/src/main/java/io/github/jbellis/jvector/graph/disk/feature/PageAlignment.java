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

package io.github.jbellis.jvector.graph.disk.feature;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.disk.CommonHeader;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Marks an index whose base-layer node records never cross a page boundary (format version 7 and later).
 * Records that fit in a page are packed as many per page as fit; larger records each start a whole number of
 * pages. The header is padded to a whole number of pages so the first record starts on a page boundary.
 * Writers add this through {@code withPageAlignment(pageBytes)} on a graph writer builder, never directly. It
 * contributes no bytes to a record; readers derive record offsets from the page size and record size.
 */
@Experimental
public final class PageAlignment extends AbstractFeature {
    private final int pageBytes;
    private final int headerPad;

    public PageAlignment(int pageBytes, int headerPad) {
        if (pageBytes <= 0 || Integer.bitCount(pageBytes) != 1) throw new IllegalArgumentException("Page size must be a power of two: " + pageBytes);
        if (headerPad < 0 || headerPad >= pageBytes) throw new IllegalArgumentException("Header padding must be less than a page: " + headerPad);
        this.pageBytes = pageBytes;
        this.headerPad = headerPad;
    }

    @Override
    public FeatureId id() {
        return FeatureId.PAGE_ALIGNMENT;
    }

    /** The page size records are aligned to. */
    public int pageBytes() {
        return pageBytes;
    }

    /** Zero bytes at the end of the header. */
    public int headerPad() {
        return headerPad;
    }

    /** Page alignment adds no bytes inside a record. */
    @Override
    public int featureSize() {
        return 0;
    }

    @Override
    public int headerSize() {
        return 2 * Integer.BYTES + headerPad;
    }

    @Override
    public void writeHeader(IndexWriter out) throws IOException {
        out.writeInt(pageBytes);
        out.writeInt(headerPad);
        out.write(new byte[headerPad]);
    }

    static PageAlignment load(CommonHeader header, RandomAccessReader in) {
        try {
            int pageBytes = in.readInt();
            int headerPad = in.readInt();
            var feature = new PageAlignment(pageBytes, headerPad);
            in.seek(in.getPosition() + headerPad);
            return feature;
        } catch (IOException | IllegalArgumentException e) {
            throw new UncheckedIOException(new IOException("Invalid page alignment header", e));
        }
    }
}

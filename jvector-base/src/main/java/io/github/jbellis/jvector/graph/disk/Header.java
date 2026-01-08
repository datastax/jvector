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

import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.AbstractMutableGraphIndex;
import io.github.jbellis.jvector.graph.representations.GlobalInformation;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorRepresentation;

import java.io.IOException;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Header information for an on-disk graph index, containing both common metadata and feature-specific headers.
 * <p>
 * This class encapsulates:
 * - Common header information (version, dimension, entry node, etc.)
 * - Feature set information (which features are included in the index)
 * - Feature-specific header data
 * <p>
 * The header can be written at the beginning of the index file or alternatively in a separate metadata file and is read when loading an index.
 * It provides all the metadata needed to correctly interpret the on-disk format of the graph.
 */
class Header<Primary extends VectorRepresentation, Secondary extends VectorRepresentation> implements Accountable {
    final CommonHeader common;

    final GlobalInformation<Primary> primaryGlobalInformation;
    final GlobalInformation<Secondary> secondaryGlobalInformation;

    Header(CommonHeader common, AbstractMutableGraphIndex<Primary, Secondary> graph) {
        this(common, graph.getPrimaryRepresentations().getGlobalInformation(),
                graph.getSecondaryRepresentations().getGlobalInformation());
    }

    private Header(CommonHeader common, GlobalInformation<Primary> primaryGlobalInformation, GlobalInformation<Secondary> secondaryGlobalInformation) {
        this.common = common;
        this.primaryGlobalInformation = primaryGlobalInformation;
        this.secondaryGlobalInformation = secondaryGlobalInformation;
    }

    void write(IndexWriter out) throws IOException {
        common.write(out);

        primaryGlobalInformation.write(out); // write their header
        secondaryGlobalInformation.write(out); // write their header
    }

    @Override
    public long ramBytesUsed() {
        long size = common.ramBytesUsed();
        size += this.primaryGlobalInformation.ramBytesUsed();
        size += this.primaryGlobalInformation.ramBytesUsed();
        return size;
    }

    static <Primary extends VectorRepresentation, Secondary extends VectorRepresentation> Header<Primary, Secondary> load(RandomAccessReader reader, long offset) throws IOException {
        reader.seek(offset);

        CommonHeader common = CommonHeader.load(reader);

        // read the headers for the primary and secondary representations and store them somewhere, not sure what is the right data structure or location yet
        GlobalInformation<Primary> primaryGlobalInformation = null;
        GlobalInformation<Secondary> secondaryGlobalInformation = null;

        return new Header(common, primaryGlobalInformation, secondaryGlobalInformation);
    }
}
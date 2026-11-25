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

import java.io.IOException;

/**
 * Format for version 8 of the on-disk graph format.
 * Version 8 characteristics:
 * - Everything in version 7: the same graph records, features and footer
 * - Records the similarity function the graph was built with in the common header, after the layer info
 *   (see {@link RecordedSimilarity}), so that searches can be checked against it
 * - Then the compression of the codes the graph is searched with (see {@link RecordedCompression})
 */
final class GraphIndexFormatV8 extends GraphIndexFormatV7 {

    /** Creates the singleton format for version 8. */
    GraphIndexFormatV8() {
        super(8, v8Features());
    }

    @Override
    public void writeCommonHeader(IndexWriter out, CommonHeader header) throws IOException {
        super.writeCommonHeader(out, header);
        header.similarity.write(out);
        header.compression.write(out);
    }

    @Override
    public CommonHeader readCommonHeader(RandomAccessReader in, int size) throws IOException {
        CommonHeader common = super.readCommonHeader(in, size);
        RecordedSimilarity similarity = RecordedSimilarity.read(in);
        RecordedCompression compression = RecordedCompression.read(in);
        return new CommonHeader(common.version, common.dimension, common.entryNode, common.layerInfo,
                common.idUpperBound, similarity, compression);
    }

    @Override
    public int commonHeaderSize() {
        return super.commonHeaderSize() + RecordedSimilarity.SIZE + RecordedCompression.SIZE;
    }
}

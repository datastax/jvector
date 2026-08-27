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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.util.work.ProgressTracker;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Appends a {@link NodeTokenStream} section (with {@link SimilarityKey}s) to a graph file that
 * ends in its jvector footer: reads the finished base layer back in ordinal order, writes the
 * section where the footer was, and rewrites the footer behind it. The compactor does this at the
 * end of every merge; standalone it retrofits an index written before the stream existed, which
 * is what makes such an index usable as a second-generation source (plan from keys, resident
 * search, key-window pretouch).
 */
@Experimental
public final class TokenStreamRetrofit {
    private static final Logger log = LoggerFactory.getLogger(TokenStreamRetrofit.class);
    private static final int WINDOW = 1 << 20;

    private TokenStreamRetrofit() {
    }

    /** Result of one retrofit. */
    public static final class Result {
        public final long nodes;
        public final long liveNodes;
        public final long edges;
        public final long sectionBytes;
        public final long millis;
        public final boolean alreadyPresent;

        Result(long nodes, long liveNodes, long edges, long sectionBytes, long millis, boolean alreadyPresent) {
            this.nodes = nodes;
            this.liveNodes = liveNodes;
            this.edges = edges;
            this.sectionBytes = sectionBytes;
            this.millis = millis;
            this.alreadyPresent = alreadyPresent;
        }
    }

    /**
     * @param file        a graph file whose jvector footer is the last thing in it
     * @param startOffset where the graph's header starts in the file
     * @param scope       receives progress in nodes; {@link ProgressTracker.PhaseScope#NOOP} if unwanted
     */
    public static Result append(Path file, long startOffset, ProgressTracker.PhaseScope scope) throws IOException {
        long t0 = System.nanoTime();
        // A mapped reader: the read-back is one vector and one edge list per node, and a
        // RandomAccessFile-backed reader turns each float into a syscall (measured 98 s for 4M
        // nodes against ~19 GB of records — 25 µs a node, all in read(2)).
        try (var supplier = ReaderSupplierFactory.open(file);
             var merged = OnDiskGraphIndex.load(supplier, startOffset)) {
            if (merged.tokenStreamSection().isPresent()) {
                return new Result(merged.getIdUpperBound(), -1, -1, merged.tokenStreamSection().get().length, 0, true);
            }
            long headerOffset;
            byte[] headerBytes;
            try (var in = supplier.get()) {
                long len = in.length();
                in.seek(len - AbstractGraphIndexWriter.FOOTER_MAGIC_SIZE);
                int magic = in.readInt();
                if (magic != AbstractGraphIndexWriter.FOOTER_MAGIC) {
                    throw new IllegalStateException(file + " does not end in a jvector footer: magic " + Integer.toHexString(magic));
                }
                in.seek(len - AbstractGraphIndexWriter.FOOTER_SIZE);
                headerOffset = in.readLong();
                headerBytes = new byte[(int) (len - AbstractGraphIndexWriter.FOOTER_SIZE - headerOffset)];
                in.seek(headerOffset);
                in.readFully(headerBytes);
            }
            int n = merged.getIdUpperBound();
            int maxLevel = Math.min(merged.getMaxLevel(), NodeTokenStream.MAX_LEVEL);
            byte[] levelOf = null;
            if (maxLevel > 0) {
                levelOf = new byte[n];
                for (int level = 1; level <= maxLevel; level++) {
                    for (var it = merged.getNodes(level); it.hasNext(); ) {
                        int node = it.nextInt();
                        if (node >= 0 && node < n && levelOf[node] < level) {
                            levelOf[node] = (byte) level;
                        }
                    }
                }
            }
            scope.onProgress(0, n);
            var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
            SimilarityKey keyFn = SimilarityKey.randomProjection(merged.getDimension());
            VectorFloat<?> vec = vts.createFloatVector(merged.getDimension());
            try (var writer = new BufferedRandomAccessWriter(file);
                 var view = merged.getView()) {
                writer.seek(headerOffset);
                var enc = new NodeTokenStream.Encoder(writer, NodeTokenStream.encodingByDefault(), n, merged.getDegree(0), maxLevel,
                                                      keyFn.id());
                for (int node = 0; node < n; node++) {
                    if ((node & (WINDOW - 1)) == 0) {
                        merged.prefetchL0Records(node, Math.min(n - 1, node + WINDOW - 1));
                        scope.onProgress(node, n);
                    }
                    var it = view.getNeighborsIterator(0, node);
                    boolean live = it.size() > 0;
                    int key = 0;
                    if (live) {
                        view.getVectorInto(node, vec, 0);
                        key = keyFn.keyOf(vec);
                    }
                    enc.node(node, live, live && levelOf != null ? levelOf[node] : 0, key);
                    while (it.hasNext()) {
                        enc.neighbor(it.nextInt());
                    }
                }
                long sectionBytes = enc.finish();
                long newHeaderOffset = writer.position();
                writer.write(headerBytes);
                writer.writeLong(newHeaderOffset);
                writer.writeInt(AbstractGraphIndexWriter.FOOTER_MAGIC);
                writer.flush();
                scope.onProgress(n, n);
                long ms = (System.nanoTime() - t0) / 1_000_000L;
                log.info("Token stream: {} nodes ({} live), {} edges, {} bytes ({} encoding, {} B/edge; raw-equivalent {} bytes) in {} ms",
                        enc.nodes(), enc.liveNodes(), enc.edges(), sectionBytes, NodeTokenStream.encodingName(enc.encoding()),
                        String.format("%.2f", enc.edges() == 0 ? 0.0 : (double) sectionBytes / enc.edges()),
                        NodeTokenStream.rawEquivalentBytes(enc.nodes(), enc.edges()), ms);
                return new Result(enc.nodes(), enc.liveNodes(), enc.edges(), sectionBytes, ms, false);
            }
        }
    }
}

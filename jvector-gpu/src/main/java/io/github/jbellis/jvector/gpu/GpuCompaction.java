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

package io.github.jbellis.jvector.gpu;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexCompactor;
import io.github.jbellis.jvector.graph.disk.OrdinalMapper;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Compaction by rebuilding: the live vectors of on-disk segments are read back from their inline vectors and the
 * merged graph is built from scratch on the GPU, with {@link TornadoGraphBuildAccelerator}, instead of being stitched
 * together from the segments' graphs by {@link OnDiskGraphIndexCompactor}. The inputs are the compactor's, and the
 * output is an ordinary on-disk index with inline vectors.
 * <p>
 * The merged graph gets the sources' max degrees (a hierarchy when they have more than one layer, otherwise as
 * {@code GraphIndexBuilderConfig} says); {@code beamWidth}, {@code neighborOverflow} and
 * {@code alpha} are the builder's settings (by default efConstruction 100, overflow 1.2, alpha 1.2).
 * <p>
 * The rebuild is in-core: the merged set must fit in device memory (its vectors in FP16) and its build structures in
 * host memory. {@link #compact} decides from the sizes alone, before it reads any vector, and otherwise runs
 * {@link OnDiskGraphIndexCompactor}, which merges out of core. The live vectors are streamed to a temporary raw
 * float32 file next to the output and read through a memory mapping, so they are never held on the Java heap.
 */
public final class GpuCompaction {

    private static final java.util.logging.Logger LOG = java.util.logging.Logger.getLogger(GpuCompaction.class.getName());

    private final List<OnDiskGraphIndex> sources;
    private final List<FixedBitSet> liveNodes;
    /** Old to new ordinals per source, assigned by {@link #compact}; null before. */
    private List<OrdinalMapper> remappers;
    private final VectorSimilarityFunction similarity;
    private int beamWidth = 100;
    private float neighborOverflow = 1.2f;
    private float alpha = 1.2f;

    /**
     * As {@link OnDiskGraphIndexCompactor}, the compaction assigns the new ordinals; read them from
     * {@link #ordinalMappers()} after {@link #compact}.
     *
     * @param sources    on-disk indexes with inline vectors
     * @param liveNodes  per source, the nodes to keep
     * @param similarity the similarity function of the sources
     */
    public GpuCompaction(List<OnDiskGraphIndex> sources, List<FixedBitSet> liveNodes, VectorSimilarityFunction similarity) {
        if (sources.isEmpty() || sources.size() != liveNodes.size()) {
            throw new IllegalArgumentException("sources and liveNodes must be non-empty and of the same size");
        }
        for (var source : sources) {
            if (!source.getFeatures().containsKey(FeatureId.INLINE_VECTORS)) {
                throw new IllegalArgumentException("every source needs inline vectors");
            }
            if (source.getDimension() != sources.get(0).getDimension()) {
                throw new IllegalArgumentException("sources have different dimensions");
            }
        }
        this.sources = sources;
        this.liveNodes = liveNodes;
        this.similarity = similarity;
    }

    public GpuCompaction withBeamWidth(int beamWidth) {
        this.beamWidth = beamWidth;
        return this;
    }

    public GpuCompaction withNeighborOverflow(float neighborOverflow) {
        this.neighborOverflow = neighborOverflow;
        return this;
    }

    public GpuCompaction withAlpha(float alpha) {
        this.alpha = alpha;
        return this;
    }

    /**
     * Writes the merged index to {@code outputPath}: rebuilt on the GPU if the merged set fits (see the class
     * documentation), else by the CPU compactor. On the GPU the live nodes are numbered in source order, then node
     * order; the CPU compactor numbers them by region. Either way {@link #ordinalMappers()} gives the mapping.
     */
    public void compact(Path outputPath) throws IOException {
        int n = liveNodes.stream().mapToInt(FixedBitSet::cardinality).sum();
        int dim = sources.get(0).getDimension();
        int maxDegree = maxDegrees().get(0);
        String declined = declineReason(n, dim, maxDegree);
        if (declined != null) {
            LOG.info(() -> "GPU compaction declined, merging with OnDiskGraphIndexCompactor: " + declined);
            var compactor = new OnDiskGraphIndexCompactor(sources, liveNodes, similarity, null);
            compactor.compact(outputPath);
            remappers = compactor.ordinalMappers();
            return;
        }
        remappers = sequentialMappers();
        Path vectorsFile = outputPath.resolveSibling(outputPath.getFileName() + ".live-vectors.tmp");
        try {
            writeLiveVectors(vectorsFile);
            try (var vectors = new MappedVectors(vectorsFile, dim, n)) {
                var builder = GraphIndexBuilder.builder(vectors, similarity, maxDegrees())
                        .withBeamWidth(beamWidth)
                        .withNeighborOverflow(neighborOverflow)
                        .withAlpha(alpha)
                        .build();
                new TornadoGraphBuildAccelerator().build(builder, vectors, similarity);
                builder.cleanup();
                OnDiskGraphIndex.write(builder.getGraph(), vectors, outputPath);
            }
        } finally {
            Files.deleteIfExists(vectorsFile);
        }
    }

    /**
     * Why the GPU rebuild cannot take this merge, or null if it can, decided from the sizes alone: the build's host
     * memory (see {@link HostMemory}; the live vectors are memory-mapped, so they are not counted), then the device.
     */
    static String declineReason(int n, int dim, int maxDegree) {
        return declineReason(n, dim, maxDegree, HostMemory.shortfall(n, dim, maxDegree, true));
    }

    /** As {@link #declineReason(int, int, int)}, with the host-memory check's result given. */
    static String declineReason(int n, int dim, int maxDegree, String hostShortfall) {
        if (hostShortfall != null) {
            return n + " live nodes: the build " + hostShortfall;
        }
        if (!TornadoGraphBuildAccelerator.supports(n, dim, maxDegree, true)) {
            return String.format(java.util.Locale.ROOT, "the GPU build does not take %d x %d (no device, too little device memory, or degree %d)",
                    n, dim, maxDegree);
        }
        return null;
    }

    /** Per source, old to new ordinals of the merged index; available once {@link #compact} has run. */
    public List<OrdinalMapper> ordinalMappers() {
        if (remappers == null) {
            throw new IllegalStateException("ordinalMappers() is available after compact()");
        }
        return remappers;
    }

    /** The live nodes numbered 0..n-1: sources in order, nodes in order within each source. */
    private List<OrdinalMapper> sequentialMappers() {
        var mappers = new java.util.ArrayList<OrdinalMapper>(sources.size());
        int next = 0;
        for (var live : liveNodes) {
            var oldToNew = new java.util.HashMap<Integer, Integer>();
            for (int node = live.nextSetBit(0); node != DocIdSetIterator.NO_MORE_DOCS; node = live.nextSetBit(node + 1)) {
                oldToNew.put(node, next++);
            }
            mappers.add(new OrdinalMapper.MapMapper(oldToNew));
        }
        return mappers;
    }

    /**
     * Writes the live vectors of all sources to {@code file} as raw float32 rows, at the ordinals the GPU rebuild
     * gives them (source order, then node order). Sources are written in parallel, each to its own range of the file.
     */
    void writeLiveVectors(Path file) throws IOException {
        int dim = sources.get(0).getDimension();
        long rowBytes = (long) dim * Float.BYTES;
        long[] firstRow = new long[sources.size() + 1];
        for (int s = 0; s < sources.size(); s++) {
            firstRow[s + 1] = firstRow[s] + liveNodes.get(s).cardinality();
        }
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            IntStream.range(0, sources.size()).parallel().forEach(s -> {
                var live = liveNodes.get(s);
                int rowsPerBuffer = (int) Math.max(1, (4L << 20) / rowBytes);
                ByteBuffer buffer = ByteBuffer.allocate((int) (rowsPerBuffer * rowBytes)).order(ByteOrder.LITTLE_ENDIAN);
                long position = firstRow[s] * rowBytes;
                try (var view = sources.get(s).getView()) {
                    for (int node = live.nextSetBit(0); node != DocIdSetIterator.NO_MORE_DOCS; node = live.nextSetBit(node + 1)) {
                        VectorFloat<?> v = view.getVector(node);
                        for (int d = 0; d < dim; d++) {
                            buffer.putFloat(v.get(d));
                        }
                        if (!buffer.hasRemaining()) {
                            position += flush(channel, buffer, position);
                        }
                    }
                    flush(channel, buffer, position);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        }
    }

    /** Writes the buffer's contents at {@code position}, clears it, and returns the bytes written. */
    private static long flush(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        buffer.flip();
        long written = 0;
        while (buffer.hasRemaining()) {
            written += channel.write(buffer, position + written);
        }
        buffer.clear();
        return written;
    }

    /** The max degrees of the source with the most layers, as {@link OnDiskGraphIndexCompactor} uses. */
    private List<Integer> maxDegrees() {
        return sources.stream().map(OnDiskGraphIndex::maxDegrees).max((a, b) -> Integer.compare(a.size(), b.size())).orElseThrow();
    }
}

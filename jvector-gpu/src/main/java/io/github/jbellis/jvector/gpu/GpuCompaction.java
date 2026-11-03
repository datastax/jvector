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
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
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
 * {@code alpha} are the builder's settings (by default efConstruction 100, overflow 1.2, alpha 1.2). When the GPU
 * build is not available for the merged set, {@link #compact} falls back to {@link OnDiskGraphIndexCompactor}.
 */
public final class GpuCompaction {

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
     * Writes the merged index to {@code outputPath}: rebuilt on the GPU if possible, else by the CPU
     * compactor. On the GPU the live nodes are numbered in source order, then node order; the CPU
     * compactor numbers them by region. Either way {@link #ordinalMappers()} gives the mapping.
     */
    public void compact(Path outputPath) throws IOException {
        remappers = sequentialMappers();
        RandomAccessVectorValues vectors = liveVectors();
        var builder = GraphIndexBuilder.builder(vectors, similarity, maxDegrees())
                .withBeamWidth(beamWidth)
                .withNeighborOverflow(neighborOverflow)
                .withAlpha(alpha)
                .build();
        var accelerator = new TornadoGraphBuildAccelerator();
        if (!accelerator.supports(builder, vectors, similarity)) {
            var compactor = new OnDiskGraphIndexCompactor(sources, liveNodes, similarity, null);
            compactor.compact(outputPath);
            remappers = compactor.ordinalMappers();
            return;
        }
        accelerator.build(builder, vectors, similarity);
        builder.cleanup();
        OnDiskGraphIndex.write(builder.getGraph(), vectors, outputPath);
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

    /** The live vectors of all sources, at the ordinals the GPU rebuild gives them (source order, then node order). */
    RandomAccessVectorValues liveVectors() {
        var sequential = sequentialMappers();
        int total = liveNodes.stream().mapToInt(FixedBitSet::cardinality).sum();
        VectorFloat<?>[] vectors = new VectorFloat<?>[total];
        IntStream.range(0, sources.size()).parallel().forEach(s -> {
            var live = liveNodes.get(s);
            var mapper = sequential.get(s);
            try (var view = sources.get(s).getView()) {
                for (int node = live.nextSetBit(0); node != DocIdSetIterator.NO_MORE_DOCS; node = live.nextSetBit(node + 1)) {
                    int ordinal = mapper.oldToNew(node);
                    if (ordinal < 0 || ordinal >= total) {
                        throw new IllegalArgumentException("new ordinals of the live nodes must be in 0.." + (total - 1) + ", got " + ordinal);
                    }
                    vectors[ordinal] = view.getVector(node);
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
        if (Arrays.stream(vectors).anyMatch(v -> v == null)) {
            throw new IllegalArgumentException("new ordinals of the live nodes must be distinct");
        }
        return new ListRandomAccessVectorValues(Arrays.asList(vectors), sources.get(0).getDimension());
    }

    /** The max degrees of the source with the most layers, as {@link OnDiskGraphIndexCompactor} uses. */
    private List<Integer> maxDegrees() {
        return sources.stream().map(OnDiskGraphIndex::maxDegrees).max((a, b) -> Integer.compare(a.size(), b.size())).orElseThrow();
    }
}

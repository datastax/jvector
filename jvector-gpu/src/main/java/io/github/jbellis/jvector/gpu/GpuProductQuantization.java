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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.util.Random;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.quantization.ImmutablePQVectors;
import io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Product quantization on the GPU, returning JVector's own {@link ProductQuantization} and {@link PQVectors}:
 * <ul>
 * <li>{@link #compute} trains the codebooks like {@code ProductQuantization.compute} (unweighted): the same subspaces,
 * training sample (at most {@value ProductQuantization#MAX_PQ_TRAINING_SET_SIZE} vectors), optional global centering,
 * k-means++ seeding and {@value ProductQuantization#K_MEANS_ITERATIONS} Lloyd iterations, for all subspaces at once;</li>
 * <li>{@link #encodeAll} encodes vectors in batches with any unweighted {@link ProductQuantization}, including one
 * trained on the CPU.</li>
 * </ul>
 * Both run under the TornadoVM launcher, like {@link TornadoGraphBuildAccelerator}. Without the TornadoVM runtime, or
 * for subspaces wider than {@value PqKernels#MAX_SUBSPACE} dimensions, they run JVector's CPU implementation instead.
 */
public final class GpuProductQuantization {

    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    /** {@code ProductQuantization}'s serialization header. */
    private static final int PQ_MAGIC = 0x75EC4012;
    private static final int PQ_VERSION = 3;
    private static final int LOCAL = 256;
    /** Rows encoded per GPU batch. */
    static final int ENCODE_BATCH = 131_072;

    private GpuProductQuantization() {
    }

    /**
     * Trains PQ codebooks on the GPU.
     *
     * @param M              number of subspaces
     * @param clusterCount   centroids per subspace (at most 256)
     * @param globallyCenter subtract the training set's centroid first
     */
    public static ProductQuantization compute(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter) {
        int dim = ravv.dimension();
        int[][] subspaces = subvectorSizesAndOffsets(dim, M);
        if (subspaces[0][0] > PqKernels.MAX_SUBSPACE || !runtimeAvailable() || trainOnCpu()) {
            // too wide for the kernels' registers, no GPU, or training asked for on the CPU
            return ProductQuantization.compute(ravv, M, clusterCount, globallyCenter);
        }
        int[] sample = trainingSample(ravv.size(), new Random(42));
        int t = sample.length;
        FloatArray rows = new FloatArray(t * dim);
        var supplier = ravv.threadLocalSupplier();
        MemorySegment rowSegment = rows.getSegment();
        IntStream.range(0, t).parallel().forEach(i -> HostRows.copyRow(supplier.get().getVector(sample[i]), rowSegment, i, dim));
        float[] centroid = null;
        if (globallyCenter) {
            double[] sum = IntStream.range(0, t).parallel().collect(() -> new double[dim], (acc, i) -> {
                for (int d = 0; d < dim; d++) {
                    acc[d] += rows.get(i * dim + d);
                }
            }, (a, b) -> {
                for (int d = 0; d < dim; d++) {
                    a[d] += b[d];
                }
            });
            float[] c = new float[dim];
            for (int d = 0; d < dim; d++) {
                c[d] = (float) (sum[d] / t);
            }
            IntStream.range(0, t).parallel().forEach(i -> {
                for (int d = 0; d < dim; d++) {
                    rows.set(i * dim + d, rows.get(i * dim + d) - c[d]);
                }
            });
            centroid = c;
        }

        IntArray sizes = new IntArray(M);
        IntArray offsets = new IntArray(M);
        int maxSize = 0;
        for (int m = 0; m < M; m++) {
            sizes.set(m, subspaces[m][0]);
            offsets.set(m, subspaces[m][1]);
            maxSize = Math.max(maxSize, subspaces[m][0]);
        }
        FloatArray codebooks = new FloatArray(clusterCount * dim);
        FloatArray d2 = new FloatArray(M * t);
        IntArray assignment = new IntArray(M * t);
        IntArray picked = new IntArray(M);
        IntArray params = IntArray.fromElements(t, dim, M, clusterCount, PqKernels.NONE, 0, 0x5EED, maxSize);
        KernelContext ctx = new KernelContext();
        TaskGraph tg = new TaskGraph("pq")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, rows, sizes, offsets, codebooks, d2, assignment, picked)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, params)
                .task("dist", PqKernels::seedDistances, ctx, rows, codebooks, sizes, offsets, params, d2)
                .task("pick", PqKernels::pick, ctx, d2, params, picked)
                .task("set", PqKernels::setCentroid, ctx, rows, codebooks, sizes, offsets, params, picked)
                .task("assign", PqKernels::assign, ctx, rows, codebooks, sizes, offsets, params, assignment)
                .task("update", PqKernels::update, ctx, rows, codebooks, sizes, offsets, params, assignment)
                .transferToHost(DataTransferMode.UNDER_DEMAND, codebooks);
        GridScheduler grids = new GridScheduler();
        grids.addWorkerGrid("pq.dist", grid((long) M * t, LOCAL));
        grids.addWorkerGrid("pq.pick", grid((long) M * PqKernels.PICK_THREADS, PqKernels.PICK_THREADS));
        grids.addWorkerGrid("pq.set", grid((long) M * maxSize, LOCAL));
        grids.addWorkerGrid("pq.assign", grid((long) M * t, LOCAL));
        grids.addWorkerGrid("pq.update", grid((long) M * clusterCount, LOCAL));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withGridScheduler(grids);
            // k-means++: pick a row (uniformly, then by squared distance), make it the next centroid, update distances
            for (int p = 0; p < clusterCount; p++) {
                params.set(PqKernels.P_PICK, p);
                for (int step : new int[] { PqKernels.SEED_PICK, PqKernels.SEED_SET, PqKernels.SEED_DISTANCES }) {
                    params.set(PqKernels.P_STEP, step);
                    plan.execute();
                }
            }
            params.set(PqKernels.P_STEP, PqKernels.NONE);
            var result = plan.execute(); // a no-op step; the codebooks are fetched from the last result
            for (int iter = 0; iter < ProductQuantization.K_MEANS_ITERATIONS; iter++) {
                for (int step : new int[] { PqKernels.ASSIGN, PqKernels.UPDATE }) {
                    params.set(PqKernels.P_STEP, step);
                    result = plan.execute();
                }
            }
            result.transferToHost(codebooks);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return toProductQuantization(codebooks, subspaces, clusterCount, centroid);
    }

    /**
     * Whether {@code -Djvector.gpu.pq.train=cpu} asks for JVector's CPU training. Training uses a sample of at most
     * 128k vectors, so its cost hardly grows with the dataset, and a many-core CPU can train faster than a mid-range
     * GPU (64 cores train ada002 M=192 in 3.8 s, an A10 in 6.0 s; an RTX 4090 in 2.0 s against 32 threads' 7.0 s).
     * Encoding, which grows with the dataset, stays on the GPU.
     */
    static boolean trainOnCpu() {
        return "cpu".equalsIgnoreCase(System.getProperty("jvector.gpu.pq.train", "gpu"));
    }

    /**
     * Encodes every vector with {@code pq} on the GPU.
     *
     * @throws IllegalArgumentException for an anisotropic (weighted) {@code pq}, which needs the CPU path
     */
    public static PQVectors encodeAll(ProductQuantization pq, RandomAccessVectorValues ravv) {
        Codebooks cb = codebooksOf(pq);
        if (java.util.Arrays.stream(cb.subspaces).anyMatch(sub -> sub[0] > PqKernels.MAX_SUBSPACE) || !runtimeAvailable()) {
            return pq.encodeAll(ravv, io.github.jbellis.jvector.util.PhysicalCoreExecutor.pool()); // subspaces wider than the kernels hold in registers
        }
        int n = ravv.size();
        int dim = ravv.dimension();
        int M = cb.subspaces.length;
        int batch = Math.min(ENCODE_BATCH, n);
        IntArray sizes = new IntArray(M);
        IntArray offsets = new IntArray(M);
        for (int m = 0; m < M; m++) {
            sizes.set(m, cb.subspaces[m][0]);
            offsets.set(m, cb.subspaces[m][1]);
        }
        FloatArray codebooks = FloatArray.fromArray(cb.codebooks);
        FloatArray centroid = FloatArray.fromArray(cb.centroid != null ? cb.centroid : new float[dim]);
        FloatArray rows = new FloatArray(batch * dim);
        ByteArray codes = new ByteArray(batch * M);
        IntArray params = IntArray.fromElements(batch, dim, M, cb.clusterCount, 0, 0, 0, 0);
        KernelContext ctx = new KernelContext();
        TaskGraph tg = new TaskGraph("pqEncode")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, codebooks, centroid, sizes, offsets)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, rows, params)
                .task("encode", PqKernels::encode, ctx, rows, centroid, codebooks, sizes, offsets, params, codes)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, codes);
        WorkerGrid encodeGrid = grid((long) batch * M, LOCAL);
        GridScheduler grids = new GridScheduler("pqEncode.encode", encodeGrid);

        var layout = new PQVectors.PQLayout(n, M);
        ByteSequence<?>[] chunks = new ByteSequence<?>[layout.totalChunks];
        for (int c = 0; c < layout.totalChunks; c++) {
            chunks[c] = VTS.createByteSequence(c < layout.fullSizeChunks ? layout.fullChunkBytes : layout.lastChunkBytes);
        }
        var supplier = ravv.threadLocalSupplier();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            for (int from = 0; from < n; from += batch) {
                int size = Math.min(batch, n - from);
                int first = from;
                MemorySegment rowSegment = rows.getSegment();
                IntStream.range(0, size).parallel().forEach(i -> HostRows.copyRow(supplier.get().getVector(first + i), rowSegment, i, dim));
                params.set(PqKernels.P_ROWS, size);
                encodeGrid.setGlobalWork(roundUp((long) size * M, LOCAL), 1, 1);
                plan.withGridScheduler(grids).execute();
                // codes of consecutive ordinals are contiguous in a chunk too: copy each run within a chunk at once
                MemorySegment codeSegment = codes.getSegment();
                int last = first + size;
                IntStream.iterate(first, o -> o < last, o -> (o / layout.fullChunkVectors + 1) * layout.fullChunkVectors).parallel().forEach(o -> {
                    int end = Math.min(last, (o / layout.fullChunkVectors + 1) * layout.fullChunkVectors);
                    ByteSequence<?> chunk = chunks[o / layout.fullChunkVectors];
                    copyCodes(codeSegment, (long) (o - first) * M, chunk, (o % layout.fullChunkVectors) * M, (end - o) * M);
                });
            }
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return new ImmutablePQVectors(pq, chunks, n, layout.fullChunkVectors);
    }

    /** Copies {@code length} code bytes into a chunk at {@code to}, in bulk when its storage allows. */
    private static void copyCodes(MemorySegment codes, long from, ByteSequence<?> chunk, int to, int length) {
        Object backing = chunk.get();
        if (backing instanceof byte[] array) {
            MemorySegment.copy(codes, ValueLayout.JAVA_BYTE, from, array, chunk.offset() + to, length);
        } else if (backing instanceof MemorySegment segment) {
            MemorySegment.copy(codes, from, segment, chunk.offset() + to, length);
        } else {
            for (int b = 0; b < length; b++) {
                chunk.set(to + b, codes.get(ValueLayout.JAVA_BYTE, from + b));
            }
        }
    }

    /** {@code ProductQuantization}'s subspaces: sizes differ by at most one, the larger ones first. */
    static int[][] subvectorSizesAndOffsets(int dim, int M) {
        int[][] sizes = new int[M][];
        int base = dim / M;
        int remainder = dim % M;
        int offset = 0;
        for (int i = 0; i < M; i++) {
            int size = base + (i < remainder ? 1 : 0);
            sizes[i] = new int[] { size, offset };
            offset += size;
        }
        return sizes;
    }

    /** All ordinals, or a random sample of {@code MAX_PQ_TRAINING_SET_SIZE} (Floyd's algorithm). */
    static int[] trainingSample(int n, Random random) {
        int max = ProductQuantization.MAX_PQ_TRAINING_SET_SIZE;
        if (n <= max) {
            return IntStream.range(0, n).toArray();
        }
        var chosen = new java.util.HashSet<Integer>(max * 2);
        for (int j = n - max; j < n; j++) {
            int r = random.nextInt(j + 1);
            chosen.add(chosen.contains(r) ? j : r);
        }
        return chosen.stream().mapToInt(Integer::intValue).sorted().toArray();
    }

    record Codebooks(int[][] subspaces, int clusterCount, float[] codebooks, float[] centroid) {
    }

    /** Builds a JVector {@link ProductQuantization} through its own serialization format. */
    static ProductQuantization toProductQuantization(FloatArray codebooks, int[][] subspaces, int clusterCount, float[] centroid) {
        int dim = codebooks.getSize() / clusterCount;
        ByteBuffer buffer = ByteBuffer.allocate(64 + 4 * subspaces.length + 4 * (centroid == null ? 0 : dim) + 4 * clusterCount * dim);
        buffer.putInt(PQ_MAGIC).putInt(PQ_VERSION);
        buffer.putInt(centroid == null ? 0 : dim);
        if (centroid != null) {
            for (float c : centroid) {
                buffer.putFloat(c);
            }
        }
        buffer.putInt(subspaces.length);
        for (int[] s : subspaces) {
            buffer.putInt(s[0]);
        }
        buffer.putFloat(KMeansPlusPlusClusterer.UNWEIGHTED);
        buffer.putInt(clusterCount);
        for (int i = 0; i < clusterCount * dim; i++) {
            buffer.putFloat(codebooks.get(i));
        }
        buffer.flip();
        try {
            return ProductQuantization.load(new ByteBufferReader(buffer));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Reads a {@link ProductQuantization}'s codebooks back from its serialization format. */
    static Codebooks codebooksOf(ProductQuantization pq) {
        int dim = pq.getOriginalDimension();
        ByteBuffer buffer = ByteBuffer.allocate(64 + 4 * (pq.getSubspaceCount() + dim + pq.getClusterCount() * dim));
        try {
            pq.write(new ByteBufferIndexWriter(buffer), OnDiskGraphIndex.CURRENT_VERSION);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        buffer.flip();
        if (buffer.getInt() != PQ_MAGIC) {
            throw new IllegalStateException("unexpected ProductQuantization format");
        }
        buffer.getInt(); // version
        int centroidLength = buffer.getInt();
        float[] centroid = null;
        if (centroidLength > 0) {
            centroid = new float[centroidLength];
            for (int d = 0; d < centroidLength; d++) {
                centroid[d] = buffer.getFloat();
            }
        }
        int M = buffer.getInt();
        int[][] subspaces = new int[M][];
        int offset = 0;
        for (int m = 0; m < M; m++) {
            int size = buffer.getInt();
            subspaces[m] = new int[] { size, offset };
            offset += size;
        }
        float anisotropic = buffer.getFloat();
        if (anisotropic != KMeansPlusPlusClusterer.UNWEIGHTED) {
            throw new IllegalArgumentException("GPU encoding supports unweighted PQ only");
        }
        int clusterCount = buffer.getInt();
        float[] codebooks = new float[clusterCount * offset];
        for (int i = 0; i < codebooks.length; i++) {
            codebooks[i] = buffer.getFloat();
        }
        return new Codebooks(subspaces, clusterCount, codebooks, centroid);
    }

    /**
     * Whether TornadoVM has a device to run on (the JVM was started by the {@code tornado} launcher, or with its
     * argfile). Loading the runtime class is not enough: TornadoVM creates its backends lazily (8.0), so the class
     * loads even without a backend, and the first device request then fails.
     */
    static boolean runtimeAvailable() {
        try {
            var runtime = uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider.getTornadoRuntime();
            return runtime.getNumBackends() > 0 && runtime.getDefaultDevice() != null;
        } catch (Throwable t) {
            return false;
        }
    }

    private static WorkerGrid grid(long global, int local) {
        WorkerGrid g = new WorkerGrid1D(Math.toIntExact(roundUp(global, local)));
        g.setLocalWork(local, 1, 1);
        return g;
    }

    private static long roundUp(long n, int multiple) {
        return (n + multiple - 1) / multiple * multiple;
    }
}

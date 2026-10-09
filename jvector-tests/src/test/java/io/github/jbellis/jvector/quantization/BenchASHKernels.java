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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import org.openjdk.jmh.annotations.*;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Kernel-only comparison: identical codes/query, 32 scores per invocation, no training
 * or query preparation in timed work. Both paths traverse many blocks in permuted order.
 * Each representation is at least workingSetMiB; choose a size larger than the machine's
 * last-level cache for a cold-data experiment. Smaller sets measure cache-resident work.
 *
 * <p>Use JMH forks (never -f 0): tuning is fixed when the backend is initialized.
 * The default is one worker; -t N measures aggregate throughput over shared read-only
 * codes with independent worker cursors and output buffers.</p>
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 2, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", "-Xmx8g"})
@Threads(1)
public class BenchASHKernels {
    @Param({"64", "384", "768"}) public int dimensions;
    @Param({"2", "4"}) public int bits;
    @Param({"1", "2", "4", "8"}) public int accumulators;
    @Param({"1", "2", "4"}) public int unroll;
    @Param({"1024"}) public int workingSetMiB;

    private VectorUtilSupport backend;
    private byte[][] canonical;
    private byte[][] packed;
    private float[] query, lut;
    private int groups;
    private boolean tunedProjection;

    @State(Scope.Thread)
    public static class Worker {
        int cursor;
        final float[] scores = new float[32];

        @Setup(Level.Trial)
        public void setup(org.openjdk.jmh.infra.ThreadParams thread) {
            cursor = thread.getThreadIndex() * 15485863;
        }
    }

    @Setup(Level.Trial)
    public void setup() {
        System.setProperty("jvector.ash.lut.accumulators", Integer.toString(accumulators));
        System.setProperty("jvector.ash.lut.unroll", Integer.toString(unroll));
        System.setProperty("jvector.ash.projection.accumulators", Integer.toString(accumulators));
        System.setProperty("jvector.ash.projection.unroll", Integer.toString(unroll));
        backend = VectorizationProvider.getInstance().getVectorUtilSupport();
        tunedProjection = backend.usesAshProjectionTuning();
        if (!backend.supportsAshLutScoring() || !backend.supportsAshProjectionScoring()) {
            throw new IllegalStateException("This benchmark requires the Panama ASH SIMD backend");
        }
        if (!backend.ashKernelDescription().contains("LUT accumulators=" + accumulators + ", unroll=" + unroll)) {
            throw new IllegalStateException("Tuning already initialized: run each parameter combination in its own JMH fork");
        }
        if (workingSetMiB <= 0) throw new IllegalArgumentException("workingSetMiB must be positive");
        groups = FusedASHLayout.codeGroups(dimensions, bits);
        int bytes = FusedASHLayout.canonicalCodeBytes(dimensions, bits);
        long target = (long) workingSetMiB * 1024 * 1024;
        int blocks = 1;
        while ((long) blocks * bytes * 32 < target) blocks = Math.multiplyExact(blocks, 2);
        canonical = new byte[Math.multiplyExact(blocks, 32)][bytes];
        packed = new byte[blocks][Math.multiplyExact(bytes, 32)];
        Random random = new Random(89712);
        for (int b = 0; b < blocks; b++) {
            for (int lane = 0; lane < 32; lane++) {
                byte[] code = canonical[b * 32 + lane];
                random.nextBytes(code);
                for (int i = 0; i < bytes; i++) packed[b][i * 32 + lane] = code[i];
            }
        }
        query = new float[dimensions];
        for (int i = 0; i < dimensions; i++) query[i] = random.nextFloat() - 0.5f;
        lut = new float[Math.multiplyExact(groups, 16)];
        FusedASHLayout.buildQueryLut(query, dimensions, bits, lut);
        System.out.println(backend.ashKernelDescription() + "; code bytes per representation=" + (long) blocks * bytes * 32);
    }

    private int nextBlock(Worker worker) {
        worker.cursor = (worker.cursor + 104729) & (packed.length - 1);
        return worker.cursor;
    }

    @Benchmark
    @OperationsPerInvocation(32)
    public float single(Worker worker) {
        int base = nextBlock(worker) * 32;
        float sum = 0;
        for (int i = 0; i < 32; i++) {
            sum += tunedProjection
                    ? backend.ashProjectionDotTuned(query, canonical[base + i], dimensions, bits)
                    : backend.ashProjectionDot(query, canonical[base + i], dimensions, bits);
        }
        return sum;
    }

    @Benchmark
    @OperationsPerInvocation(32)
    public float block(Worker worker) {
        backend.ashLutScore(packed[nextBlock(worker)], 0, groups, 32, 0, 32, lut, worker.scores, 0);
        float sum = 0;
        for (float score : worker.scores) sum += score;
        return sum;
    }
}

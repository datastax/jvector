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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Guards the Panama provider against vectors boxed on the heap. A Vector API value passed to or returned from a
 * method that C2 does not inline becomes a heap object, and in some JVMs a large or deep caller left such calls in
 * place, slowing PQ builds about 2x. The rules checked here: no method takes or returns vectors, and the kernels
 * listed stay above C2's inline limit so they are compiled on their own. The allocation test then measures realistic
 * callers at several depths.
 */
class PanamaNoBoxingTest {
    private static final int FREQ_INLINE_SIZE = 325;    // C2 FreqInlineSize: never inlined above this

    private static final Class<?>[] PANAMA_CLASSES = {PanamaVectorUtilSupport.class, PanamaASHKernels.class,
            PanamaASHProjection.class, PanamaASHProjection.Int8.class, PanamaASHProjection.Fp16.class};

    /**
     * Wrappers that take or return vectors. No Panama kernel calls them any more; they remain only because the native
     * provider overrides them and its store test uses one. No new code may call them.
     */
    private static final java.util.Set<String> KEPT_WRAPPERS = java.util.Set.of(
            "fromVectorFloat", "intoVectorFloat", "fromByteSequence", "intoByteSequence");

    /** Kernels too large to inline, so always compiled on their own (for overloads, the largest). */
    private static final String[] BULK_KERNELS = {
            "transposedDotSums", "transposedSquareSums", "dotProduct4", "squareDistance4", "cosine4", "dotProductPreferred",
            "dotProductMulti", "tryClosestCentroid", "tryUpdateKMeansDistances", "lutScore", "ashMaskedAdd8",
            "projectRows", "packedBatch", "projectTile", "halfRows", "ashInt8NeighborhoodEstimates", "ashInt8NeighborhoodWords",
            "lutNeighborhoodEstimates", "canonicalLutDot", "ashSymmetricTwoBitRows", "ashSymmetricFourBitRows",
            "ashSymmetricOffsetBinary", "ashMaskedAddFlat", "ashMaskedAdd4",
            "pairDot8", "pairDistance8", "pairDot4", "pairDistance4", "pairDot2", "pairDistance2",
            "pairPadded512", "pairPadded256", "pairPadded128", "pairPadded64", "pairTable512", "pairTable256",
    };

    @Test
    void noMethodTakesOrReturnsVectors() throws IOException {
        List<String> violations = new ArrayList<>();
        for (Class<?> c : PANAMA_CLASSES) {
            for (Method m : methods(c)) {
                if (carriesVector(m.descriptor) && !(c == PanamaVectorUtilSupport.class && KEPT_WRAPPERS.contains(m.name))) {
                    violations.add(c.getSimpleName() + "." + m.name + m.descriptor + ": " + m.codeLength + " bytecodes");
                }
            }
        }
        assertTrue(violations.isEmpty(), "No method may take or return Vector API values: a call C2 does not inline "
                + "(even a tiny one, at a call site it judges cold) boxes the vectors on the heap. Write the vector code "
                + "in the kernel that uses it instead: " + violations);
    }

    @Test
    void bulkKernelsAreCompiledOnTheirOwn() throws IOException {
        Map<String, Integer> sizes = new HashMap<>();
        for (Class<?> c : PANAMA_CLASSES) {
            for (Method m : methods(c)) sizes.merge(m.name, m.codeLength, Math::max);
        }
        List<String> violations = new ArrayList<>();
        for (String name : BULK_KERNELS) {
            Integer size = sizes.get(name);
            assertNotNull(size, name + " not found");
            if (size <= FREQ_INLINE_SIZE) violations.add(name + ": " + size + " bytecodes");
        }
        assertTrue(violations.isEmpty(), "Bulk kernels must stay above C2's FreqInlineSize (" + FREQ_INLINE_SIZE
                + " bytecodes), so they are never inlined into a deep caller where their vector operations can be "
                + "left as heap-allocating calls: " + violations);
    }

    @Test
    void transposedPartialSumsMatchScalar() {
        var panama = new PanamaVectorUtilSupport();
        var scalar = new DefaultVectorUtilSupport();
        var vectors = new ArrayVectorProvider();
        Random random = new Random(1701);
        // Even and uneven splits; cluster counts that fill whole groups of vectors, single vectors, or neither.
        int[][] splits = {{2, 2, 2, 2}, {4, 4, 4}, {8, 8, 8}, {16, 16}, {32}, {3, 3, 2}, {7, 9, 1}, {24, 25}, {1}};
        for (int[] split : splits) {
            for (int k : new int[]{256, 64, 16, 10, 100, 1}) {
                for (var vsf : new VectorSimilarityFunction[]{VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.EUCLIDEAN}) {
                    int dimension = java.util.Arrays.stream(split).sum();
                    int[][] sizesAndOffsets = new int[split.length][];
                    for (int m = 0, offset = 0; m < split.length; offset += split[m], m++) sizesAndOffsets[m] = new int[]{split[m], offset};
                    int[] sizes = java.util.Arrays.stream(sizesAndOffsets).mapToInt(x -> x[0]).toArray();
                    int[] offsets = java.util.Arrays.stream(sizesAndOffsets).mapToInt(x -> x[1]).toArray();
                    var codebooks = new VectorFloat<?>[split.length];
                    var transposed = vectors.createFloatVector(k * dimension);
                    for (int m = 0; m < split.length; m++) {
                        int size = split[m], base = k * sizesAndOffsets[m][1];
                        codebooks[m] = vectors.createFloatVector(k * size);
                        for (int c = 0; c < k; c++) {
                            for (int d = 0; d < size; d++) {
                                float x = (float) random.nextGaussian();
                                codebooks[m].set(c * size + d, x);
                                transposed.set(base + d * k + c, x);
                            }
                        }
                    }
                    var query = vectors.createFloatVector(dimension);
                    var center = vectors.createFloatVector(dimension);
                    var centered = vectors.createFloatVector(dimension);
                    for (int i = 0; i < dimension; i++) {
                        query.set(i, (float) random.nextGaussian());
                        center.set(i, (float) random.nextGaussian());
                        centered.set(i, query.get(i) - center.get(i));
                    }
                    var expected = vectors.createFloatVector(split.length * k);
                    var actual = vectors.createFloatVector(split.length * k);
                    for (int m = 0; m < split.length; m++) {
                        scalar.calculatePartialSums(codebooks[m], m, split[m], k, centered, sizesAndOffsets[m][1], vsf, expected);
                    }
                    panama.calculatePartialSumsTransposed(transposed, sizes, offsets, k, query, center, vsf, actual);
                    var fallback = vectors.createFloatVector(split.length * k);
                    scalar.calculatePartialSumsTransposed(transposed, sizes, offsets, k, query, center, vsf, fallback);
                    for (int i = 0; i < actual.length(); i++) {
                        int m = i / k, c = i % k, size = split[m], offset = sizesAndOffsets[m][1];
                        double scale = 0;
                        for (int d = 0; d < size; d++) {
                            double x = codebooks[m].get(c * size + d), y = centered.get(offset + d);
                            scale += vsf == VectorSimilarityFunction.DOT_PRODUCT ? Math.abs(x * y) : (x - y) * (x - y);
                        }
                        String where = vsf + " split=" + java.util.Arrays.toString(split) + " k=" + k + " i=" + i;
                        assertEquals(expected.get(i), actual.get(i), 1e-6 * size * scale + 1e-30, where);
                        assertEquals(expected.get(i), fallback.get(i), 1e-6 * size * scale + 1e-30, where + " (default)");
                    }
                }
            }
        }
        var v = vectors.createFloatVector(8);
        int[] sizes = {8}, offsets = {0};
        assertThrows(UnsupportedOperationException.class,
                () -> panama.calculatePartialSumsTransposed(v, sizes, offsets, 1, v, v, VectorSimilarityFunction.COSINE, v));
        assertThrows(IndexOutOfBoundsException.class,
                () -> panama.calculatePartialSumsTransposed(v, sizes, offsets, 2, v, v, VectorSimilarityFunction.DOT_PRODUCT, v));
    }

    /** The whole projection in one call gives the per-row {@code ashDotRow} results. */
    @Test
    void ashProjectionMatchesRowDots() {
        var panama = new PanamaVectorUtilSupport();
        Random random = new Random(11);
        for (int rows : new int[]{1, 7, 8, 9, 64, 100}) {
            for (int columns : new int[]{1, 15, 16, 17, 100, 384}) {
                float[][] matrix = new float[rows][columns];
                for (float[] row : matrix) for (int j = 0; j < columns; j++) row[j] = (float) random.nextGaussian();
                float[] x = new float[columns];
                for (int j = 0; j < columns; j++) x[j] = (float) random.nextGaussian();
                float[] out = new float[rows];
                panama.prepareAshProjection(matrix).project(x, out);
                // Same arithmetic as ashDotRow; the Vector API leaves the lane-reduction order open, so two compiled
                // methods can differ in the last bit.
                for (int r = 0; r < rows; r++) {
                    double scale = 0;
                    for (int j = 0; j < columns; j++) scale += Math.abs((double) matrix[r][j] * x[j]);
                    assertEquals(panama.ashDotRow(matrix[r], x), out[r], 1e-6 * scale,
                            "rows=" + rows + " columns=" + columns + " r=" + r);
                }
            }
        }
    }

    /** All rows in one call give exactly the per-row {@code ashSymmetricDotAt} results. */
    @Test
    void ashSymmetricRowsMatchPerRow() {
        var panama = new PanamaVectorUtilSupport();
        if (!panama.supportsAshSymmetricDotRows()) return;
        Random random = new Random(12);
        for (int bits = 2; bits <= 9; bits++) {
            for (int dimensions : new int[]{1, 31, 32, 33, 100, 384, 1000}) {
                int bytes = (dimensions * bits + 7) / 8;
                for (int stride : new int[]{bytes, bytes + 5}) {
                    int count = 13;
                    byte[] query = new byte[bytes + 3];
                    byte[] rows = new byte[(count - 1) * stride + bytes];
                    random.nextBytes(query);
                    random.nextBytes(rows);
                    float[] out = new float[count];
                    panama.ashSymmetricDotRows(query, 3, rows, stride, count, dimensions, bits, out);
                    for (int i = 0; i < count; i++) {
                        assertEquals(panama.ashSymmetricDotAt(query, 3, rows, i * stride, dimensions, bits), out[i],
                                "bits=" + bits + " d=" + dimensions + " stride=" + stride + " row=" + i);
                    }
                }
            }
        }
    }

    /**
     * A whole INT8 field-block neighborhood in one call gives the native kernel's result: the integer lane dot, then
     * {@code fma(scale, dot, landmarkDot) + offset} from the lane's headers, on the DOT_PRODUCT or EUCLIDEAN scale.
     */
    @Test
    void ashInt8NeighborhoodMatchesPerLane() {
        var panama = new PanamaVectorUtilSupport();
        if (!panama.supportsAshInt8NeighborhoodScoring()) return;
        Random random = new Random(13);
        for (int bits : new int[]{3, 5, 6, 7, 8, 9}) {
            for (int dimensions : new int[]{8, 100, 384}) {
                for (int blockSize : new int[]{8, 16, 32}) {
                    for (int capacity : new int[]{1, blockSize, 32, 33}) {
                        int units = (dimensions - 1) / 8 + 1;
                        int body = units * bits * blockSize, blockBytes = body + 5 * blockSize;
                        int blocks = (capacity - 1) / blockSize + 1;
                        byte[] codes = new byte[blocks * blockBytes];
                        random.nextBytes(codes);
                        for (int b = 0; b < blocks; b++) {
                            for (int lane = 0; lane < blockSize; lane++) {
                                io.github.jbellis.jvector.quantization.FusedASHLayout.writeLaneHeader(codes, b * blockBytes,
                                        lane, dimensions, bits, blockSize, (float) random.nextGaussian(),
                                        (float) random.nextGaussian(), (byte) 0);
                            }
                        }
                        byte[] q = new byte[units * 8];
                        int[] wide = new int[units * 8];
                        long qSum = 0;
                        for (int j = 0; j < dimensions; j++) {
                            q[j] = (byte) (random.nextInt(255) - 127);
                            wide[j] = q[j];
                            qSum += q[j];
                        }
                        float invScale = 0.013f, center = (float) (((1 << bits) - 1) * 0.5 * qSum), landmarkDot = 0.37f;
                        float perEstimate = -0.5f;
                        float[] dot = new float[capacity], euclidean = new float[capacity];
                        panama.ashScoreNeighborhoodInt8(codes, bits, units, blockSize, capacity, q, invScale, center,
                                landmarkDot, dot);
                        panama.ashScoreNeighborhoodInt8Euclidean(codes, bits, units, blockSize, capacity, q, invScale,
                                center, landmarkDot, perEstimate, euclidean);
                        for (int n = 0; n < capacity; n++) {
                            int b = n / blockSize, lane = n % blockSize, offset = b * blockBytes;
                            long sum = 0;
                            for (int u = 0; u < units; u++) {
                                for (int k = 0; k < 8; k++) {
                                    int bit = k * bits, at = offset + (u * blockSize + lane) * bits + bit / 8;
                                    int window = (codes[at] & 0xFF) | (bit / 8 + 1 < bits ? (codes[at + 1] & 0xFF) << 8 : 0);
                                    sum += (long) wide[u * 8 + k] * ((window >>> (bit & 7)) & ((1 << bits) - 1));
                                }
                            }
                            float laneDot = ((float) sum - center) * invScale;
                            float e = Math.fma(io.github.jbellis.jvector.quantization.FusedASHLayout.readScale(codes, offset,
                                    lane, dimensions, bits, blockSize), laneDot, landmarkDot)
                                    + io.github.jbellis.jvector.quantization.FusedASHLayout.readOffset(codes, offset, lane,
                                    dimensions, bits, blockSize);
                            String where = "bits=" + bits + " d=" + dimensions + " block=" + blockSize + " n=" + n;
                            assertEquals(Math.fma(e, 0.5f, 0.5f), dot[n], where);
                            assertEquals(1f / (1f + Math.max(perEstimate * e, 0f)), euclidean[n], where);
                        }
                        // A lane that names a landmark other than the single one is rejected
                        codes[body + 4 * blockSize] = 1;
                        float[] out = new float[capacity];
                        assertThrows(IllegalStateException.class, () -> panama.ashScoreNeighborhoodInt8(codes, bits, units,
                                blockSize, capacity, q, invScale, center, landmarkDot, out));
                    }
                }
            }
        }
    }

    /**
     * Every case of {@link PanamaAllocationProbe}, each in a fresh JVM on the Panama provider: after warm-up, the
     * bytes allocated per operation stay within what the operation allocates by design. It starts about 80 JVMs
     * (around 12 minutes), so it runs only with {@code -Djvector.test.allocationProbe=true}. One JVM per case is
     * evidence, not proof: a kernel can box in only some JVMs (by compile order), so run a suspect case in several.
     */
    @Test
    @org.junit.jupiter.api.condition.EnabledIfSystemProperty(named = "jvector.test.allocationProbe", matches = "true")
    void realisticCallersDoNotAllocateVectors() throws Exception {
        String java = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";
        String classPath = System.getProperty("java.class.path");
        List<String> failures = new ArrayList<>();
        for (String name : PanamaAllocationProbe.caseNames()) {
            var command = List.of(java, "--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                    "-Djvector.vectorization_provider=panama", "-Dprobe.case=" + name, "-Xmx2g",
                    "-cp", classPath, PanamaAllocationProbe.class.getName());
            Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
            StringBuilder output = new StringBuilder();
            String result = null;
            try (var reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                for (String line; (line = reader.readLine()) != null; ) {
                    output.append(line).append('\n');
                    if (line.startsWith("RESULT ")) result = line;
                }
            }
            assertTrue(process.waitFor(5, TimeUnit.MINUTES), name + " timed out");
            assertEquals(0, process.exitValue(), name + " failed:\n" + output);
            assertNotNull(result, name + " printed no result:\n" + output);
            String[] fields = result.split(" ");
            double bytes = Double.parseDouble(fields[2]), allowed = Double.parseDouble(fields[3]);
            if (bytes > allowed) failures.add(name + ": " + bytes + " bytes per operation, allowed " + allowed);
        }
        assertTrue(failures.isEmpty(), "Panama kernels allocated more than designed, which means vectors were "
                + "boxed on the heap somewhere on the path: " + failures);
    }

    // ---- minimal class-file reader: method names, descriptors and bytecode lengths ----

    private static final class Method {
        final String name, descriptor;
        final int codeLength;

        Method(String name, String descriptor, int codeLength) {
            this.name = name;
            this.descriptor = descriptor;
            this.codeLength = codeLength;
        }
    }

    /** Whether a method descriptor has a Vector API value (vector, mask or shuffle) as a parameter or result. */
    private static boolean carriesVector(String descriptor) {
        for (String type : new String[]{"FloatVector", "IntVector", "ByteVector", "ShortVector", "LongVector",
                "DoubleVector", "VectorMask", "VectorShuffle", "Vector;"}) {
            if (descriptor.contains("Ljdk/incubator/vector/" + type)) return true;
        }
        return false;
    }

    private static List<Method> methods(Class<?> c) throws IOException {
        try (InputStream raw = c.getResourceAsStream(c.getName().substring(c.getName().lastIndexOf('.') + 1) + ".class")) {
            assertNotNull(raw, c + " class file");
            var in = new DataInputStream(raw);
            assertEquals(0xCAFEBABE, in.readInt());
            in.readUnsignedShort();
            in.readUnsignedShort();
            int count = in.readUnsignedShort();
            String[] utf8 = new String[count];
            for (int i = 1; i < count; i++) {
                int tag = in.readUnsignedByte();
                switch (tag) {
                    case 1: utf8[i] = in.readUTF(); break;
                    case 3: case 4: in.readInt(); break;
                    case 5: case 6: in.readLong(); i++; break;
                    case 7: case 8: case 16: case 19: case 20: in.readUnsignedShort(); break;
                    case 9: case 10: case 11: case 12: case 17: case 18: in.readInt(); break;
                    case 15: in.readUnsignedByte(); in.readUnsignedShort(); break;
                    default: fail("unknown constant pool tag " + tag);
                }
            }
            in.readUnsignedShort();
            in.readUnsignedShort();
            in.readUnsignedShort();
            in.skipBytes(2 * in.readUnsignedShort());
            int fields = in.readUnsignedShort();
            for (int f = 0; f < fields; f++) {
                in.skipBytes(6);
                skipAttributes(in);
            }
            List<Method> methods = new ArrayList<>();
            int methodCount = in.readUnsignedShort();
            for (int m = 0; m < methodCount; m++) {
                in.readUnsignedShort();
                String name = utf8[in.readUnsignedShort()];
                String descriptor = utf8[in.readUnsignedShort()];
                int codeLength = 0;
                int attributes = in.readUnsignedShort();
                for (int a = 0; a < attributes; a++) {
                    String attribute = utf8[in.readUnsignedShort()];
                    int length = in.readInt();
                    if (attribute.equals("Code")) {
                        in.readUnsignedShort();
                        in.readUnsignedShort();
                        codeLength = in.readInt();
                        in.skipBytes(length - 8);
                    } else {
                        in.skipBytes(length);
                    }
                }
                methods.add(new Method(name, descriptor, codeLength));
            }
            return methods;
        }
    }

    private static void skipAttributes(DataInputStream in) throws IOException {
        int attributes = in.readUnsignedShort();
        for (int a = 0; a < attributes; a++) {
            in.readUnsignedShort();
            in.skipBytes(in.readInt());
        }
    }
}

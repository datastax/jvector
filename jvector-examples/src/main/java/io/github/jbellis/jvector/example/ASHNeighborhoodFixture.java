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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.AsymmetricHashing;
import io.github.jbellis.jvector.quantization.FusedASHLayout;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Untimed fixture creation and replay for the production ASH neighborhood layout.
 *
 * <p>Version 1 stores big-endian ints/floats: magic, version, capacity, block width, base count,
 * model byte count and model, query count and original-dimensional queries, then neighborhood
 * count, bytes per neighborhood, and packed payloads. fvecs input remains little-endian.
 * Model metadata and lengths are checked before allocating dimensions or payload arrays.</p>
 *
 * <p>Fixture bytes and query vectors are owned by the fixture and treated as immutable during
 * replay. Queries are normalized on creation, matching the inputs passed to the graph scorer.
 * The final neighborhood is padded to capacity and remains part of the scored work.</p>
 */
final class ASHNeighborhoodFixture {
    private static final int MAGIC = 0x4153484e;
    private static final int VERSION = 1;
    private static final int MAX_FVECS_DIMENSION = 100_000;
    private static final int FIXED_BYTES = 9 * Integer.BYTES;

    final AsymmetricHashing ash;
    final List<VectorFloat<?>> queries;
    final byte[][] neighborhoods;
    final int capacity;
    final int blockSize;
    final int baseCount;

    ASHNeighborhoodFixture(AsymmetricHashing ash, List<VectorFloat<?>> queries, byte[][] neighborhoods, int capacity,
            int blockSize, int baseCount) {
        this.ash = Objects.requireNonNull(ash, "ash");
        if (ash.landmarkCount != 1) {
            throw new IllegalArgumentException("Neighborhood benchmark requires C=1");
        }
        FusedASHLayout.validateBitsPerDimension(ash.bitsPerDimension);
        FusedASHLayout.validateBlockSize(blockSize);
        if (capacity < 1 || baseCount < 1 || queries.isEmpty()) {
            throw new IllegalArgumentException("Empty fixture");
        }
        if (neighborhoods.length != (baseCount - 1) / capacity + 1) {
            throw new IllegalArgumentException("Neighborhood count mismatch");
        }
        int bytes = FusedASHLayout.featureSize(capacity, ash.quantizedDim, ash.bitsPerDimension, blockSize);
        for (byte[] neighborhood : neighborhoods) {
            if (neighborhood.length != bytes) {
                throw new IllegalArgumentException("Payload size mismatch");
            }
        }
        for (VectorFloat<?> query : queries) {
            if (query.length() != ash.originalDimension) {
                throw new IllegalArgumentException("Query dimension mismatch");
            }
        }
        this.queries = List.copyOf(queries);
        this.neighborhoods = neighborhoods;
        this.capacity = capacity;
        this.blockSize = blockSize;
        this.baseCount = baseCount;
    }

    static ASHNeighborhoodFixture create(String baseFile, String queryFile) throws IOException {
        int capacity = ASHNeighborhoodBenchmark.positive("jvector.bench.neighborhoodSize", 32);
        int blockSize =
                ASHNeighborhoodBenchmark.positive("jvector.bench.blockSize", FusedASHLayout.chooseBlockSize(capacity));
        FusedASHLayout.validateBlockSize(blockSize);
        ASHNeighborhoodBenchmark.checkSetting("jvector.ash.landmarkCount", 1);
        List<VectorFloat<?>> base =
                readFvecs(baseFile, ASHNeighborhoodBenchmark.positive("jvector.bench.maxVectors", 8192));
        List<VectorFloat<?>> queries =
                readFvecs(queryFile, ASHNeighborhoodBenchmark.positive("jvector.bench.maxQueries", 8));
        if (base.isEmpty() || queries.isEmpty()) {
            throw new IllegalArgumentException("Empty vector input");
        }
        int dimensions = base.get(0).length();
        for (VectorFloat<?> query : queries) {
            if (query.length() != dimensions) {
                throw new IllegalArgumentException("Base/query dimensions differ");
            }
            VectorUtil.l2normalize(query);
        }
        int bits = ASHNeighborhoodBenchmark.positive("jvector.ash.bitsPerDimension", 4);
        FusedASHLayout.validateBitsPerDimension(bits);
        int projected = ASHNeighborhoodBenchmark.positive("jvector.ash.quantizedDimensions", dimensions / 2);
        if (projected > dimensions) {
            throw new IllegalArgumentException("Projected dimensions exceed original dimensions");
        }
        var values = new ListRandomAccessVectorValues(base, dimensions);
        var ash = AsymmetricHashing.initialize(values, DistancesASH.parseOptimizerFromProperty(),
                Math.addExact(AsymmetricHashing.HEADER_BITS, Math.multiplyExact(projected, bits)), 1, bits);
        ASHVectors encoded = (ASHVectors) ash.encodeAll(values);
        int count = (base.size() - 1) / capacity + 1;
        byte[][] packed = new byte[count][FusedASHLayout.featureSize(capacity, projected, bits, blockSize)];
        for (int i = 0; i < base.size(); i++) {
            int neighbor = i % capacity;
            int offset = FusedASHLayout.blockOffset(neighbor / blockSize, projected, bits, blockSize);
            FusedASHLayout.packForFused(
                    packed[i / capacity], offset, neighbor % blockSize, encoded.get(i), ash, blockSize);
        }
        return new ASHNeighborhoodFixture(ash, queries, packed, capacity, blockSize, base.size());
    }

    /** Read only the requested prefix, rejecting truncated, non-finite, or inconsistent vectors. */
    static List<VectorFloat<?>> readFvecs(String filename, int limit) throws IOException {
        if (limit < 1) {
            throw new IllegalArgumentException("Vector limit must be positive");
        }
        List<VectorFloat<?>> result = new ArrayList<>();
        var types = VectorizationProvider.getInstance().getVectorTypeSupport();
        try (var input = new DataInputStream(new BufferedInputStream(Files.newInputStream(Path.of(filename))))) {
            while (result.size() < limit) {
                int first = input.read();
                if (first == -1) {
                    break;
                }
                int dimensions = first | input.readUnsignedByte() << 8 | input.readUnsignedByte() << 16
                        | input.readUnsignedByte() << 24;
                if (dimensions < 1 || dimensions > MAX_FVECS_DIMENSION
                        || (!result.isEmpty() && dimensions != result.get(0).length())) {
                    throw new IOException("Invalid fvecs dimension: " + dimensions);
                }
                float[] vector = new float[dimensions];
                for (int i = 0; i < dimensions; i++) {
                    vector[i] = Float.intBitsToFloat(Integer.reverseBytes(input.readInt()));
                    if (!Float.isFinite(vector[i])) {
                        throw new IOException("Non-finite fvecs value at row " + result.size() + ", dimension " + i);
                    }
                }
                result.add(types.createFloatVector(vector));
            }
        }
        return result;
    }

    /** Publish a complete fixture; a failed write or existing destination never leaves a partial file. */
    static void write(Path path, ASHNeighborhoodFixture fixture) throws IOException {
        Path absolute = path.toAbsolutePath();
        Path temporary = Files.createTempFile(absolute.getParent(), ".ash-neighborhood-", ".tmp");
        try {
            writeComplete(temporary, fixture);
            Files.move(temporary, absolute); // No REPLACE_EXISTING: preserve existing fixture inputs.
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private static void writeComplete(Path path, ASHNeighborhoodFixture fixture) throws IOException {
        try (var model = ByteBufferIndexWriter.create(fixture.ash.compressorSize(), false)) {
            fixture.ash.write(model, OnDiskGraphIndex.CURRENT_VERSION);
            ByteBuffer buffer = model.getWrittenData();
            byte[] bytes = new byte[buffer.remaining()];
            buffer.get(bytes);
            try (var out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
                out.writeInt(MAGIC);
                out.writeInt(VERSION);
                out.writeInt(fixture.capacity);
                out.writeInt(fixture.blockSize);
                out.writeInt(fixture.baseCount);
                out.writeInt(bytes.length);
                out.write(bytes);
                out.writeInt(fixture.queries.size());
                for (VectorFloat<?> query : fixture.queries) {
                    for (int d = 0; d < query.length(); d++) {
                        out.writeFloat(query.get(d));
                    }
                }
                out.writeInt(fixture.neighborhoods.length);
                out.writeInt(fixture.neighborhoods[0].length);
                for (byte[] block : fixture.neighborhoods) {
                    out.write(block);
                }
            }
        }
    }

    static ASHNeighborhoodFixture read(Path path) throws IOException {
        long fileBytes = Files.size(path);
        try (var input = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            if (input.readInt() != MAGIC || input.readInt() != VERSION) {
                throw new IOException("Invalid neighborhood fixture header");
            }
            int capacity = input.readInt();
            int blockSize = input.readInt();
            int baseCount = input.readInt();
            int modelSize = input.readInt();
            if (capacity < 1 || baseCount < 1 || modelSize < 1 || modelSize > fileBytes - FIXED_BYTES) {
                throw new IOException("Invalid fixture sizes");
            }
            FusedASHLayout.validateBlockSize(blockSize);
            byte[] model = new byte[modelSize];
            input.readFully(model);
            validateModelEnvelope(model);
            var modelBuffer = ByteBuffer.wrap(model).order(ByteOrder.BIG_ENDIAN);
            var ash = AsymmetricHashing.load(new ByteBufferReader(modelBuffer));
            if (modelBuffer.hasRemaining()) {
                throw new IOException("Trailing bytes in fixture model");
            }
            FusedASHLayout.validateBitsPerDimension(ash.bitsPerDimension);
            int queries = input.readInt();
            long queryBytes = (long) queries * ash.originalDimension * Float.BYTES;
            long payloadStart = FIXED_BYTES + (long) modelSize + queryBytes;
            if (queries < 1 || queryBytes < 0 || payloadStart > fileBytes) {
                throw new IOException("Invalid query count");
            }
            var types = VectorizationProvider.getInstance().getVectorTypeSupport();
            List<VectorFloat<?>> vectors = new ArrayList<>();
            for (int i = 0; i < queries; i++) {
                float[] query = new float[ash.originalDimension];
                for (int d = 0; d < query.length; d++) {
                    query[d] = input.readFloat();
                    if (!Float.isFinite(query[d])) {
                        throw new IOException("Non-finite fixture query value");
                    }
                }
                vectors.add(types.createFloatVector(query));
            }
            int count = input.readInt();
            int bytes = input.readInt();
            int expected = FusedASHLayout.featureSize(capacity, ash.quantizedDim, ash.bitsPerDimension, blockSize);
            if (count != (baseCount - 1) / capacity + 1 || bytes != expected
                    || payloadStart + (long) count * bytes != fileBytes) {
                throw new IOException("Invalid packed payload sizes or trailing bytes");
            }
            byte[][] blocks = new byte[count][bytes];
            for (byte[] block : blocks) {
                input.readFully(block);
            }
            return new ASHNeighborhoodFixture(ash, vectors, blocks, capacity, blockSize, baseCount);
        } catch (IllegalArgumentException | ArithmeticException | BufferUnderflowException error) {
            throw new IOException("Invalid ASH neighborhood fixture: " + path, error);
        }
    }

    private static void validateModelEnvelope(byte[] model) throws IOException {
        // Fixture V1 embeds ASH's eight-int prefix, one landmark, and its row-major transform.
        // Check declared dimensions against the bytes before ASH.load allocates model arrays.
        int fixedModelBytes = 11 * Integer.BYTES;
        if (model.length < fixedModelBytes) {
            throw new IOException("Truncated fixture model");
        }
        ByteBuffer header = ByteBuffer.wrap(model).order(ByteOrder.BIG_ENDIAN);
        int original = header.getInt(2 * Integer.BYTES);
        int projected = header.getInt(4 * Integer.BYTES);
        int landmarks = header.getInt(7 * Integer.BYTES);
        long floats = (long) original * (projected + 1L);
        if (original < 1 || projected < 1 || projected > original || landmarks != 1
                || floats > (model.length - fixedModelBytes) / Float.BYTES
                || fixedModelBytes + floats * Float.BYTES != model.length) {
            throw new IOException("Invalid C=1 fixture model dimensions");
        }
    }

    static String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (var input = new DigestInputStream(Files.newInputStream(path), digest)) {
                input.transferTo(OutputStream.nullOutputStream());
            }
            byte[] hash = digest.digest();
            StringBuilder hex = new StringBuilder(hash.length * 2);
            for (byte value : hash) {
                hex.append(Character.forDigit((value >>> 4) & 0xF, 16));
                hex.append(Character.forDigit(value & 0xF, 16));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new AssertionError("SHA-256 is required by the Java platform", error);
        }
    }
}

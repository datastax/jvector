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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

public class SiftLoader {

    private static final VectorTypeSupport vectorTypeSupport =
            VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * Read all vectors from an .fvecs file.
     *
     * <p>
     * Reads the entire file and materializes all vectors into memory.
     * </p>
     */
    public static List<VectorFloat<?>> readFvecs(String filePath) {
        var vectors = new ArrayList<VectorFloat<?>>();
        try (var dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {

            while (dis.available() > 0) {
                int dimension = Integer.reverseBytes(dis.readInt());
                if (dimension <= 0) {
                    throw new IOException("Invalid vector dimension: " + dimension);
                }

                byte[] buffer = new byte[dimension * Float.BYTES];
                dis.readFully(buffer);

                ByteBuffer byteBuffer =
                        ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

                float[] data = new float[dimension];
                byteBuffer.asFloatBuffer().get(data);

                vectors.add(vectorTypeSupport.createFloatVector(data));
            }
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
        return vectors;
    }

    /**
     * Read at most {@code numVectors} vectors from an .fvecs file.
     *
     * <p>
     * This method stops reading once {@code numVectors} vectors
     * have been loaded, preventing unnecessary heap usage.
     * </p>
     *
     * <p>
     * If the file contains fewer than {@code numVectors} vectors, an exception is thrown.
     * </p>
     *
     * @param filePath  path to the .fvecs file
     * @param numVectors number of vectors to load
     * @return list of vectors of size {@code numVectors}
     *
     * @throws IllegalArgumentException if {@code numVectors <= 0}
     * @throws IOException if the file ends before {@code numVectors}
     *                     vectors are read
     */
    public static List<VectorFloat<?>> readFvecs(String filePath, int numVectors)
            throws IOException {

        if (numVectors <= 0) {
            throw new IllegalArgumentException(
                    "numVectors must be > 0, got " + numVectors);
        }

        var vectors = new ArrayList<VectorFloat<?>>(
                Math.min(numVectors, 1024));

        try (var dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {

            int count = 0;

            while (count < numVectors) {
                if (dis.available() <= 0) {
                    throw new IOException(
                            "File ended early while reading " + filePath +
                                    ": requested " + numVectors +
                                    " vectors, but only read " + count);
                }

                int dimension = Integer.reverseBytes(dis.readInt());
                if (dimension <= 0) {
                    throw new IOException("Invalid vector dimension: " + dimension);
                }

                byte[] buffer = new byte[dimension * Float.BYTES];
                dis.readFully(buffer);

                ByteBuffer byteBuffer =
                        ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

                float[] data = new float[dimension];
                byteBuffer.asFloatBuffer().get(data);

                vectors.add(vectorTypeSupport.createFloatVector(data));
                count++;
            }
        }

        // Defensive sanity check
        if (vectors.size() != numVectors) {
            throw new IOException(
                    "Internal error: requested " + numVectors +
                            " vectors but read " + vectors.size());
        }

        return vectors;
    }

    /**
     * Read all ground-truth neighbors from an .ivecs file as a list of primitive arrays.
     */
    public static List<int[]> readIvecsAsArrays(String filename) {
        try {
            return readIvecsAsArrays(filename, Integer.MAX_VALUE);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Read at most {@code numVectors} from an .ivecs file as a list of primitive arrays.
     */
    public static List<int[]> readIvecsAsArrays(String filePath, int numVectors) throws IOException {
        if (numVectors <= 0) {
            throw new IllegalArgumentException("numVectors must be > 0, got " + numVectors);
        }

        var groundTruth = new ArrayList<int[]>(Math.min(numVectors, 1024));

        try (var dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            int count = 0;
            while (count < numVectors && dis.available() > 0) {
                int k = Integer.reverseBytes(dis.readInt());
                int[] neighbors = new int[k];
                for (int i = 0; i < k; i++) {
                    neighbors[i] = Integer.reverseBytes(dis.readInt());
                }
                groundTruth.add(neighbors);
                count++;
            }
        }

        if (numVectors != Integer.MAX_VALUE && groundTruth.size() != numVectors) {
            throw new IOException(String.format("File %s ended early: requested %d, read %d",
                    filePath, numVectors, groundTruth.size()));
        }

        return groundTruth;
    }

    /**
     * Read all ground-truth neighbors from an .ivecs file.
     */
    public static List<List<Integer>> readIvecs(String filename) {
        var groundTruthTopK = new ArrayList<List<Integer>>();

        try (var dis = new DataInputStream(new FileInputStream(filename))) {
            while (dis.available() > 0) {
                int numNeighbors = Integer.reverseBytes(dis.readInt());
                var neighbors = new ArrayList<Integer>(numNeighbors);

                for (int i = 0; i < numNeighbors; i++) {
                    neighbors.add(Integer.reverseBytes(dis.readInt()));
                }

                groundTruthTopK.add(neighbors);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        return groundTruthTopK;
    }

    // Fast, allocation-free scan to count vectors in an .fvecs file
    public static int countFvecs(String filePath) throws IOException {
        Path path = Path.of(filePath);

        long fileSize = Files.size(path);
        if (fileSize < Integer.BYTES) {
            throw new IOException("File too small to be a valid .fvecs file: " + filePath);
        }

        // Read dimension from first record
        int dimension;
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            fc.read(buf);
            buf.flip();
            dimension = buf.getInt();
        }

        if (dimension <= 0) {
            throw new IOException("Invalid vector dimension: " + dimension);
        }

        long recordBytes = Integer.BYTES + (long) dimension * Float.BYTES;
        if (fileSize % recordBytes != 0) {
            throw new IOException(
                    "File size not a multiple of record size: " +
                            "fileSize=" + fileSize + ", recordBytes=" + recordBytes);
        }

        long count = fileSize / recordBytes;
        if (count > Integer.MAX_VALUE) {
            throw new IOException("Vector count exceeds Integer.MAX_VALUE: " + count);
        }

        return (int) count;
    }
}

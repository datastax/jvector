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

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Utility for loading Deep1B dataset files.
 */
public class Deep1BLoader {
    private Deep1BLoader() {
    }

    /**
     * Reads vectors from a binary file.
     * @param filePath the file path
     * @param count the number of vectors to read
     * @return the list of vectors
     * @throws IOException if an error occurs
     */
    public static List<float[]> readFBin(String filePath, int count) throws IOException {
        var vectors = new float[count][];

        int n;
        int dimension;
        try (var raf = new RandomAccessFile(filePath, "r")) {
            n = Integer.reverseBytes(raf.readInt());
            dimension = Integer.reverseBytes(raf.readInt());
        }
        System.out.printf("%s contains %d vectors; reading %d%n", filePath, n, count);

        var threadCount = 16;
        int perThread = count / threadCount;
        int remainder = count % threadCount;

        IntStream.range(0, threadCount).parallel().forEach(threadNum -> {
            try (var raf = new RandomAccessFile(filePath, "r")) {
                long startPosition = 8L + ((long) threadNum * perThread + Math.min(threadNum, remainder)) * dimension * Float.BYTES;
                raf.seek(startPosition);

                int itemsToProcess = perThread + (threadNum < remainder ? 1 : 0);
                for (int j = 0; j < itemsToProcess; j++) {
                    var buffer = new byte[dimension * Float.BYTES];
                    raf.readFully(buffer);
                    var byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

                    var vector = new float[dimension];
                    var floatBuffer = byteBuffer.asFloatBuffer();
                    floatBuffer.get(vector);
                    vectors[threadNum * perThread + (Math.min(threadNum, remainder)) + j] = vector;
                }
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });

        System.out.println("Completed reading " + filePath);
        return List.of(vectors);
    }

    /**
     * Reads ground truth data from a file.
     * @param filePath the file path
     * @return the ground truth data
     */
    public static ArrayList<HashSet<Integer>> readGT(String filePath) {
        var groundTruthTopK = new ArrayList<HashSet<Integer>>();

        try (var dis = new DataInputStream(new FileInputStream(filePath))) {
            var n = Integer.reverseBytes(dis.readInt());
            var topK = Integer.reverseBytes(dis.readInt());
            for (int i = 0; i < n; i++) {
                var neighbors = new HashSet<Integer>(topK);
                for (var j = 0; j < topK; j++) {
                    var neighbor = Integer.reverseBytes(dis.readInt());
                    neighbors.add(neighbor);
                }
                groundTruthTopK.add(neighbors);
            }
            // GT file also contains scores, we don't need those
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        System.out.println("Completed reading " + filePath);
        return groundTruthTopK;
    }
}

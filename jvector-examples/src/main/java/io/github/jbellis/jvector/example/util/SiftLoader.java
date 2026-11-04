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
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class SiftLoader {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static List<VectorFloat<?>> readFvecs(String filePath) {
        var vectors = new ArrayList<VectorFloat<?>>();
        try (var dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            while (dis.available() > 0) {
                var dimension = Integer.reverseBytes(dis.readInt());
                if (dimension <= 0) {
                    throw new IOException("Corrupt fvecs file: negative or zero dimension " + dimension + " (possible file corruption or wrong format)");
                }
                if (dimension > 100_000) {
                    throw new IOException("Unreasonable dimension " + dimension + " in fvecs file (possible file corruption or wrong format)");
                }
                var buffer = new byte[dimension * Float.BYTES];
                dis.readFully(buffer);
                var byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

                var vector = new float[dimension];
                var floatBuffer = byteBuffer.asFloatBuffer();
                floatBuffer.get(vector);
                vectors.add(vectorTypeSupport.createFloatVector(vector));
            }
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
        return vectors;
    }

    /// Reads a `.bvecs`-layout file whose components are **signed int8** values (-128..127).
    ///
    /// The layout is the same as the TexMex `.bvecs` format (a little-endian int dimension followed by
    /// `dimension` bytes, per vector), but the payload is interpreted as signed.
    /// **Unsigned uint8 payloads, which is what conventional `.bvecs` files such as SIFT1B/BIGANN contain
    /// (0..255), are not supported.**
    /// Bytes are copied as-is, so an unsigned file would be silently misread (200 becomes -56)
    /// and distances would change. Convert unsigned data to signed int8 before loading it.
    ///
    /// @param filePath path to a signed-int8 `.bvecs`-layout file
    /// @return the vectors, as signed int8 byte sequences
    public static List<ByteSequence<?>> readInt8Bvecs(String filePath) {
        var vectors = new ArrayList<ByteSequence<?>>();
        try (var dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            while (dis.available() > 0) {
                var dimension = Integer.reverseBytes(dis.readInt());
                if (dimension <= 0) {
                    throw new IOException("Corrupt bvecs file: negative or zero dimension " + dimension + " (possible file corruption or wrong format)");
                }
                if (dimension > 100_000) {
                    throw new IOException("Unreasonable dimension " + dimension + " in bvecs file (possible file corruption or wrong format)");
                }
                var data = new byte[dimension];
                dis.readFully(data);
                vectors.add(vectorTypeSupport.createByteSequence(data));
            }
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
        return vectors;
    }

    public static List<List<Integer>> readIvecs(String filename) {
        var groundTruthTopK = new ArrayList<List<Integer>>();

        try (var dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))) {
            while (dis.available() > 0) {
                var numNeighbors = Integer.reverseBytes(dis.readInt());
                var neighbors = new ArrayList<Integer>(numNeighbors);

                for (var i = 0; i < numNeighbors; i++) {
                    var neighbor = Integer.reverseBytes(dis.readInt());
                    neighbors.add(neighbor);
                }

                groundTruthTopK.add(neighbors);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        return groundTruthTopK;
    }
}

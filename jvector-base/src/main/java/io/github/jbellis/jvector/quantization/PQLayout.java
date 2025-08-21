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

/**
 * Chunk Dimensions and Layout
 * This is emulative of modern Java records, but keeps to J11 standards.
 * This class consolidates the layout calculations for PQ data into one place
 */
public class PQLayout {

    /** total number of vectors **/
    public final int vectorCount;
    /** total number of chunks, including any partial **/
    public final int totalChunks;
    /** total number of fully-filled chunks **/
    public final int fullSizeChunks;
    /** number of vectore per fullSize chunk **/
    public final int fullChunkVectors;
    /** number of vectors in last partially filled chunk, if any **/
    public final int lastChunkVectors;
    /** compressed dimension of vectors **/
    public final int compressedDimension;
    /** number of bytes in each fully-filled chunk **/
    public final int fullChunkBytes;
    /** number of bytes in the last partially-filled chunk, if any **/
    public final int lastChunkBytes;

    public PQLayout(int vectorCount, int compressedDimension) {
        if (vectorCount < 0) {
            throw new IllegalArgumentException("Invalid vector count " + vectorCount);
        }
        this.vectorCount = vectorCount;

        if (compressedDimension < 0) {
            throw new IllegalArgumentException("Invalid compressed dimension " + compressedDimension);
        }
        this.compressedDimension = compressedDimension;

        long totalSize = (long) vectorCount * compressedDimension;

        this.fullChunkVectors = totalSize <= PQVectors.MAX_CHUNK_SIZE ? vectorCount : PQVectors.MAX_CHUNK_SIZE / compressedDimension;
        if (fullChunkVectors == 0) {
            throw new IllegalArgumentException("Compressed dimension " + compressedDimension + " too large for chunking");
        }
        this.lastChunkVectors = vectorCount % this.fullChunkVectors;

        this.fullChunkBytes = Math.multiplyExact(compressedDimension, this.fullChunkVectors);
        this.lastChunkBytes = Math.multiplyExact(compressedDimension, lastChunkVectors);

        this.fullSizeChunks = vectorCount / fullChunkVectors;
        this.totalChunks = fullSizeChunks + ((vectorCount % fullChunkVectors == 0) ? 0 : 1);
    }
}

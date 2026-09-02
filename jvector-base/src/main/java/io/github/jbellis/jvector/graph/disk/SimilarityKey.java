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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.Random;

/**
 * A 32-bit key whose numeric order places similar vectors near each other, computable from the
 * vector alone. The merge assigns similarity ordinals by sorting live nodes on this key.
 *
 * <p>The key exists so it can be carried in the {@link NodeTokenStream}: the merge's ordinal pass
 * then reads keys from each source's stream instead of reading every source vector and encoding
 * it. That requires a function every index computes the same way, at construction, with nothing
 * that is only known at merge time — which rules out the previous key (the leading bytes of the
 * PQ code under the codebook retrained for that merge). The random-projection key below depends
 * only on the dimension and a fixed seed, so streams written by any generation are comparable.
 */
@Experimental
public interface SimilarityKey {
    /** No key function: the stream carries no keys. */
    byte NONE = 0;
    /** {@link #randomProjection}: sign bits of 32 fixed Gaussian random projections. */
    byte RANDOM_PROJECTION = 1;

    /** Identifies the function, so a reader can tell whether two streams' keys are comparable. */
    byte id();

    int keyOf(VectorFloat<?> vector);

    /**
     * Sign-hash locality: bit {@code 31 - b} is the sign of the vector's dot product with the
     * {@code b}-th of 32 hyperplanes drawn once from a fixed seed. Two vectors agree on a
     * hyperplane with probability {@code 1 - angle/pi}, so near vectors share their leading bits
     * and sort together; the first bits dominate the order the way the leading PQ subspaces did.
     */
    static SimilarityKey randomProjection(int dimension) {
        return new RandomProjectionKey(dimension);
    }

    final class RandomProjectionKey implements SimilarityKey {
        static final long SEED = 0x5EED5EED5EEDL;
        static final int BITS = 32;
        private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        private final VectorFloat<?>[] planes = new VectorFloat<?>[BITS];
        private final int dimension;

        RandomProjectionKey(int dimension) {
            if (dimension <= 0) {
                throw new IllegalArgumentException("dimension must be positive: " + dimension);
            }
            this.dimension = dimension;
            Random random = new Random(SEED);
            for (int b = 0; b < BITS; b++) {
                VectorFloat<?> plane = vts.createFloatVector(dimension);
                for (int i = 0; i < dimension; i++) {
                    plane.set(i, (float) random.nextGaussian());
                }
                planes[b] = plane;
            }
        }

        @Override
        public byte id() {
            return RANDOM_PROJECTION;
        }

        @Override
        public int keyOf(VectorFloat<?> vector) {
            if (vector.length() != dimension) {
                throw new IllegalArgumentException("vector has " + vector.length() + " dimensions, key expects " + dimension);
            }
            int key = 0;
            for (int b = 0; b < BITS; b++) {
                key <<= 1;
                if (VectorUtil.dotProduct(planes[b], vector) >= 0) {
                    key |= 1;
                }
            }
            return key;
        }
    }
}

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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * ASH knows the largest length of the vectors it holds codes for, however they were encoded: by the same model
 * instance, or by another instance of the model whose codes are wrapped afterwards.
 */
public class TestASHEncodedNorms {
    @Test
    public void codesRecoverTheLengthsTheyEncode() throws Exception {
        var random = new Random(23);
        int n = 400, dim = 48;
        List<VectorFloat<?>> base = new ArrayList<>();
        float maxNorm = 0;
        for (int i = 0; i < n; i++) {
            var v = TestUtil.randomVector(random, dim);
            VectorUtil.scale(v, (float) Math.pow(10, -1 + 2 * random.nextDouble()));
            base.add(v);
            maxNorm = Math.max(maxNorm, (float) Math.sqrt(VectorUtil.dotProduct(v, v)));
        }
        var ravv = new ListRandomAccessVectorValues(base, dim);
        for (int bits : new int[] {1, 2, 3, 4, 9}) {
            for (int landmarks : new int[] {1, 4}) {
                String where = bits + " bits, " + landmarks + " landmarks";
                var ash = AsymmetricHashing.initialize(ravv, AsymmetricHashing.RANDOM,
                        AsymmetricHashing.HEADER_BITS + 40 * bits, landmarks, bits);
                // A second instance of the same model, saved before it encoded anything.
                File file = File.createTempFile("ash-model", ".bin");
                file.deleteOnExit();
                try (var writer = new SimpleWriter(file.toPath())) {
                    ash.write(writer, OnDiskGraphIndex.CURRENT_VERSION);
                }
                AsymmetricHashing other;
                try (var readers = new SimpleMappedReader.Supplier(file.toPath())) {
                    other = AsymmetricHashing.load(readers.get());
                }
                var codes = new AsymmetricHashing.QuantizedVector[n];
                for (int i = 0; i < n; i++) {
                    codes[i] = ash.encode(base.get(i));
                    float exact = VectorUtil.dotProduct(base.get(i), base.get(i));
                    float fromCode = ash.encodedSquaredNorm(codes[i]);
                    assertEquals(where + " node " + i, exact, fromCode, exact * 2e-3f + 1e-4f);
                }
                assertEquals(where, maxNorm, ash.maxEncodedNorm(), maxNorm * 1e-5f);
                // Wrapping the codes with the other instance counts them, never below their true lengths.
                new ASHVectors(other, codes);
                assertTrue(where + ": " + other.maxEncodedNorm(), other.maxEncodedNorm() >= maxNorm);
                assertEquals(where, maxNorm, other.maxEncodedNorm(), maxNorm * 5e-3f);
            }
        }
    }
}

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

import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Offset kernels on codes embedded in larger arrays must equal the standalone kernels, bit for bit. */
class NativeASHSymmetricDotAtTest {
    @BeforeAll
    static void loadLibrary() {
        assumeTrue(LibraryLoader.loadJvector(), "Native library unavailable");
        assertTrue(NativeASHSymmetricKernels.hasPairKernel());
    }

    @Test
    void offsetCodesMatchStandaloneCodes() {
        Random random = new Random(20261108);
        for (int bits : new int[]{2, 4}) {
            for (int dimensions : new int[]{0, 1, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                                            257, 384, 512, 768, 1024, 1536, 3072, 65537}) {
                int bytes = (dimensions * bits + 7) / 8;
                for (int trial = 0; trial < 4; trial++) {
                    byte[] a = new byte[bytes], b = new byte[bytes];
                    random.nextBytes(a); random.nextBytes(b);
                    // Embed at arbitrary offsets surrounded by noise, as in a code page.
                    int aOff = random.nextInt(130), bOff = random.nextInt(130);
                    byte[] aPage = new byte[aOff + bytes + random.nextInt(70)];
                    byte[] bPage = new byte[bOff + bytes + random.nextInt(70)];
                    random.nextBytes(aPage); random.nextBytes(bPage);
                    System.arraycopy(a, 0, aPage, aOff, bytes);
                    System.arraycopy(b, 0, bPage, bOff, bytes);
                    float expected = NativeASHSymmetricKernels.dot(null, a, null, b, dimensions, bits);
                    assertEquals(expected, NativeASHSymmetricKernels.dotAt(aPage, aOff, bPage, bOff, dimensions, bits), 0f,
                            "bits=" + bits + " d=" + dimensions);
                    assertEquals(expected, NativeASHSymmetricKernels.dotAt(bPage, bOff, aPage, aOff, dimensions, bits), 0f);
                }
            }
        }
    }

    /** Every dimension count, including partial words and vectors, must equal a per-dimension decode. */
    @Test
    void codesMatchPerDimensionReference() {
        Random random = new Random(20261111);
        for (int bits : new int[]{2, 4}) {
            for (int dimensions = 1; dimensions <= 400; dimensions++) {
                int bytes = (dimensions * bits + 7) / 8;
                byte[] a = new byte[bytes], b = new byte[bytes];
                // Random padding bits as well: the kernels must ignore them.
                random.nextBytes(a); random.nextBytes(b);
                long doubled = 0;
                for (int j = 0; j < dimensions; j++) doubled += (long) component(a, j, bits) * component(b, j, bits);
                assertEquals(doubled * 0.25f, NativeASHSymmetricKernels.dot(null, a, null, b, dimensions, bits), 0f,
                        "bits=" + bits + " d=" + dimensions);
            }
        }
    }

    /** Doubled signed-magnitude component j: the sign is the field's high bit, the magnitude its low bits. */
    private static int component(byte[] code, int j, int bits) {
        int perByte = 8 / bits;
        int field = (code[j / perByte] & 0xFF) >>> ((j % perByte) * bits);
        int sign = 1 << (bits - 1);
        int magnitude = 2 * (field & (sign - 1)) + 1;
        return (field & sign) != 0 ? magnitude : -magnitude;
    }

    @Test
    void offsetSignWordsMatchStandaloneSignWords() {
        Random random = new Random(20261109);
        for (int dimensions : new int[]{0, 1, 63, 64, 65, 384, 1000, 65537}) {
            int words = (dimensions + 63) / 64;
            for (int trial = 0; trial < 4; trial++) {
                long[] a = random.longs(words).toArray(), b = random.longs(words).toArray();
                int aOff = random.nextInt(20), bOff = random.nextInt(20);
                long[] aPage = random.longs(aOff + words + random.nextInt(9)).toArray();
                long[] bPage = random.longs(bOff + words + random.nextInt(9)).toArray();
                System.arraycopy(a, 0, aPage, aOff, words);
                System.arraycopy(b, 0, bPage, bOff, words);
                float expected = NativeASHSymmetricKernels.dot(a, null, b, null, dimensions, 1);
                assertEquals(expected, NativeASHSymmetricKernels.binaryDotAt(aPage, aOff, bPage, bOff, dimensions), 0f,
                        "d=" + dimensions);
            }
        }
        long[] page = new long[4];
        assertThrows(IllegalArgumentException.class, () -> NativeASHSymmetricKernels.binaryDotAt(page, 0, page, 0, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHSymmetricKernels.binaryDotAt(page, 3, page, 0, 128));
        assertDoesNotThrow(() -> NativeASHSymmetricKernels.binaryDotAt(page, 2, page, 0, 128));
    }

    @Test
    void argumentAndBoundsChecks() {
        byte[] page = new byte[200];
        assertThrows(IllegalArgumentException.class, () -> NativeASHSymmetricKernels.dotAt(page, 0, page, 0, 8, 1));
        assertThrows(IllegalArgumentException.class, () -> NativeASHSymmetricKernels.dotAt(page, 0, page, 0, -1, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHSymmetricKernels.dotAt(page, 150, page, 0, 384, 2));
        assertThrows(IndexOutOfBoundsException.class, () -> NativeASHSymmetricKernels.dotAt(page, 0, page, -1, 8, 2));
        // Exactly fitting at the end of the page is legal.
        assertDoesNotThrow(() -> NativeASHSymmetricKernels.dotAt(page, 104, page, 0, 384, 2));
    }
}

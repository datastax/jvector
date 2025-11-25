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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.OptionalDouble;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestReconstructionError extends RandomizedTest {
    private Path testDirectory;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testReconstructionError_withProductQuantization() {
        testReconstructionError_withProductQuantization(1_000, 1.15, 2.5);
        testReconstructionError_withProductQuantization(10_000, 0.14, 0.29);
    }

    public void testReconstructionError_withProductQuantization(int nVectors, double toleranceAvg, double toleranceSTD) {
        int dimensions = 32;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(nVectors,  dimensions), dimensions);
        var ravvTest = new ListRandomAccessVectorValues(createRandomVectors(nVectors,  dimensions), dimensions);

        ProductQuantization pq = ProductQuantization.compute(ravv, 8, 256, true);

        compareErrors(pq, ravv, ravvTest, toleranceAvg, toleranceSTD);
    }

    @Test
    public void testReconstructionError_withBinaryQuantization() {
        testReconstructionError_withBinaryQuantization(1_000, 0.05, 0.25);
        testReconstructionError_withBinaryQuantization(10_000, 0.008, 0.09);
    }

    public void testReconstructionError_withBinaryQuantization(int nVectors, double toleranceAvg, double toleranceSTD) {
        int dimensions = 32;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(nVectors,  dimensions), dimensions);
        var ravvTest = new ListRandomAccessVectorValues(createRandomVectors(nVectors,  dimensions), dimensions);

        BinaryQuantization bq = new BinaryQuantization(dimensions);

        compareErrors(bq, ravv, ravvTest, toleranceAvg, toleranceSTD);
    }

    @Test
    public void testReconstructionError_withNVQuantization() {
        testReconstructionError_withBinaryQuantization(1_000, 4e-2, 0.25);
        testReconstructionError_withBinaryQuantization(10_000, 1e-2, 0.08);
    }

    public void testReconstructionError_withNVQuantization(int nVectors, double toleranceAvg, double toleranceSTD) {
        int dimensions = 32;
        var ravv = new ListRandomAccessVectorValues(createRandomVectors(nVectors,  dimensions), dimensions);
        var ravvTest = new ListRandomAccessVectorValues(createRandomVectors(nVectors,  dimensions), dimensions);

        NVQuantization nvq = NVQuantization.compute(ravv, 2);

        compareErrors(nvq, ravv, ravvTest, toleranceAvg, toleranceSTD);
    }

    void compareErrors(VectorCompressor<?> compressor, RandomAccessVectorValues sample1, RandomAccessVectorValues sample2, double toleranceAvg, double toleranceSTD) {
        double[] errors1 = compressor.reconstructionErrors(sample1);
        double averageError1 = Arrays.stream(errors1).average().getAsDouble();
        double varError1 = Arrays.stream(errors1).map(x -> (x - averageError1) * (x - averageError1)).average().getAsDouble();

        double[] errors2 = compressor.reconstructionErrors(sample2);
        double averageError2 = Arrays.stream(errors2).average().getAsDouble();
        double varError2 = Arrays.stream(errors2).map(x -> (x - averageError2) * (x - averageError2)).average().getAsDouble();

        // check relative error
        assertEquals(1, averageError2 / averageError1, toleranceAvg);
        assertEquals(1, varError2 / varError1, toleranceSTD);
    }
}

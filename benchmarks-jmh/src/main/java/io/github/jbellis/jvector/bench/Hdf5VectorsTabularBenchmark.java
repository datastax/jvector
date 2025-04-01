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
package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.bench.output.*;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class Hdf5VectorsTabularBenchmark extends AbstractVectorsBenchmark {
    private static final Logger log = LoggerFactory.getLogger(Hdf5VectorsTabularBenchmark.class);

    @Param({"glove-100-angular.hdf5"})  // default value
    String hdf5Filename;

    @Param({"TEXT"}) // Default value, can be overridden via CLI
    String outputFormat;

    private TableRepresentation getTableRepresentation() {
        if (outputFormat == null) {
            outputFormat = "TEXT";
        }
        switch (outputFormat) {
            case "TEXT":
                return new TextTable();
            case "JSON":
                return new JsonTable();
            case "SQLITE":
                return new SqliteTable();
            case "PERSISTENT_TEXT":
                return new PersistentTextTable();
            default:
                throw new IllegalArgumentException("Invalid output format: " + outputFormat);
        }
    }

    @Setup
    public void setup() throws IOException {
        DownloadHelper.maybeDownloadHdf5(hdf5Filename);
        DataSet dataSet = Hdf5Loader.load(hdf5Filename);
        baseVectors = dataSet.baseVectors;
        queryVectors = dataSet.queryVectors;
        groundTruth = dataSet.groundTruth;
        tableRepresentation = getTableRepresentation();
        commonSetupStatic(true);
    }

    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        groundTruth.clear();
        graphIndexBuilder.close();
        scheduler.shutdown();
        tableRepresentation.print();
        tableRepresentation.tearDown();
    }

    @Benchmark
    public void testOnHeapWithStaticQueryVectors(Blackhole blackhole) {
        commonTest(blackhole);
    }

}

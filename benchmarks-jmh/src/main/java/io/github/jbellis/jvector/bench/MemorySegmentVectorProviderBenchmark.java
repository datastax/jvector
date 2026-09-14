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

import io.github.jbellis.jvector.vector.MemorySegmentVectorProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.disk.IndexWriter;
import org.openjdk.jmh.annotations.*;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Fork(2)
@State(Scope.Benchmark)
public class MemorySegmentVectorProviderBenchmark {

    @Param({"512","1024","1536"})
    public int length;

    private MemorySegmentVectorProvider provider;
    private VectorFloat<?> fvector;
    private ByteSequence<?> bvector;

    @Setup(Level.Trial)
    public void setup() {
        provider = new MemorySegmentVectorProvider();
        float[] fdata = new float[length];
        byte[] bdata = new byte[length];
        for (int i = 0; i < length; i++) {
            fdata[i] = (float) i;
            bdata[i] = (byte) i;
        }
        fvector = provider.createFloatVector(fdata);
        bvector = provider.createByteSequence(bdata);
    }

    @Benchmark
    public void writeFloatVector() throws IOException {
        try (MemoryIndexWriter w = new MemoryIndexWriter(length * 4)) {
            provider.writeFloatVector(w, fvector);
        }
    }

    @Benchmark
    public void writeByteVector() throws IOException {
        try (MemoryIndexWriter w = new MemoryIndexWriter(length * 4)) {
            provider.writeByteSequence(w, bvector);
        }
    }

    static final class MemoryIndexWriter implements IndexWriter {
        private final java.io.ByteArrayOutputStream bos;
        private final java.io.DataOutputStream out;

        MemoryIndexWriter(int capacity) {
            this.bos = new java.io.ByteArrayOutputStream(capacity);
            this.out = new java.io.DataOutputStream(bos);
        }

        byte[] toByteArray() { return bos.toByteArray(); }

        @Override public long position() { return bos.size(); }
        @Override public void close() throws IOException { out.close(); }

        @Override public void write(int b) throws IOException { out.write(b); }
        @Override public void write(byte[] b) throws IOException { out.write(b); }
        @Override public void write(byte[] b, int off, int len) throws IOException { out.write(b, off, len); }
        @Override public void writeFloat(float v) throws IOException { out.writeFloat(v); }

        @Override public void writeBoolean(boolean v) throws IOException { out.writeBoolean(v); }
        @Override public void writeByte(int v) throws IOException { out.writeByte(v); }
        @Override public void writeShort(int v) throws IOException { out.writeShort(v); }
        @Override public void writeChar(int v) throws IOException { out.writeChar(v); }
        @Override public void writeInt(int v) throws IOException { out.writeInt(v); }
        @Override public void writeLong(long v) throws IOException { out.writeLong(v); }
        @Override public void writeDouble(double v) throws IOException { out.writeDouble(v); }
        @Override public void writeBytes(String s) throws IOException { out.writeBytes(s); }
        @Override public void writeChars(String s) throws IOException { out.writeChars(s); }
        @Override public void writeUTF(String s) throws IOException { out.writeUTF(s); }
    }
}

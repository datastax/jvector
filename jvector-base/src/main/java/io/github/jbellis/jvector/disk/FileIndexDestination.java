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

package io.github.jbellis.jvector.disk;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * File-backed {@link IndexDestination}: the normative Java IO mapping of the contract and the
 * standalone fast path. Backs {@link IndexDestination#toFile}, {@link IndexDestination#toFiles}
 * and {@link IndexDestination#inFile}.
 * <p>
 * Standalone mode streams each artifact into a sibling temporary file and renames it over the
 * target at {@link OutputSession#commit()}, so a reader of the target never observes a partial
 * artifact and an abort leaves any previous file intact. Region mode streams into a caller-owned
 * file from a fixed offset and never truncates, deletes or renames it.
 * <p>
 * Limitation: with several standalone artifacts, a failure part-way through the renames of
 * {@code commit()} leaves the artifacts already renamed in place; plain files offer no
 * multi-file atomic publish.
 */
final class FileIndexDestination implements IndexDestination {
    private final Map<OutputArtifact, Path> targets;
    /** Region start within the single target, or {@code -1} in standalone mode. */
    private final long regionOffset;

    static FileIndexDestination standalone(Map<OutputArtifact, Path> paths) {
        Objects.requireNonNull(paths, "paths");
        if (paths.isEmpty()) {
            throw new IllegalArgumentException("at least one artifact path is required");
        }
        paths.forEach((artifact, path) -> {
            Objects.requireNonNull(artifact, "artifact");
            Objects.requireNonNull(path, "path for " + artifact);
        });
        return new FileIndexDestination(paths, -1L);
    }

    static FileIndexDestination region(Path path, long offset) {
        Objects.requireNonNull(path, "path");
        if (offset < 0) {
            throw new IllegalArgumentException("offset must be >= 0, got " + offset);
        }
        return new FileIndexDestination(Collections.singletonMap(OutputArtifact.GRAPH, path), offset);
    }

    private FileIndexDestination(Map<OutputArtifact, Path> targets, long regionOffset) {
        EnumMap<OutputArtifact, Path> absolute = new EnumMap<>(OutputArtifact.class);
        targets.forEach((artifact, path) -> absolute.put(artifact, path.toAbsolutePath()));
        this.targets = Collections.unmodifiableMap(absolute);
        this.regionOffset = regionOffset;
    }

    private boolean standalone() {
        return regionOffset < 0;
    }

    @Override
    public OutputSession open() {
        return new Session();
    }

    @Override
    public String toString() {
        return standalone()
                ? "IndexDestination.toFiles" + targets
                : "IndexDestination.inFile[" + targets.get(OutputArtifact.GRAPH) + " @ " + regionOffset + "]";
    }

    private static Path parentOf(Path absolutePath) {
        Path parent = absolutePath.getParent();
        return parent != null ? parent : absolutePath.getRoot();
    }

    private static IOException suppress(IOException first, IOException next) {
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }

    private enum SessionState { OPEN, COMMITTED, ABORTED, CLOSED }

    /** One write operation: lazily opens one file per reserved artifact, publishes on commit. */
    private final class Session implements OutputSession {
        private final EnumMap<OutputArtifact, Reservation> reservations = new EnumMap<>(OutputArtifact.class);
        private SessionState state = SessionState.OPEN;

        @Override
        public OutputReservation reserve(OutputArtifact artifact) throws IOException {
            checkOpen();
            Objects.requireNonNull(artifact, "artifact");
            Path target = targets.get(artifact);
            if (target == null) {
                throw new IllegalArgumentException(FileIndexDestination.this + " has no placement for " + artifact);
            }
            if (reservations.containsKey(artifact)) {
                throw new IllegalStateException(artifact + " is already reserved in this session");
            }
            Reservation reservation;
            if (standalone()) {
                Path tmp = Files.createTempFile(parentOf(target), target.getFileName() + ".", ".tmp");
                FileChannel channel;
                try {
                    channel = FileChannel.open(tmp, StandardOpenOption.WRITE);
                } catch (IOException e) {
                    Files.deleteIfExists(tmp);
                    throw e;
                }
                reservation = new Reservation(artifact, target, tmp, channel, 0L);
            } else {
                FileChannel channel = FileChannel.open(target, StandardOpenOption.WRITE, StandardOpenOption.CREATE);
                reservation = new Reservation(artifact, target, null, channel, regionOffset);
            }
            reservations.put(artifact, reservation);
            return reservation;
        }

        @Override
        public void commit() throws IOException {
            checkOpen();
            for (Reservation r : reservations.values()) {
                if (!r.completed) {
                    throw new IllegalStateException(r.artifact + " was reserved but never completed");
                }
                if (!r.closed) {
                    throw new IllegalStateException("the reservation for " + r.artifact + " is still open");
                }
            }
            if (standalone()) {
                try {
                    for (Reservation r : reservations.values()) {
                        r.publish();
                    }
                } catch (IOException e) {
                    state = SessionState.ABORTED;
                    throw e;
                }
            }
            state = SessionState.COMMITTED;
        }

        @Override
        public void close() throws IOException {
            if (state == SessionState.CLOSED) {
                return;
            }
            boolean committed = state == SessionState.COMMITTED;
            state = SessionState.CLOSED;
            IOException first = null;
            for (Reservation r : reservations.values()) {
                try {
                    r.close();
                } catch (IOException e) {
                    first = suppress(first, e);
                }
                if (!committed && r.tmp != null) {
                    // Completed but never published: the temporary file is a discarded artifact.
                    try {
                        Files.deleteIfExists(r.tmp);
                    } catch (IOException e) {
                        first = suppress(first, e);
                    }
                }
            }
            if (first != null) {
                throw first;
            }
        }

        private void checkOpen() {
            if (state != SessionState.OPEN) {
                throw new IllegalStateException("session is " + state.name().toLowerCase(Locale.ROOT));
            }
        }
    }

    /** One artifact's output: an append-only stream into a channel from {@code base}. */
    private final class Reservation implements OutputReservation {
        final OutputArtifact artifact;
        final Path target;
        /** The temporary file in standalone mode; {@code null} in region mode. */
        final Path tmp;
        final FileChannel channel;
        final long base;
        private ChannelStream stream;
        boolean completed;
        boolean closed;

        Reservation(OutputArtifact artifact, Path target, Path tmp, FileChannel channel, long base) {
            this.artifact = artifact;
            this.target = target;
            this.tmp = tmp;
            this.channel = channel;
            this.base = base;
        }

        @Override
        public OutputArtifact artifact() {
            return artifact;
        }

        @Override
        public IndexWriter stream() {
            checkOpen();
            if (stream == null) {
                stream = new ChannelStream(channel, base);
            }
            return stream;
        }

        @Override
        public void complete() throws IOException {
            checkOpen();
            if (stream != null) {
                stream.flush();
            }
            channel.force(false);
            completed = true;
        }

        @Override
        public void close() throws IOException {
            if (closed) {
                return;
            }
            closed = true;
            IOException first = null;
            if (stream != null) {
                stream.abandon();
            }
            try {
                channel.close();
            } catch (IOException e) {
                first = suppress(first, e);
            }
            if (tmp != null && !completed) {
                try {
                    Files.deleteIfExists(tmp);
                } catch (IOException e) {
                    first = suppress(first, e);
                }
            }
            if (first != null) {
                throw first;
            }
        }

        /** Standalone only: renames the completed temporary file over the target. */
        void publish() throws IOException {
            try {
                Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        private void checkOpen() {
            if (closed) {
                throw new IllegalStateException("the reservation for " + artifact + " is closed");
            }
            if (completed) {
                throw new IllegalStateException(artifact + " is already completed");
            }
        }
    }

    /**
     * Buffered append-only {@link IndexWriter} over a channel region: byte {@code i} of the
     * stream lands at channel position {@code base + i}, in write order. {@link #position()} is
     * the number of bytes written so far. {@link #close()} flushes and never closes the channel;
     * {@link #abandon()} drops buffered bytes on an abort.
     */
    private static final class ChannelStream implements IndexWriter {
        private static final int BUFFER_SIZE = 1 << 16;

        private final FileChannel channel;
        private final long base;
        private final ByteBuffer buffer = ByteBuffer.allocate(BUFFER_SIZE).order(ByteOrder.BIG_ENDIAN);
        private long flushed;
        private boolean closed;

        ChannelStream(FileChannel channel, long base) {
            this.channel = channel;
            this.base = base;
        }

        @Override
        public long position() {
            return flushed + buffer.position();
        }

        void flush() throws IOException {
            if (buffer.position() == 0) {
                return;
            }
            buffer.flip();
            long abs = base + flushed;
            int n = buffer.remaining();
            while (buffer.hasRemaining()) {
                abs += channel.write(buffer, abs);
            }
            buffer.clear();
            flushed += n;
        }

        void abandon() {
            closed = true;
            buffer.clear();
        }

        @Override
        public void close() throws IOException {
            if (closed) {
                return;
            }
            try {
                flush();
            } finally {
                closed = true;
            }
        }

        private void ensure(int n) throws IOException {
            if (closed) {
                throw new IOException("stream is closed");
            }
            if (buffer.remaining() < n) {
                flush();
            }
        }

        @Override
        public void write(int b) throws IOException {
            ensure(1);
            buffer.put((byte) b);
        }

        @Override
        public void write(byte[] b) throws IOException {
            write(b, 0, b.length);
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            ensure(0);
            if (len >= buffer.capacity()) {
                // Larger than the buffer: flush what is pending, then write straight through.
                flush();
                ByteBuffer src = ByteBuffer.wrap(b, off, len);
                long abs = base + flushed;
                while (src.hasRemaining()) {
                    abs += channel.write(src, abs);
                }
                flushed += len;
                return;
            }
            ensure(len);
            buffer.put(b, off, len);
        }

        @Override
        public void writeBoolean(boolean v) throws IOException {
            write(v ? 1 : 0);
        }

        @Override
        public void writeByte(int v) throws IOException {
            write(v);
        }

        @Override
        public void writeShort(int v) throws IOException {
            ensure(Short.BYTES);
            buffer.putShort((short) v);
        }

        @Override
        public void writeChar(int v) throws IOException {
            ensure(Character.BYTES);
            buffer.putChar((char) v);
        }

        @Override
        public void writeInt(int v) throws IOException {
            ensure(Integer.BYTES);
            buffer.putInt(v);
        }

        @Override
        public void writeLong(long v) throws IOException {
            ensure(Long.BYTES);
            buffer.putLong(v);
        }

        @Override
        public void writeFloat(float v) throws IOException {
            ensure(Float.BYTES);
            buffer.putFloat(v);
        }

        @Override
        public void writeDouble(double v) throws IOException {
            ensure(Double.BYTES);
            buffer.putDouble(v);
        }

        @Override
        public void writeBytes(String s) throws IOException {
            int len = s.length();
            for (int i = 0; i < len; i++) {
                write((byte) s.charAt(i));
            }
        }

        @Override
        public void writeChars(String s) throws IOException {
            int len = s.length();
            for (int i = 0; i < len; i++) {
                writeChar(s.charAt(i));
            }
        }

        /** Modified UTF-8 with a two-byte length prefix, byte-identical to {@link DataOutputStream#writeUTF}. */
        @Override
        public void writeUTF(String s) throws IOException {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream(s.length() + 2);
            new DataOutputStream(bytes).writeUTF(s);
            write(bytes.toByteArray());
        }

        @Override
        public void writeFloats(float[] floats, int offset, int count) throws IOException {
            int i = offset;
            int end = offset + count;
            while (i < end) {
                ensure(Float.BYTES);
                int n = Math.min(end - i, buffer.remaining() / Float.BYTES);
                buffer.asFloatBuffer().put(floats, i, n);
                buffer.position(buffer.position() + n * Float.BYTES);
                i += n;
            }
        }
    }
}

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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 * {@link SeekableSink} over a {@link FileChannel}, translating region-relative positions by a fixed
 * base offset. The channel is owned by the caller; {@link #close()} does not close it.
 */
final class FileChannelSeekableSink implements SeekableSink {
    private final FileChannel channel;
    private final long baseOffset;

    FileChannelSeekableSink(FileChannel channel, long baseOffset) {
        if (channel == null) {
            throw new NullPointerException("channel");
        }
        if (baseOffset < 0) {
            throw new IllegalArgumentException("baseOffset must be >= 0, got " + baseOffset);
        }
        this.channel = channel;
        this.baseOffset = baseOffset;
    }

    @Override
    public void writeAt(long position, ByteBuffer src) throws IOException {
        if (position < 0) {
            throw new IllegalArgumentException("position must be >= 0, got " + position);
        }
        long abs = baseOffset + position;
        while (src.hasRemaining()) {
            abs += channel.write(src, abs);
        }
    }

    @Override
    public int readAt(long position, ByteBuffer dst) throws IOException {
        if (position < 0) {
            throw new IllegalArgumentException("position must be >= 0, got " + position);
        }
        return channel.read(dst, baseOffset + position);
    }

    @Override
    public void force() throws IOException {
        channel.force(false);
    }

    @Override
    public void close() {
        // The channel is owned by the caller, per SeekableSink.over(...).
    }
}

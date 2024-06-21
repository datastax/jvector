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

import java.io.Closeable;
import java.io.DataOutput;
import java.io.IOException;

/**
 * A DataOutput that adds methods for random access writes
 */
public interface RandomAccessWriter extends DataOutput, Closeable {
    void seek(long position) throws IOException;

    long position() throws IOException;

    void flush() throws IOException;

    long checksum(long startOffset, long endOffset) throws IOException;
}

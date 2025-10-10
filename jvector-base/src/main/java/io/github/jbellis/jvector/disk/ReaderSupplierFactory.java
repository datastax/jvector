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
import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Factory for creating ReaderSupplier instances with fallback support across multiple implementations.
 * Tries MemorySegmentReader first (JDK 20+), then MMapReader, and finally MappedChunkReader.
 */
public class ReaderSupplierFactory {
    private static final Logger LOG = Logger.getLogger(ReaderSupplierFactory.class.getName());

    /** Private constructor to prevent instantiation of this utility class */
    private ReaderSupplierFactory() {
        throw new UnsupportedOperationException("Utility class");
    }
    private static final String MEMORY_SEGMENT_READER_CLASSNAME = "io.github.jbellis.jvector.disk.MemorySegmentReader$Supplier";
    private static final String MMAP_READER_CLASSNAME = "io.github.jbellis.jvector.example.util.MMapReader$Supplier";

    /**
     * Opens a ReaderSupplier for the specified file, using the best available implementation.
     * @param path the path to the file to read
     * @return a ReaderSupplier for creating readers for the file
     * @throws IOException if an I/O error occurs while opening the file
     */
    public static ReaderSupplier open(Path path) throws IOException {
        try {
            // prefer MemorySegmentReader (available under JDK 20+)
            var supplierClass = Class.forName(MEMORY_SEGMENT_READER_CLASSNAME);
            Constructor<?> ctor = supplierClass.getConstructor(Path.class);
            return (ReaderSupplier) ctor.newInstance(path);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "MemorySegmentReaderSupplier not available, falling back to MMapReaderSupplier. Reason: {0}: {1}",
                    new Object[]{e.getClass().getName(), e.getMessage()});
        }

        try {
            // fall back to MMapReader (requires a 3rd party linux-only native mmap library that is only included
            // in the build with jvector-example; this allows Bench to not embarrass us on older JDKs)
            var supplierClass = Class.forName(MMAP_READER_CLASSNAME);
            Constructor<?> ctor = supplierClass.getConstructor(Path.class);
            return (ReaderSupplier) ctor.newInstance(path);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "MMapReaderSupplier not available, falling back to MappedChunkReader. More details available at level FINE.");
            LOG.log(Level.FINE, "MMapReaderSupplier instantiation exception:", e);

            // finally, fall back to MappedChunkReader
            return new MappedChunkReader.Supplier(path);
        }
    }
}

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
package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.SimpleMappedReaderSupplier;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ReaderSupplierFactory {
    private static final Logger LOG = Logger.getLogger(ReaderSupplierFactory.class.getName());
    private static final String MEMORY_SEGMENT_READER_CLASSNAME = "io.github.jbellis.jvector.disk.MemorySegmentReaderSupplier";

    public static ReaderSupplier open(Path path) throws IOException {
        try {
            var supplierClass = Class.forName(MEMORY_SEGMENT_READER_CLASSNAME);
            Constructor<?> ctor = supplierClass.getConstructor(Path.class);
            return (ReaderSupplier) ctor.newInstance(path);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "MemorySegmentReaderSupplier not available, falling back to MMapReaderSupplier. Reason: {0}: {1}",
                    new Object[]{e.getClass().getName(), e.getMessage()});
        }

        try {
            return new MMapReaderSupplier(path);
        } catch (UnsatisfiedLinkError|NoClassDefFoundError e) {
            LOG.log(Level.WARNING, "MMapReaderSupplier not available, falling back to SimpleMappedReaderSupplier. More details available at level FINE.");
            LOG.log(Level.FINE, "MMapReaderSupplier instantiation exception:", e);
            if (Files.size(path) > Integer.MAX_VALUE) {
                throw new RuntimeException("File sizes greater than 2GB are not supported on Windows--contributions welcome");
            }

            return new SimpleMappedReaderSupplier(path);
        }
    }
}

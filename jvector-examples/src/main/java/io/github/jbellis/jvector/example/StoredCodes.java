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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.quantization.ASHVectors;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.PQVectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * The traversal codes of an index that does not fuse them into its records, written beside the index file when the
 * index is written. A search loads them when it opens the index, as a deployed searcher would, instead of encoding
 * every base vector again. The file holds the compressor and its codes, so it describes itself.
 */
final class StoredCodes {
    private static final int MAGIC = 0x4A56_4344; // "JVCD"
    /** ASH codes, the only codes a batch build stores. */
    private static final int FORMAT = 1;
    /** PQ codes, stored only by a streaming index, which searches with the codes it built. */
    private static final int PQ_FORMAT = 2;

    private StoredCodes() {}

    /** Codes loaded from beside an index, and the seconds loading them took. */
    static final class Loaded {
        final CompressedVectors codes;
        final double seconds;
        /** True for codes an in-memory index built itself, which are not loaded from a file. */
        final boolean builtWithIndex;

        private Loaded(CompressedVectors codes, double seconds, boolean builtWithIndex) {
            this.codes = codes;
            this.seconds = seconds;
            this.builtWithIndex = builtWithIndex;
        }

        private Loaded(CompressedVectors codes, double seconds) {
            this(codes, seconds, false);
        }

        /** Codes an in-memory index built while it was built; searches use them as they are. */
        static Loaded built(CompressedVectors codes) {
            return new Loaded(codes, 0, true);
        }

        /** These codes, loaded from a file, marked as built by the index, so searches use them as they are. */
        Loaded builtByIndex() {
            return new Loaded(codes, seconds, true);
        }
    }

    /** The codes file for {@code index}. */
    static Path pathFor(Path index) {
        return index.resolveSibling(index.getFileName() + ".codes");
    }

    /**
     * Whether codes of this type can be stored: ASH codes, and also PQ codes when the index built them itself, as a
     * streaming index does.
     */
    static boolean supports(CompressedVectors codes, boolean builtByIndex) {
        return codes instanceof ASHVectors || builtByIndex && codes instanceof PQVectors;
    }

    /** Writes {@code codes} beside {@code index} and returns the seconds it took. */
    static double write(CompressedVectors codes, Path index) throws IOException {
        long start = System.nanoTime();
        Path path = pathFor(index);
        try (var out = new BufferedRandomAccessWriter(path)) {
            out.writeInt(MAGIC);
            out.writeInt(codes instanceof PQVectors ? PQ_FORMAT : FORMAT);
            codes.write(out, OnDiskGraphIndex.CURRENT_VERSION);
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        System.out.printf("Wrote %d traversal codes [%.2f MB] beside the index in %.2f s%n",
                codes.count(), Files.size(path) / 1024.0 / 1024.0, seconds);
        return seconds;
    }

    /** Moves the codes written beside {@code from} to sit beside {@code to}, if there are any. */
    static void move(Path from, Path to) throws IOException {
        Path source = pathFor(from);
        if (Files.exists(source)) {
            Files.move(source, pathFor(to), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        }
    }

    /** Deletes the codes beside {@code index}, if there are any. */
    static void delete(Path index) throws IOException {
        Files.deleteIfExists(pathFor(index));
    }

    /**
     * Loads the codes stored beside {@code index}, or returns null when there are none or they do not cover the
     * index's {@code nodes} ordinals.
     */
    static Loaded load(Path index, int nodes) throws IOException {
        Path path = pathFor(index);
        if (!Files.exists(path)) return null;
        long start = System.nanoTime();
        try (var supplier = ReaderSupplierFactory.open(path); var in = supplier.get()) {
            int format = in.readInt() == MAGIC ? in.readInt() : -1;
            if (format != FORMAT && format != PQ_FORMAT) {
                System.out.printf("Ignoring %s: not a codes file this version reads%n", path.getFileName());
                return null;
            }
            CompressedVectors codes = format == FORMAT ? ASHVectors.load(in) : PQVectors.load(in);
            if (codes.count() != nodes) {
                System.out.printf("Ignoring %s: %d codes for an index of %d nodes%n", path.getFileName(), codes.count(), nodes);
                return null;
            }
            double seconds = (System.nanoTime() - start) / 1e9;
            System.out.printf("Loaded %d traversal codes stored with the index in %.2f s%n", codes.count(), seconds);
            return new Loaded(codes, seconds);
        }
    }
}

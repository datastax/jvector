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

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * Sets the BLAS thread count in the native library that netlib loads for JVector's BLAS and LAPACK
 * calls: {@code dev.ludovic.netlib.blas.nativeLibPath}, else {@code dev.ludovic.netlib.blas.nativeLib},
 * else {@code libblas.so.3}. Opening it again returns the handle netlib uses.
 */
final class NativeBlasThreads {
    private static final Logger LOG = Logger.getLogger(NativeBlasThreads.class.getName());
    private static final Control CONTROL = find();

    private NativeBlasThreads() {
    }

    /** Applies the count and returns the value to restore, or -1 when no supported library is loaded. */
    static int set(int threads) {
        return CONTROL == null ? -1 : CONTROL.set(threads);
    }

    /** The mechanism in use, for diagnostics. */
    static String description() {
        return CONTROL == null ? "none" : CONTROL.name;
    }

    private static final class Control {
        final String name;
        final MethodHandle setter;   // int f(int) returning the previous value, or void f(int)
        final MethodHandle getter;   // int f(void), only for process-wide setters
        final boolean global;

        Control(String name, MethodHandle setter, MethodHandle getter, boolean global) {
            this.name = name;
            this.setter = setter;
            this.getter = getter;
            this.global = global;
        }

        int set(int threads) {
            try {
                if (!global) return (int) setter.invokeExact(threads);
                int previous = (int) getter.invokeExact();
                if (previous != threads) setter.invokeExact(threads);
                return previous;
            } catch (Throwable t) {
                throw new IllegalStateException("Setting " + name + " to " + threads + " failed", t);
            }
        }
    }

    private static Control find() {
        String path = System.getProperty("dev.ludovic.netlib.blas.nativeLibPath");
        String library = path != null ? path : System.getProperty("dev.ludovic.netlib.blas.nativeLib", "libblas.so.3");
        SymbolLookup lookup;
        try {
            lookup = SymbolLookup.libraryLookup(library, Arena.global());
        } catch (IllegalArgumentException | UnsupportedOperationException e) {
            LOG.fine("No native BLAS library " + library + "; BLAS threads follow its own configuration.");
            return null;
        }
        var linker = Linker.nativeLinker();
        var intToInt = FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT);
        var intToVoid = FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT);
        var voidToInt = FunctionDescriptor.of(ValueLayout.JAVA_INT);

        // MKL's C interface is the mixed-case name; its lower- and upper-case exports are the
        // Fortran interface, which takes a pointer.
        Optional<Control> control = lookup.find("MKL_Set_Num_Threads_Local")
                .map(s -> new Control("MKL thread-local threads", linker.downcallHandle(s, intToInt), null, false))
                .or(() -> lookup.find("openblas_set_num_threads_local")
                        .map(s -> new Control("OpenBLAS thread-local threads", linker.downcallHandle(s, intToInt), null, false)))
                .or(() -> lookup.find("openblas_set_num_threads").flatMap(set -> lookup.find("openblas_get_num_threads")
                        .map(get -> new Control("OpenBLAS process-wide threads",
                                linker.downcallHandle(set, intToVoid), linker.downcallHandle(get, voidToInt), true))));
        if (control.isEmpty()) {
            LOG.fine("Native BLAS library " + library + " has no thread-count setting; BLAS threads follow its own configuration.");
            return null;
        }
        LOG.fine("BLAS threads are set through " + control.get().name + " in " + library);
        return control.get();
    }
}

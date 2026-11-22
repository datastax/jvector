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

import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import io.github.jbellis.jvector.vector.cnative.NativeSimdOps;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.lang.foreign.SymbolLookup;
import java.lang.reflect.Method;

/**
 * VectorizationProvider using MemorySegment vectors, Highway kernels, and Panama
 * for operations without native implementations.
 */
public class NativeVectorizationProvider extends VectorizationProvider {
    private static final int REQUIRED_ABI = 1;

    private final VectorUtilSupport vectorUtilSupport;
    private final VectorTypeSupport vectorTypeSupport;

    /// Load the native library and create the provider.
    ///
    /// Throws `UnsupportedOperationException` when the architecture is unsupported, the library
    /// can't be loaded, the bindings can't initialize (including when native access is denied),
    /// or the loaded library doesn't support the required ABI or export every bound function, so automatic
    /// selection falls back to the Java Vector API instead of failing at the first kernel call.
    public NativeVectorizationProvider() {
        var arch = System.getProperty("os.arch", "");
        if (!arch.equals("amd64") && !arch.equals("x86_64") && !arch.equals("aarch64")) {
            throw new UnsupportedOperationException("Native SIMD operations are only supported on x86_64 and aarch64.");
        }
        var libraryLoaded = LibraryLoader.loadJvector();
        if (!libraryLoaded) {
            String reason = NativeVectorizationProvider.class.getModule().isNativeAccessEnabled() ? ""
                    : " Native access is not enabled for this module; pass --enable-native-access=ALL-UNNAMED"
                      + " (or the jvector module name).";
            throw new UnsupportedOperationException("Failed to load supporting native library." + reason);
        }
        var support = NativeVectorUtilSupport.create();
        try {
            if (SymbolLookup.loaderLookup().find("jvector_supports_abi").isEmpty()) {
                throw new UnsupportedOperationException("The loaded jvector native library does not export "
                        + "jvector_supports_abi; compatibility with required ABI " + REQUIRED_ABI + " is unknown.");
            }
            if (NativeSimdOps.jvector_supports_abi(REQUIRED_ABI) != 1) {
                throw new UnsupportedOperationException("The loaded jvector native library does not support required ABI "
                        + REQUIRED_ABI + "; a different libjvector may be on java.library.path.");
            }
            String missing = firstMissingSymbol();
            if (missing != null) {
                throw new UnsupportedOperationException("The loaded jvector native library does not export " + missing
                        + "; a different libjvector may be on java.library.path.");
            }
            support.getActiveIsa();
        } catch (IllegalCallerException | LinkageError e) {
            throw new UnsupportedOperationException("Native bindings could not initialize.", e);
        }
        this.vectorUtilSupport = support;
        this.vectorTypeSupport = new MemorySegmentVectorProvider();
    }

    /// Returns the first function bound in [NativeSimdOps] (one per generated `name$address()`
    /// accessor) that the libraries loaded by this class loader don't export, or `null` when all
    /// are present. Uses a symbol lookup only, so no binding is initialized.
    private static String firstMissingSymbol() {
        var lookup = SymbolLookup.loaderLookup();
        for (Method method : NativeSimdOps.class.getMethods()) {
            String name = method.getName();
            if (method.getParameterCount() == 0 && name.endsWith("$address")) {
                String symbol = name.substring(0, name.length() - "$address".length());
                if (lookup.find(symbol).isEmpty()) {
                    return symbol;
                }
            }
        }
        return null;
    }

    @Override
    public VectorUtilSupport getVectorUtilSupport() {
        return vectorUtilSupport;
    }

    @Override
    public VectorTypeSupport getVectorTypeSupport() {
        return vectorTypeSupport;
    }
}

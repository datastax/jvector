package io.github.jbellis.jvector.fastervector;

import io.github.jbellis.jvector.fastervector.FastervectorLibrary;

/**
 * Lazy holder for native library initialization.
 * The static initializer runs only when {@link #LOADED} is first accessed,
 * which happens just before the first native call — not at class-load time of
 * the wrapper classes.
 */
final class NativeHolder {
    static final boolean LOADED;

    static {
        FastervectorLibrary.load();
        LOADED = true;
    }

    private NativeHolder() {}

    /** Trigger native library loading. Call once before any FFI usage. */
    static void ensureLoaded() {
        // Accessing LOADED triggers the class initializer if not yet run.
        if (!LOADED) throw new AssertionError("unreachable");
    }
}

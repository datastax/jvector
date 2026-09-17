// Copyright DataStax, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{env, path::PathBuf};

fn main() {
    // Resolve the directory that contains libjvector.so.
    //
    // Resolution order:
    //   1. JVECTOR_LIB_DIR environment variable (explicit override).
    //   2. <native-root>/builddir  — the in-tree Meson build directory produced
    //      by running `meson setup builddir && meson compile -C builddir` from
    //      the native root directory.
    let lib_dir = if let Ok(dir) = env::var("JVECTOR_LIB_DIR") {
        PathBuf::from(dir)
    } else {
        // This file lives at rust/build.rs.
        // CARGO_MANIFEST_DIR is the crate root (rust/).
        // Go up one level to reach the native root, then into builddir/.
        let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        manifest.join("..").join("builddir")
    };

    // Verify that libjvector.so is a real, resolved file — not just that the
    // directory exists.  canonicalize() only stat-s the directory, so it would
    // succeed even when Meson has created builddir/ and its symlinks but hasn't
    // compiled the library yet (leaving libjvector.so as a dangling symlink).
    // fs::canonicalize on the .so itself follows the symlink chain and returns
    // Err if the final target is absent, which is exactly what we need.
    let so = lib_dir.join("libjvector.so");
    let lib_dir = so
        .canonicalize()
        .unwrap_or_else(|_| {
            panic!(
                "\n\
                libjvector.so not found in {}.\n\
                Build the native library first:\n\
                \n\
                \x20 meson setup builddir --wipe --buildtype=release\n\
                \x20 meson compile -C builddir\n\
                \n\
                Or point JVECTOR_LIB_DIR at the directory containing libjvector.so.\n",
                lib_dir.display()
            )
        })
        .parent()
        .unwrap()
        .to_path_buf();

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=jvector");

    // Embed the search path as an rpath so the binary finds libjvector.so at
    // runtime without requiring LD_LIBRARY_PATH to be set.
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        lib_dir.display()
    );

    // Re-run this script when the library or the override variable changes.
    println!("cargo:rerun-if-changed={}/libjvector.so", lib_dir.display());
    println!("cargo:rerun-if-env-changed=JVECTOR_LIB_DIR");
}

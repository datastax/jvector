# JVector — Rust bindings

Rust bindings for `libjvector.so`, the native SIMD backend of JVector.
Exposes the full public C ABI (`cosine_f32`, `dot_product_f32`, `euclidean_f32`,
element-wise arithmetic, PQ kernels, NVQ kernels, and ISA diagnostics) as
`unsafe extern "C"` functions.

---

## Directory layout

```
rust/
├── Cargo.toml          — crate manifest (name: jvector)
├── build.rs            — locates libjvector.so and emits rustc-link-* directives
├── src/
│   └── lib.rs          — extern "C" declarations for every exported symbol
└── examples/
    └── similarity_demo.rs  — runnable demo: cosine, dot product, euclidean
```

---

## Prerequisites

| Tool | Minimum | Notes |
|------|---------|-------|
| g++ / clang++ | GCC 11+ | Must support `-march=skylake-avx512` |
| [Meson](https://mesonbuild.com/) | 0.55 | `pip install meson` |
| [Ninja](https://ninja-build.org/) | any | `sudo apt install ninja-build` |
| Rust | stable | `rustup update stable` |
| Git submodules | — | `git submodule update --init` (once) |

---

## Steps

### 1. Build the native library

Run these commands from the **native root** (the directory containing this
`rust/` folder and `meson.build`):

```bash
meson setup builddir --wipe --buildtype=release
meson compile -C builddir
```

This produces `builddir/libjvector.so`.

### 2. Build the Rust crate

```bash
cd rust
cargo build
```

`build.rs` automatically finds `../builddir/libjvector.so` — no environment
variables needed.

### 3. Run the example

```bash
cargo run --example similarity_demo
```

Expected output (ISA tier will vary by CPU):

```
Active ISA : avx2

a          = [1.0, 2.0, 3.0, 4.0]
b          = [4.0, 3.0, 2.0, 1.0]

dot product  = 20.000000
cosine sim   = 0.666667
euclidean    = 20.000000   (squared L2 distance)

dot product (with offset=2) = 20.000000  (same result: 20.000000)
```

---

## Optional overrides

**Different build type:**
```bash
meson setup builddir --wipe --buildtype=debug
meson compile -C builddir
cargo build
```

**Point to a `libjvector.so` built elsewhere:**
```bash
JVECTOR_LIB_DIR=/some/other/path cargo build
```

**Cap the ISA tier at runtime** (no recompile needed):
```bash
JVECTOR_MAX_ISA=avx2  cargo run --example similarity_demo
JVECTOR_MAX_ISA=sse42 cargo run --example similarity_demo
```

Accepted values: `avx3_spr`, `avx3_dl`, `avx3`, `avx2`, `sse42`.

---

## Using the crate in your own project

Add it as a path dependency:

```toml
[dependencies]
jvector = { path = "../rust" }
```

Then call any exported function:

```rust
use jvector::dot_product_f32;

let a = vec![1.0_f32, 2.0, 3.0, 4.0];
let b = vec![4.0_f32, 3.0, 2.0, 1.0];
let result = unsafe { dot_product_f32(a.as_ptr(), 0, b.as_ptr(), 0, a.len()) };
println!("{result}"); // 20
```

All functions are `unsafe` because they accept raw pointers. You are
responsible for ensuring the pointers are valid and `length` does not exceed
the allocated slice.

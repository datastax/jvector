# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project('jvector_simd_kernels', 'cpp',
  version: '0.1.0',
  meson_version: '>= 0.55.0',
  default_options: ['cpp_std=c++17']
)

# Highway headers are used as an include directory only.
# HWY_COMPILE_ONLY_STATIC / HWY_COMPILE_ONLY_SCALAR bypass the
# dynamic-dispatch runtime, so only headers are needed at compile time.
hwy_inc = include_directories('third_party/highway')

# Each ISA variant: name, JV_ISA namespace, and extra compiler flags.
# These all compile jvector_simd_kernels.cpp (the generic kernel file).
cpu = host_machine.cpu_family()

if cpu == 'x86_64'
  isa_variants = [
    {
      'name'     : 'avx3',
      'namespace': 'AVX3',
      'args'     : ['-march=skylake-avx512',
                    '-DHWY_COMPILE_ONLY_STATIC',
                    '-DJV_REQUIRE_HWY_AVX3'],
    },
    {
      'name'     : 'avx2',
      'namespace': 'AVX2',
      'args'     : ['-march=haswell',
                    '-maes',
                    '-DHWY_COMPILE_ONLY_STATIC',
                    '-DJV_REQUIRE_HWY_AVX2'],
    },
    {
      'name'     : 'sse42',
      'namespace': 'SSE42',
      'args'     : ['-msse4.2', '-mpclmul', '-maes',
                    '-DHWY_COMPILE_ONLY_STATIC',
                    '-DJV_REQUIRE_HWY_SCALAR'],
    },
  ]
elif cpu == 'aarch64'
  # Three tiers in ascending capability order, all compiled to fixed-width
  # Highway targets so MaxLanes is a compile-time constant and all kernel
  # fast-paths (Shuffle*, LoadDup256, etc.) work without ISA-specific guards.
  #
  #   NEON     — baseline AArch64.  Highway target: HWY_NEON.
  #              All Graviton generations and all Apple Silicon.
  #
  #   SVE_256  — Graviton 3 / Neoverse V1.  Highway target: HWY_SVE_256.
  #              -msve-vector-bits=256 asserts the physical vector width and
  #              causes Highway to select the fixed-width HWY_SVE_256 target
  #              (MaxLanes = 8 floats).  Graviton-only; no Apple Silicon SVE.
  #
  #   SVE2_128 — Graviton 4 / Neoverse V2/N2.  Highway target: HWY_SVE2_128.
  #              -msve-vector-bits=128 selects the fixed-width HWY_SVE2_128
  #              target (MaxLanes = 4 floats).  At runtime on Graviton 4 the
  #              hardware executes 256-bit vectors, so Highway's loop unrolling
  #              gives 256-bit throughput with 128-bit-wide loop bodies.
  #              Requires GCC >= 10 or Clang >= 21.  Graviton-only (no Apple).
  isa_variants = [
    {
      'name'     : 'neon',
      'namespace': 'NEON',
      'args'     : ['-march=armv8-a+crypto',
                    '-DHWY_COMPILE_ONLY_STATIC',
                    '-DJV_REQUIRE_HWY_NEON'],
    },
    {
      'name'     : 'sve',
      'namespace': 'SVE',
      'args'     : ['-march=armv8.4-a+sve', '-msve-vector-bits=256',
                    '-DHWY_COMPILE_ONLY_STATIC',
                    '-DJV_REQUIRE_HWY_SVE'],
    },
    {
      'name'     : 'sve2',
      'namespace': 'SVE2',
      'args'     : ['-march=armv9-a+sve2', '-msve-vector-bits=128',
                    '-DHWY_COMPILE_ONLY_STATIC',
                    '-DJV_REQUIRE_HWY_SVE2'],
    },
  ]
else
  error('Unsupported CPU family: ' + cpu + '. Supported: x86_64, aarch64.')
endif

isa_libs = []
foreach isa : isa_variants
  lib = static_library(
    'simdKernels_' + isa['name'],
    sources            : 'src/jvector_simd_kernels.cpp',
    include_directories: hwy_inc,
    cpp_args           : isa['args'] + ['-DJV_ISA=' + isa['namespace'], '-fvisibility=hidden']
  )
  isa_libs += lib
endforeach

if cpu == 'x86_64'
  # AVX3_DL (Ice Lake) tier: -march=icelake-server enables the full ICX feature
  # set (AVX3 + VNNI, VBMI, VBMI2, IFMA, BITALG, VPOPCNTDQ, GFNI, VAES, VPCLMULQDQ).
  avx3_dl_lib = static_library(
    'simdKernels_avx3_dl',
    sources            : 'src/jvector_avx3_dl_kernels.cpp',
    include_directories: hwy_inc,
    cpp_args           : [
      '-march=icelake-server',
      '-DHWY_COMPILE_ONLY_STATIC',
      '-DJV_REQUIRE_HWY_AVX3_DL',
      '-fvisibility=hidden',
    ],
  )
  isa_libs += avx3_dl_lib

  # AVX3_SPR (Sapphire Rapids) tier: -march=sapphirerapids adds AVX512FP16 and
  # AVX512BF16 on top of the icelake-server feature set.
  # Requires GCC >= 12 or Clang >= 14.
  avx3_spr_lib = static_library(
    'simdKernels_avx3_spr',
    sources            : 'src/jvector_avx3_spr_kernels.cpp',
    include_directories: hwy_inc,
    cpp_args           : [
      '-march=sapphirerapids',
      '-DHWY_COMPILE_ONLY_STATIC',
      '-DJV_REQUIRE_HWY_AVX3_SPR',
      '-fvisibility=hidden',
    ],
  )
  isa_libs += avx3_spr_lib
endif

# vectorUtil provides a single runtime-dispatch entry point.
# link_whole pulls every ISA object into the shared library so that no
# symbols are dropped when consumers link against it.
# abort.cc defines hwy::Abort, referenced by the SSE4 scalar fallbacks in
# generic_ops-inl.h. It was extracted from targets.cc specifically for
# projects that manage their own dispatch (google/highway#1935).
vectorutil_lib = shared_library(
  'jvector',
  sources            : ['src/jvector_simd.cpp',
                        'third_party/highway/hwy/abort.cc'],
  include_directories: [include_directories('src'), hwy_inc],
  cpp_args           : ['-DJVECTOR_BUILD', '-fvisibility=hidden'],
  link_whole         : isa_libs,
  version            : meson.project_version(),
  install            : true,
)

# Dependency object for use by executables/tests in this build tree.
vectorutil_dep = declare_dependency(
  link_with          : vectorutil_lib,
  include_directories: include_directories('src'),
)

## Example driver that exercises the runtime-dispatch API.
#executable(
#  'cpp_driver',
#  sources     : 'examples/cpp_driver.cpp',
#  dependencies: vectorutil_dep,
#)

# ---- Tests -----------------------------------------------------------------
gtest_dep = dependency('gtest_main', required: false)

if gtest_dep.found()
  # Select the architecture-specific CPU features test.
  cpu_features_test_src = cpu == 'x86_64' \
    ? 'tests/test_x86_cpu_features.cpp' \
    : 'tests/test_aarch64_cpu_features.cpp'

  simd_kernels_test = executable(
    'test_simd_kernels',
    sources     : [
      'tests/test_helpers.cpp',
      'tests/test_similarity.cpp',
      'tests/test_elementwise.cpp',
      cpu_features_test_src,
    ],
    dependencies: [vectorutil_dep, gtest_dep],
  )

  test('simd_kernels', simd_kernels_test, protocol: 'gtest', suite: 'simd_kernels')

else
  warning('gtest_main not found — test_simd_kernels target will not be built. ' +
          'Install libgtest-dev (apt) or googletest (brew) and re-run meson setup.')
  run_target('test_simd_kernels',
    command: ['echo',
      'ERROR: test_simd_kernels is not available because gtest_main was not found. ' +
      'Install libgtest-dev (apt) or googletest (brew) and re-run: meson setup --wipe build'])
endif

# ---- Benchmarks ------------------------------------------------------------
gbench_dep = dependency('benchmark', required: false)

if gbench_dep.found()
  executable(
    'bench_simd_kernels',
    sources     : 'benchmarks/bench_similarity_f32.cpp',
    dependencies: [vectorutil_dep, gbench_dep],
    cpp_args    : ['-O3'],
  )
else
  warning('Google Benchmark not found — bench_simd_kernels target will not be built. ' +
          'Install libbenchmark-dev (apt) or google-benchmark (brew) and re-run meson setup.')
  run_target('bench_simd_kernels',
    command: ['echo',
      'ERROR: bench_simd_kernels is not available because the benchmark library was not found. ' +
      'Install libbenchmark-dev (apt) or google-benchmark (brew) and re-run: meson setup --wipe build'])
endif
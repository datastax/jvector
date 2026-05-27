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

# Each ISA variant: name, JV_ISA namespace, and extra compiler flags
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

isa_libs = []
foreach isa : isa_variants
  lib = static_library(
    'simdKernels_' + isa['name'],
    sources            : 'jvector_simd_kernels.cpp',
    include_directories: hwy_inc,
    cpp_args           : isa['args'] + ['-DJV_ISA=' + isa['namespace'], '-fvisibility=hidden']
  )
  isa_libs += lib
endforeach

# vectorUtil provides a single runtime-dispatch entry point.
# link_whole pulls every ISA object into the shared library so that no
# symbols are dropped when consumers link against it.
# abort.cc defines hwy::Abort, referenced by the SSE4 scalar fallbacks in
# generic_ops-inl.h. It was extracted from targets.cc specifically for
# projects that manage their own dispatch (google/highway#1935).
vectorutil_lib = shared_library(
  'jvector',
  sources            : ['jvector_simd.cpp',
                        'third_party/highway/hwy/abort.cc'],
  include_directories: [include_directories('.'), hwy_inc],
  cpp_args           : ['-DJVECTOR_BUILD', '-fvisibility=hidden'],
  link_whole         : isa_libs,
  version            : meson.project_version(),
  install            : true,
)

# Dependency object for use by executables/tests in this build tree.
vectorutil_dep = declare_dependency(
  link_with          : vectorutil_lib,
  include_directories: include_directories('.'),
)

## Example driver that exercises the runtime-dispatch API.
#executable(
#  'cpp_driver',
#  sources     : 'examples/cpp_driver.cpp',
#  dependencies: vectorutil_dep,
#)
#
## ---- Tests -----------------------------------------------------------------
#gtest_dep = dependency('gtest_main', required: true)
#
#test_exe = executable(
#  'test_kernels',
#  sources     : [
#    'tests/test_kernels.cpp',
#    'tests/test_cpuFeatures.cpp',
#  ],
#  dependencies: [vectorutil_dep, gtest_dep],
#)
#
#test('kernels',     test_exe, protocol: 'gtest', suite: 'kernels')
#test('cpu_features', test_exe, protocol: 'gtest', suite: 'cpu',
#     args: ['--gtest_filter=CpuFeaturesTest.*'])
#
## ---- Benchmarks ------------------------------------------------------------
#gbench_dep = dependency('benchmark', required: false)
#if gbench_dep.found()
#  executable(
#    'bench_kernels',
#    sources     : 'benchmarks/bench_kernels.cpp',
#    dependencies: [vectorutil_dep, gbench_dep],
#    cpp_args    : ['-O3'],
#  )
#endif
#
#rust_enabled = add_languages('rust', required: false)
#if rust_enabled
#  executable(
#    'rust_driver',
#    'examples/rust_driver.rs',
#    link_with: vectorutil_lib,
#  )
#endif
#
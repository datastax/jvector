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

# Optional ASH add-on: FP32 batch encoding and FP16/INT8 query projection with
# sequential oneDNN. It is a separate project from the Highway library
# (../native) and builds libjvector_ash_fp16, which Java loads on its own;
# libjvector never depends on it.
#
# oneDNN is linked statically so the packaged add-on is one self-contained
# library with no runtime dependency on a system oneDNN; oneDNN's own runtime
# dispatcher still selects the instructions the host CPU supports. Building
# oneDNN takes a few minutes, which is why Maven builds this project only with
# -Dnative.onednn=true (CI and the release profile turn it on).
project('jvector_ash_onednn', 'cpp',
  version: '0.1.0',
  meson_version: '>= 0.55.0',
  default_options: ['cpp_std=c++17']
)

cmake = import('cmake')
onednn_options = cmake.subproject_options()
onednn_options.add_cmake_defines({
  'CMAKE_POSITION_INDEPENDENT_CODE': true,
  'ONEDNN_LIBRARY_TYPE': 'STATIC',
  'ONEDNN_CPU_RUNTIME': 'SEQ',
  'ONEDNN_GPU_RUNTIME': 'NONE',
  'ONEDNN_BUILD_GRAPH': false,
  'ONEDNN_BUILD_TESTS': false,
  'ONEDNN_BUILD_EXAMPLES': false,
  'ONEDNN_ENABLE_PRIMITIVE': 'MATMUL;REORDER',
  'ONEDNN_ENABLE_WORKLOAD': 'INFERENCE',
  'ONEDNN_ARCH_OPT_FLAGS': '',
})
onednn_project = cmake.subproject('onednn', options: onednn_options)
# oneDNN's aggregate archive has no source files of its own. Meson's CMake
# converter marks that aggregate non-PIC, although all constituent object
# libraries are PIC. Re-archive those objects with the correct metadata.
onednn_archive_inputs = [onednn_project.target('dnnl').extract_all_objects(recursive: true)]
# The CMake build links KleidiAI as a separate static target on AArch64.
if 'kleidiai_ops' in onednn_project.target_list()
  onednn_archive_inputs += [onednn_project.target('kleidiai_ops').extract_all_objects(recursive: true)]
endif
onednn_objects = static_library('ash_onednn',
  objects: onednn_archive_inputs,
  pic: true,
)
onednn_dependency = onednn_project.dependency('dnnl').partial_dependency(
  compile_args: true, includes: true, link_args: true)
ash_fp16_lib = shared_library('jvector_ash_fp16',
  sources: ['src/jvector_ash_fp16.cpp', 'src/jvector_ash_int8.cpp'],
  dependencies: onednn_dependency,
  link_with: onednn_objects,
  cpp_args: ['-fvisibility=hidden'],
  link_args: ['-Wl,--exclude-libs,ALL'],
  install: true,
)

gtest_dep = dependency('gtest_main', required: false)
if gtest_dep.found()
  ash_fp16_tests = executable('test_ash_fp16',
    sources: 'tests/test_ash_fp16.cpp',
    include_directories: include_directories('src'),
    dependencies: gtest_dep,
    link_with: ash_fp16_lib,
  )
  test('ash_fp16', ash_fp16_tests, protocol: 'gtest', suite: 'ash_fp16')
endif

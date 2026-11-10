#
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
#

# Overlay build file for the Highway wrap. Highway's own CMake build compiles its
# dynamic-dispatch runtime, which jvector does not use: each ISA tier is compiled
# with HWY_COMPILE_ONLY_STATIC, so only the headers and hwy::Abort are needed.
project('highway', 'cpp')

highway_dep = declare_dependency(
  include_directories: include_directories('.'),
)
# abort.cc was split out of targets.cc for projects that do their own dispatch
# (google/highway#1935).
highway_abort_src = files('hwy/abort.cc')

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

#if defined(__x86_64__) || defined(_M_X64)
#if defined(JV_REQUIRE_HWY_AVX3)
#if HWY_STATIC_TARGET != HWY_AVX3
#error "Highway did not select HWY_AVX3 for the AVX-512 build. Check compiler flags, compiler support, and Highway blocklists."
#endif
#elif defined(JV_REQUIRE_HWY_AVX2)
#if HWY_STATIC_TARGET != HWY_AVX2
#error "Highway did not select HWY_AVX2 for the AVX2 build. Check compiler flags, compiler support, and Highway blocklists."
#endif
#endif //
#endif // __X86_64__

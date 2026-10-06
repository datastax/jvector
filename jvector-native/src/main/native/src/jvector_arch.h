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

// jvector_arch.h — canonical architecture detection macros for jvector-native.
//
// Use JV_ARCH_X86_64 / JV_ARCH_AARCH64 throughout the codebase to guard
// architecture-specific code instead of scattering raw compiler predefined
// macros (__x86_64__, __aarch64__, etc.).  This keeps the guards readable and
// makes it trivial to add a new architecture in one place.
//
// Exactly one of these will be defined to 1 on a supported build; the other
// will be defined to 0.  Unsupported architectures define neither to 1 so that
// #if JV_ARCH_X86_64 / #if JV_ARCH_AARCH64 simply evaluate false.

#ifndef JVECTOR_ARCH_H
#define JVECTOR_ARCH_H

// ---- x86-64 -----------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)
#  define JV_ARCH_X86_64  1
#  define JV_ARCH_AARCH64 0
// ---- AArch64 ----------------------------------------------------------------
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define JV_ARCH_X86_64  0
#  define JV_ARCH_AARCH64 1
// ---- Unsupported ------------------------------------------------------------
#else
#  define JV_ARCH_X86_64  0
#  define JV_ARCH_AARCH64 0
#endif

#endif // JVECTOR_ARCH_H

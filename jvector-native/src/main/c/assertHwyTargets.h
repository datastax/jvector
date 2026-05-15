
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

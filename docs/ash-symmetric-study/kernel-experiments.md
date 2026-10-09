# Encoded symmetric kernel experiments

Cohere 100k, 32X, C=1, one thread, JDK23, Xeon 6985P-C. Each final comparison mode ran in its own JVM. Five timing passes; prior query encoding is excluded for symmetric scoring; asymmetric projection and scorer setup are included once per query. Short scans repeat batches for at least 0.25 seconds per pass. Values below are median ns/pair.

## 2-bit pair implementations

| Implementation | 99,685 targets/query | 64 targets/query |
|---|---:|---:|
| Scalar word popcounts | 63.3 | 90.3 |
| SIMD unpack and multiply | 95.6 | 125.9 |
| SIMD weighted popcounts | 42.4 | 62.0 |
| Asymmetric SIMD single | 119.5 | 980.7 |

The experimental scalar-word implementation used the temporary SIMD entry point for the comparison harness; it executes scalar CPU popcount instructions. It is retained in the scalar fallback. The final SIMD implementation uses weighted vector popcounts. Experimental switches were removed.

## Block implementations

| Bits | Implementation | 99,685 targets/query | 64 targets/query |
|---:|---|---:|---:|
| 2 | Reused floating-point symmetric LUT | 37.9 | 197.6 |
| 2 | Paired-nibble exact integer symmetric LUT | 25.3 | 200.7 |
| 2 | Asymmetric SIMD block | 31.0 | 670.7 |
| 4 | Reused floating-point symmetric LUT | 36.5 | 161.5 |
| 4 | Paired-nibble exact integer symmetric LUT | 23.2 | 127.5 |
| 4 | Asymmetric SIMD block | 30.1 | 363.7 |

## Binary scoring

| Implementation | 99,685 targets/query |
|---|---:|
| symmetric-single-scalar | 13.2 |
| symmetric-single-simd | 12.2 |
| symmetric-block-simd | 14.8 |
| asymmetric-single-simd | 54.0 |
| asymmetric-block-simd | 80.5 |

The integer LUT kernel borrows the C++ fastscan pattern of packed nibble lookups and narrow accumulators. Its values are exact products of encoded integer components; it does not requantize scores. Accumulators widen in bounded chunks to prevent overflow. These are scorer measurements, not graph QPS; the two methods also have different approximation error because symmetric scoring quantizes both operands.

Shared-JVM exploratory runs exhibited large changes in unchanged control timings. They are not used in this table. Independent-process comparisons eliminate cross-mode type-profile pollution, but these are still one-host measurements rather than a universal hardware claim.

# ASH PCA correctness

PCA first computes FP64 covariance with DSYRK and its eigendecomposition with
DSYEVD. This avoids computing and discarding a full tall left-singular-vector
matrix. The input is read-only. Finite values, covariance trace, eigenvalue energy,
the retained tail, the retained/discarded spectral gap and selected eigenpair
residuals are validated before the leading directions are accepted.

Covariance squares the condition number. A retained eigenvalue or cutoff gap
at the scale of accumulated FP64 error causes fallback to direct DGESDD. Zero
energy, wide input and eigenvalue nonconvergence also use direct SVD. Its full
right basis preserves null-space completion for underdetermined training.
Invalid input, invalid native arguments and invalid workspace sizes are reported;
allocation failures are not swallowed. Models and graph formats are unchanged.

Tests compare subspace projectors with an independent direct SVD, including
repeated eigenvalues inside the retained subspace, ties at its boundary,
ill-conditioned and rank-deficient matrices, wide inputs and input preservation.

## Historical DGESVD correctness issue

During the native PQ/ASH graph comparison, JDK 25 MPNet ASH training failed in
native PCA with `DGESVD info=767`. The identical frozen-runtime retry failed too.
The 15,360-by-768 normalized residual matrix was finite, unchanged between calls,
and identical across the JDK 22/25 diagnostics (FNV fingerprint e9c6fe8880b17ba6).

The installed MKL 2020.4.304 backend's multi-threaded, optimal-workspace QR SVD
path also returned success with incorrect results on JDK 22. For the same matrix:

| Driver / configuration | Relative spectral-energy error | Eigenvector residual |
|---|---:|---:|
| DGESVD, optimal workspace, 48 threads | 1.733947% on a returned-success call; other calls failed | 0.635204 |
| DGESVD, minimum workspace, 48 threads | 5.91e-14 | 2.05e-14 |
| DGESDD, overwrite, 48 threads | 6.11e-14 | 1.35e-14 |
| DGESDD, thin, 48 threads | 6.11e-14 | 1.35e-14 |
| DGESVD, optimal workspace, 1 thread | 5.84e-14 | 1.95e-14 |

Input squared Frobenius norm was approximately 15,360. A returned-success QR SVD
reported singular-value squared sum 15,626.33427256156. The independent residual
check evaluated `X^T X v - sigma^2 v` for leading and interior singular vectors
using Java arithmetic, without relying on BLAS for the check.

These observations isolate a native QR SVD fast-path problem, rather than a JDK
selection issue or non-finite training input. They do not establish an upstream
vendor root-cause patch. Positive DGESVD status denotes QR nonconvergence; see
[the LAPACK contract](https://www.netlib.org/lapack/explore-html/d1/d7f/group__gesvd_gac6bd5d4e645049e49bb70691180abf07.html).

The initial fix selected the portable divide-and-conquer DGESDD driver, already
used by ITQ orthogonalization and retained as the direct-SVD fallback. Tall matrices overwrite the scratch input with the
unused left vectors. Wide matrices request full right vectors so projections can
include null-space directions when training has fewer rows than dimensions.
Finite input, workspace size and singular-value energy are validated before
accepting the PCA basis. There are no CPU-, vendor-, bit-width- or JDK-specific
branches or runtime thread-count workarounds.

BenchYAML and its timers are unchanged. Runtime validation uses BenchYAML on
both JDKs with 64-by-768 and 1,024-by-768 training sets, native ASH, deferred writes
and NVQ reranking. All four cases passed, including serial/concurrent result
checks. Added regression sources cover principal axes, wide null-space completion
and non-finite input; they compile for CI. Unit tests were not separately executed
for that initial study; the current regression suite executes the PCA checks.

All graph measurements made with the earlier ASH PCA implementation are retained
as superseded evidence and recollected after the fix. PQ does not use this PCA
path; its completed measurements are retained after verifying that its runtime
classes are unchanged. Earlier ITQ-trained full-scan ASH comparisons also warrant
revalidation wherever they used the affected multi-threaded QR SVD backend.

Existing saved models are not rewritten by this change. Retrain the ASH compressor
and rebuild its graph/model caches to adopt the corrected PCA. The graph study
uses fresh isolated caches for every corrected ASH construction case.

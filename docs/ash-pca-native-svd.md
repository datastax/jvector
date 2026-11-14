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
allocation failures are not swallowed. Model and graph formats are unchanged.

Tests compare subspace projectors with an independent direct SVD, including
repeated eigenvalues inside the retained subspace, ties at its boundary,
ill-conditioned and rank-deficient matrices, wide inputs and input preservation.

## Why PCA avoids the QR SVD driver

Some LAPACK builds return incorrect results from the multi-threaded,
optimal-workspace QR SVD path (DGESVD). With MKL 2020.4, a finite, well-formed
15,360 × 768 training matrix produced either `DGESVD info=767` (QR
nonconvergence; see
[the LAPACK contract](https://www.netlib.org/lapack/explore-html/d1/d7f/group__gesvd_gac6bd5d4e645049e49bb70691180abf07.html))
or a successful return with a 1.7% spectral-energy error. The same matrix was
accurate to about 1e-13 with DGESDD, with DGESVD on minimum workspace, or with
one thread.

PCA therefore never calls DGESVD. The covariance path above is the primary
method, and the divide-and-conquer DGESDD driver, also used by ITQ
orthogonalization, is the fallback. Tall inputs overwrite a scratch copy with the
unused left vectors; wide inputs request full right vectors so projections can
include null-space directions when training has fewer rows than dimensions.
There are no CPU-, vendor-, bit-width- or JDK-specific branches or thread-count
workarounds.

Saved models are not rewritten. Retrain the ASH compressor and rebuild its graph
to adopt the current PCA.

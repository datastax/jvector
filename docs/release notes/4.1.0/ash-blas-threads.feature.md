JVector now sets the native BLAS and LAPACK thread count for its own calls instead
of inheriting MKL's or OpenBLAS's default of one thread per CPU. ASH training uses
one BLAS thread, or as many as the executor passed to the new
`AsymmetricHashing.initialize(..., ForkJoinPool executor)` overload. With the native
provider, the limit is applied per calling thread through MKL or OpenBLAS, so a
database does not need to configure `MKL_NUM_THREADS`. `BlasThreads` applies the
same limit to other BLAS work.

BenchYAML index YAML can set `construction.workers` and `search.workers`, each a
count or `max` for the platform's physical cores. A search worker sweep queries one
index. `benchyaml-play.sh` builds the checkout and runs one experiment in one JVM
from the command line or IntelliJ.

See [ASH configuration and defaults](../../ash-configuration.md#blas-threads).

# Benchmark launcher and server profiles

`jvector-examples/benchyaml-play.sh` runs one
[index-parameters](../yaml-configs/index-parameters) YAML with BenchYAML in one JVM. Every launch
first rebuilds the checkout with Maven, including the native libraries, so results always come
from the current source. The run records the commit, with `-dirty` when tracked files have
uncommitted changes, and the JAR checksum.

```bash
BENCH_RUN_CONFIG=jvector-examples/yaml-configs/ash-pq/run-single-jvm.yml \
  jvector-examples/benchyaml-play.sh \
  jvector-examples/yaml-configs/index-parameters/studies/ash-vs-pq-single-jvm/mpnet10m-2bit-ash.yml
```

The index YAML sets concurrency: `construction.workers` for training, encoding, graph insertion
and writing, and `search.workers` for queries, each a number or `max` (the platform's physical
cores). `search.workers` is a sweep, so `[1, 8]` queries the same index with 1 and then 8 workers.
The launcher sets where the JVM may run, and the run YAML sets what is printed and logged.
JVector sets native BLAS threads itself (the construction pool's size during training), so the
launcher clears `MKL_NUM_THREADS` and the other BLAS thread variables.

`BENCH_RUN_CONFIG` selects the run YAML (default `yaml-configs/run-config.yml`).
`BENCH_DEBUG=<port>` waits for a debugger on that port.

Long construction phases print progress every 10 seconds: vectors inserted with an estimate of
the time left, then graph cleanup and index writing. Set `-Djvector.bench.progressSeconds=0`
to print only each phase's start and finish.

## Server profiles

A profile describes one machine: JDK, working directory root, dataset catalogs and, optionally,
cores and NUMA nodes. CPU lists come from `lscpu`, one hardware thread per physical core.
Copy [example.env](example.env) to `local/profile.env`, or keep several profiles in
`local/<name>.env` and pick one with `BENCH_PROFILE=<name>`. The `local/` directory is ignored
by git because profiles hold machine paths. Each recipe runs in its own directory under
`BENCH_WORK_ROOT`.

## IntelliJ

The repository's `.run/` directory holds run configurations for experiments. Each names a recipe
and a run YAML, and takes machine settings from your profile. To add one for another recipe,
copy an existing file, then change its name and the recipe path in `SCRIPT_OPTIONS`; every case in
[ash-vs-pq-single-jvm](../yaml-configs/index-parameters/studies/ash-vs-pq-single-jvm) runs with
`run-single-jvm.yml`.

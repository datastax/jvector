# Recall vs QPS: graph against IVF-ASH

How to produce a single chart of Recall@10 against QPS for cap-1M, with these curves:

- the jvector graph, as configured in production: fused PQ (`mFactor` 8) with NVQ reranking, with `topKOverquery` swept;
- jvector + fastervector IVF-ASH, through the FFM bindings, with `nprobe` swept at nlist 1024 and 4096;
- optionally, the same IVF-ASH builds run through fastervector's own `fvec-bench`, so the gap between the two IVF curves shows what the FFM layer costs.

The runs happen on the Linux benchmark box. The plot can be made anywhere that has Python 3 with `pandas` and `matplotlib`.

## 1. Set up the environment

Set the variables that the `jvector-fastervector` README describes, using the paths on your machine:

```bash
export JVECTOR_HOME=/path/to/jvector            # checkout on the fvec-bindings branch
export ONEAPI_SETVARS=/opt/intel/oneapi/setvars.sh
export PHYS_CORES=$(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)

cd "$JVECTOR_HOME"
git checkout fvec-bindings && git pull
source "$ONEAPI_SETVARS" >/dev/null 2>&1 || true
```

## 2. Build the jars

```bash
./mvnw -Pfastervector clean package -DskipTests
ls jvector-examples/target/*-jar-with-dependencies.jar \
   jvector-fastervector-bench/target/*-jar-with-dependencies.jar
```

You need `-Pfastervector`. Without it, the IVF bench jar isn't rebuilt and an older one stays in `target/`. Check that each directory holds only one `*-jar-with-dependencies.jar`, so the `*` in the commands below can't pick up a stale jar.

## 3. Turn on CSV logging

In `jvector-examples/yaml-configs/run-config.yml`, uncomment the `logging:` section. Both runners read this file. With logging on, each run writes one `experiments.csv` row per search point. At startup, each run prints `Logging enabled: <run dir>`, and that directory is where its `experiments.csv` goes.

## 4. Run the graph sweep

```bash
java --enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector \
     -Djvector.enable_native_vectorization=true \
     -Xmx124g \
     -cp jvector-examples/target/jvector-examples-*-jar-with-dependencies.jar \
     io.github.jbellis.jvector.example.BenchYAML \
     cap-1M-recall-qps-sweep.yml
```

- **Config:** `jvector-examples/yaml-configs/index-parameters/cap-1M-recall-qps-sweep.yml`. Pass the bare file name, as above. If the file isn't found, BenchYAML silently falls back to `default.yml`.
- **Runner:** use `BenchYAML`, not `AutoBenchYAML`. The Auto runner averages all search points into one row.

## 5. Run the IVF sweep (jvector + fastervector)

Run this in its own JVM, never in the same JVM as the graph run:

```bash
java --enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector \
     -Djvector.enable_native_vectorization=false \
     -Divf.perQuery=false \
     -Divf.buildThreads=$PHYS_CORES \
     -Xmx124g \
     -cp jvector-fastervector-bench/target/jvector-fastervector-bench-*-jar-with-dependencies.jar \
     io.github.jbellis.jvector.fastervector.bench.BenchIvfYAML \
     jvector-fastervector-bench/yaml-configs/ivf-experiments/cap-1M-recall-qps-sweep.yml
```

- **Startup check:** the log should print `Per-query throughput off`. If it doesn't, the jar is out of date.
- **Index reuse:** the indexes are written under `ivf_cache/`, and reruns reopen them. To put them on another disk, add `-Divf.ssdPaths=<dir>`.

## 6. Optional: run the same IVF sweep through fvec-bench

Run this from a fastervector checkout on the box. To isolate the FFM cost, check out the fastervector commit that jvector's bindings are pinned to.

```bash
cd /path/to/fastervector
source "$ONEAPI_SETVARS" >/dev/null 2>&1 || true
cargo build -p fvec-bench --release

SUITE=$JVECTOR_HOME/jvector-fastervector-bench/yaml-configs/ivf-experiments/fvec-bench-cap-1M-recall-qps-sweep.yaml

# cap-1M has to resolve; if it doesn't, add --catalog <a catalog that lists it> to the next commands
target/release/fvec-bench datasets list | grep -i cap-1M

target/release/fvec-bench preflight "$SUITE" --cases
target/release/fvec-bench run "$SUITE" --ssd-dir <same disk as ivf_cache>/fvbench
```

When the run finishes, it prints its run directory. The file to plot is that directory's `results.csv`.

## 7. Collect the CSVs

Copy the CSVs to the machine you'll plot on, giving each a distinct name:

| Source | Copy as |
|---|---|
| Graph run: `<run dir>/experiments.csv` | `BenchYamlexperiments.csv` |
| IVF run: `<run dir>/experiments.csv` | `BenchIVFYamlexperiments.csv` |
| fvec-bench (optional): `<run dir>/results.csv` | `results.csv` |

## 8. Plot

```bash
cd "$JVECTOR_HOME"
python3 jvector-fastervector-bench/scripts/plot_recall_qps.py \
    ~/Downloads/BenchYamlexperiments.csv \
    ~/Downloads/BenchIVFYamlexperiments.csv \
    ~/Downloads/recall_vs_qps.png \
    --fvec ~/Downloads/results.csv
```

- Leave out `--fvec` if you skipped step 6.
- The IVF curves use batched QPS by default. To plot coalesced QPS instead, add `--ivf-qps-column search.throughput.coalesced.avg_qps`.
- How the chart is drawn:
  - one color per graph build and per IVF nlist;
  - fvec-bench curves are dashed, in the color of the matching jvector IVF curve, or dotted for `prod_qd` 0 (the classic I/O producer, which jvector doesn't use);
  - every other point is labelled with its `oq` (overquery) or `np` (nprobe) value;
  - QPS is on a log scale.

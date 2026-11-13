# How to run benchmarks

JVector runs dataset benchmarks with BenchYAML. See the
[BenchYAML Guide](benchyaml-guide.md) for the construction and search fields.

To run a benchmark
- Decide which dataset(s) you want to benchmark. A dataset consists of
    - The vectors to be indexed, usually called the "base" or "target" vectors
    - The query vectors
    - The "ground truth" results that are used to compute accuracy metrics
    - The similarity metric used compute the ground truth (dot product, cosine similarity or L2 distance)
- Configure the parameters combinations for which you want to run the benchmark. This includes index construction parameters, quantization parameters and search parameters.

JVector supports datasets in the fvecs/ivecs format.  These consist of three files, for example `base.fvecs`, `queries.fvecs` and `neighbors.ivecs` containing the base vectors, query vectors, and ground truth. (`fvecs` and `ivecs` file formats are described [here](http://corpus-texmex.irisa.fr/))

The general procedure for running benchmarks is mentioned below. The following sections describe the process in more detail.
- [Specify the dataset](#specifying-datasets) names to benchmark in `datasets.yml`.
- Certain datasets will be downloaded automatically. If using a different dataset, make sure the dataset files are downloaded and made available (refer the section on [Custom datasets](#custom-datasets)).
- Adjust the benchmark parameters in `default.yml`. This will affect the parameters for all datasets benchmarked. You can specify custom parameters for a specific dataset by creating a file called `<your-dataset-name>.yml` in the `index-parameters` subfolder.
- Decide on the kind of measurements and logging you want and configure them in `run-config.yml`.

You can run the configured benchmark with maven:
```sh
mvn clean compile exec:exec@bench -pl jvector-examples -am
```

## Specifying dataset(s)

The datasets you want to benchmark should be specified in `jvector-examples/yaml-configs/datasets.yml`. You'll notice this file already contains some entries; these are datasets that `bench` can automatically download and test with minimal additional configuration.  Running `bench` without arguments and without changing this file will cause ALL the datasets to be benchmarked one by one (this is probably not what you want).

To benchmark a single dataset, comment out the entries corresponding to all other datasets. (Or provide command line arguments as described in [Running `bench` from the command line](#running-bench-from-the-command-line))

Datasets are grouped into categories. The categories can be arbitrarily chosen for convenience and are not currently considered by the benchmarking system.

Dataset similarity functions are configured in `jvector-examples/yaml-configs/dataset-metadata.yml`.

Example `datasets.yml`:

```yaml
category0:
  - my-dataset-a
  - my-dataset-b
some-other-category:
  - another-dataset-a
  - another-dataset-b
```

## Setting benchmark parameters

### default.yml / \<dataset-name\>.yml

`jvector-examples/yaml-configs/index-parameters/default.yml` specifies the default index construction and search parameters to be used by `bench` for all datasets.

You can specify a custom set of a parameters for any given dataset by creating a file called `<dataset-name>.yml`, with `<dataset-name>` replaced by the actual name of the dataset. This is the same as the identifier used in `datasets.yml`. The format of this file is exactly the same as `default.yml`. The name must match exactly, including case, and the file's `dataset:` field must name the same dataset; BenchYAML stops with an error otherwise. Place the file directly in `index-parameters`, not a subfolder. Subfolders hold [example pipelines](../jvector-examples/yaml-configs/index-parameters/README.md) that you run by path.

See the [BenchYAML Guide](benchyaml-guide.md) for the field reference and an
example index configuration.

Most parameters can be specified as an array. For these parameters, a separate benchmark is run for each value of the parameter. If multiple parameters are specified as arrays, a benchmark is run for each combination (i.e. taking the Cartesian product). For example:
```yaml
construction:
  outDegree: [32, 64]
  efConstruction: [100, 200]
```
will build and benchmark four graphs, one for each combination of M and ef in {(32, 100), (64, 100), (32, 200), (64, 200)}. This is particularly useful when running a Grid search to identify the best performing parameters.

### run-config.yml

This file contains configurations for
- Specifying the measurements you want to report, like QPS, latency and recall
- Specifying how they are measured, such as warmup, timed passes and the page cache state
- Specifying where to output these measurements, i.e. to the console, or to a file, or both.

The configurations in this file are "run-level", meaning that they are shared across all the datasets being benchmarked.
Select another file with `-Djvector.bench.runConfig=<path>`.

See [Run configuration](benchyaml-guide.md#run-configuration) for every option, `run-config.yml` for the
default, and `run-config-larger-than-memory.yml` for measuring an index served from storage.

## Running `bench` from the command line

Once configured to your liking, you can run the benchmark through maven:
```sh
mvn compile exec:exec@bench -pl jvector-examples -am
```

To benchmark a subset of the datasets in `datasets.yml`, you can provide a space-separated list of regexes as arguments.
```sh
# matches `glove-25-angular`, `glove-50-angular`, `nytimes-256-angular` etc
mvn compile exec:exec@bench -pl jvector-examples -am -DbenchArgs="glove nytimes"
```

## Custom Datasets

Datasets are configured via YAML catalog files under `jvector-examples/yaml-configs/dataset-catalogs/`. The loader recursively discovers all `.yaml`/`.yml` files in that directory tree. See `jvector-examples/yaml-configs/dataset-catalogs/local-catalog.yaml` for the full format reference.

To add a custom fvecs/ivecs dataset:

1. Add a `.yaml` file to the YAML catalog directory, mapping your dataset name to its files:
    ```yaml
    _defaults:
      cache_dir: ${DATASET_CACHE_DIR:-dataset_cache}

    my-dataset:
      base: my_base_vectors.fvecs
      query: my_query_vectors.fvecs
      gt: my_ground_truth.ivecs
    ```
2. Place your fvecs/ivecs files at the paths you specified in the YAML (or specify a `cache_dir` / `base_url` to fetch them from a remote source).
3. Add the dataset's similarity function to `jvector-examples/yaml-configs/dataset-metadata.yml`:
    ```yaml
    my-dataset:
      similarity_function: COSINE
      load_behavior: NO_SCRUB
    ```
4. Add the dataset name to `jvector-examples/yaml-configs/datasets.yml` so BenchYAML can find it:
    ```yaml
    custom:
      - my-dataset
    ```

For remote datasets, use `base_url` to specify where files should be downloaded from. The `${VAR}` and `${VAR:-default}` syntax is supported for environment variable expansion. See the example config for details.

## Final index serialization

For an on-disk index, set `construction.writerMode` to `sequential` or
`parallel` in the index parameters. Both writers serialize the completed graph
after cleanup. The parallel writer defaults to 64 records per batch. See the
[BenchYAML Guide](benchyaml-guide.md#construction) for configuration and
[Deferred batch serialization](deferred-batch-writer.md) for writer details.

## Fused PQ and ASH comparison

[Reproducing ASH versus PQ](ash-pq-benchmark.md) provides byte-matched recipes
for the 1M and the 6M/10M datasets, scripts that run one case or a whole study,
and the measurement protocol. [ASH performance](ash-performance.md) reports the
results.

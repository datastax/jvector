# How to run benchmarks

JVector comes with a built-in benchmarking system in `jvector-examples/.../BenchYAML.java`.

To run a benchmark
- Decide which dataset(s) you want to benchmark. A dataset consists of
    - The vectors to be indexed, usually called the "base" or "target" vectors.
    - The query vectors.
    - The "ground truth" results which are used to compute accuracy metrics.
    - The similarity metric which should have been used to compute the ground truth (dot product, cosine similarity or L2 distance).
- Configure the parameters combinations for which you want to run the benchmark. This includes graph index parameters, quantization parameters and search parameters.

JVector supports two types of datasets:
- **Fvec/Ivec**: The dataset consists of three files, for example `base.fvec`, `queries.fvec` and `neighbors.ivec` containing the base vectors, query vectors, and ground truth. (`fvec` and `ivec` file formats are described [here](http://corpus-texmex.irisa.fr/))
- **HDF5**: The dataset consists of a single HDF5 file with three datasets labelled `train`, `test` and `neighbors`, representing the base vectors, query vectors and the ground truth.

The general procedure for running benchmarks is mentioned below. The following sections describe the process in more detail.
- [Specify the dataset](#specifying-datasets) names to benchmark in `datasets.yml`.
- Certain datasets will be downloaded automatically. If using a different dataset, make sure the dataset files are downloaded and made available (refer the section on [Custom datasets](#custom-datasets)).
- Adjust the benchmark parameters in `default.yml`. This will affect the parameters for all datasets to be benchmarked. You can specify custom parameters for a specific dataset by creating a file called `<your-dataset-name>.yml` in the same folder.
- Decide on the kind of measurements and logging you want and configure them in `run-config.yml`.

You can run the configured benchmark with maven:
```sh
mvn clean compile exec:exec@bench -pl jvector-examples -am
```

## Specifying dataset(s)

The datasets you want to benchmark should be specified in `jvector-examples/yaml-configs/datasets.yml`. You'll notice this file already contains some entries; these are datasets that `bench` can automatically download and test with minimal additional configuration.  Running `bench` without arguments and without changing this file will cause ALL the datasets to be benchmarked one by one (this is probably not what you want).

To benchmark a single dataset, comment out the entries corresponding to all other datasets. (Or provide command line arguments as described in [Running `bench` from the command line](#running-bench-from-the-command-line))

Datasets are assumed to be Fvec/Ivec based unless the entry in the `datasets.yml` ends with `.hdf5`. In this case, `.hdf5` is not considered part of the "dataset name" referenced in other sections.

You'll notice that datasets are grouped into categories. The categories can be arbitrarily chosen for convenience and are not currently considered by the benchmarking system.

Dataset similarity functions are configured in `jvector-examples/yaml-configs/dataset-metadata.yml`.

Example `datasets.yml`:

```yaml
category0:
  - my-fvec-dataset                      # fvec/ivec dataset, cosine similarity
  - my-hdf5-dataset-angular.hdf5         # hdf5 dataset, cosine similarity
some-other-category:
  - a-huge-dataset-1024d-euclidean.hdf5  # hdf5 dataset, L2 similarity
  - my-simple-dataset-dot.hdf5           # hdf5 dataset, dot product similarity
  - some-dataset-euclidean               # fvec/ivec dataset, cosine similarity (NOT L2 unless you change the code!)
```

## Setting benchmark parameters

### default.yml / \<dataset-name\>.yml

`jvector-examples/yaml-configs/default.yml` specifies the default index construction and search parameters to be used by `bench` for all datasets.

You can specify a custom set of a parameters for any given dataset by creating a file called `<dataset-name>.yml`, with `<dataset-name>` replaced by the actual name of the dataset. This is the same as the identifier used in `datasets.yml`, but without the `.hdf5` suffix for hdf5 datasets. The format of this file is exactly the same as `default.yml`.

Refer to `default.yml` for a list of all options.

Most parameters can be specified as an array. For these parameters, a separate benchmark is run for each value of the parameter. If multiple parameters are specified as arrays, a benchmark is run for each combination (i.e. taking the Cartesian product). For example:
```yaml
construction:
  M: [32, 64]
  ef: [100, 200]
```
will build and benchmark four graphs, one for each combination of M and ef in {(32, 100), (64, 100), (32, 200), (64, 200)}. This is particularly useful when running a Grid search to identify the best performing parameters.

### run-config.yml

This file contains configurations for
- Specifying the measurements you want to report, like QPS, latency and recall
- Specifying where to output these measurements, i.e. to the console, or to a file, or both.

The configurations in this file are "run-level", meaning that they are shared across all the datasets being benchmarked.

See `run-config.yml` for a full list of all options.

## Running `bench` from the command line

Once configured to your liking, you can run the benchmark through maven:
```sh
mvn compile exec:exec@bench -pl jvector-examples -am
```

To benchmark a subset of the datasets in `datasets.yml`, you can provide a space-separated list of regexes as arguments.
```sh
# matches `glove-25-angular.hdf5`, `glove-50-angular.hdf5`, `nytimes-256-angular.hdf5` etc
mvn compile exec:exec@bench -pl jvector-examples -am -DbenchArgs="glove nytimes"
```

## Custom Datasets

Datasets are configured via YAML catalog files under `jvector-examples/datasets/`. The loader recursively discovers all `.yaml`/`.yml` files in that directory tree. See `jvector-examples/datasets/public/example_datasets_config.yaml` for the full format reference.

To add a custom fvec/ivec dataset:

1. Create a directory under `jvector-examples/datasets/` (e.g. `custom/mydata/`).
2. Add a `.yaml` file mapping your dataset name to its files:
    ```yaml
    _defaults:
      cache_dir: ${DATASET_CACHE_DIR:-dataset_cache}

    my-dataset:
      base: my_base_vectors.fvecs
      query: my_query_vectors.fvecs
      gt: my_ground_truth.ivecs
    ```
3. Place your fvec/ivec files in the same directory (or specify a `cache_dir` / `base_url` to fetch them from a remote source).
4. Add the dataset's similarity function to `jvector-examples/yaml-configs/dataset-metadata.yml`:
    ```yaml
    my-dataset:
      similarity_function: COSINE
      load_behavior: NO_SCRUB
    ```
5. Add the dataset name to `jvector-examples/yaml-configs/datasets.yml` so BenchYAML can find it:
    ```yaml
    custom:
      - my-dataset
    ```

For remote datasets, use `base_url` to specify where files should be downloaded from. The `${VAR}` and `${VAR:-default}` syntax is supported for environment variable expansion. See the example config for details.

# How to run benchmarks

JVector comes with a built-in benchmarking system in `jvector-examples/.../BenchYAML.java`.

To run a benchmark
- Decide which dataset(s) you want to benchmark.
- Configure the parameters combinations for which you want to run the benchmark. This includes graph index parameters, quantization parameters and search parameters.

To describe a dataset you will need to specify the base vectors used to construct the index, the query vectors, and the "ground truth" results which will be used to compute accuracy metrics.

JVector supports two types of datasets:
- **Fvec/Ivec**: The dataset consists of three files, for example `base.fvec`, `queries.fvec` and `neighbors.ivec`.
- **HDF5**: The dataset consists of a single HDF5 file with three datasets labelled `train`, `test` and `neighbors`, representing the base vectors, query vectors and the ground truth.

General procedure for running benchmarks:
- Specify the dataset names to benchmark in `datasets.yml`.
- Certain datasets will be downloaded automatically. If using a different datasets, make sure the dataset files are downloaded and made available (refer the section on [using datasets](#using-datasets)).
- Adjust the benchmark parameters in `default.yml`. This will affect the parameters for all datasets to be benchmarked. You can specify custom parameters for a specific dataset by creating a file called `<dataset-name>.yml` in the same folder.

You can run the configured benchmark with maven:
```sh
mvn clean compile exec:exec@bench -pl jvector-examples -am
```

## Using Datasets

### Using Fvec/Ivec datasets

Using fvec/ivec datasets requires them to be configured in `MultiFileDatasource.java`. Some datasets are already pre-configured; these will be downloaded and used automatically on running the benchmark.

To use a custom dataset consisting of files `base.fvec`, `queries.fvec` and `neighbors.ivec`, do the following:
- Ensure that you have three files:
    - `base.fvec` containing N D-dimensional float vectors. These are used to build the index.
    - `queries.fvec` containing Q D-dimensional float vectors. These are used for querying the built index.
    - `neighbors.ivec` containing Q K-dimensional integer vectors, one for each query vector, representing the exact K-nearest neighbors for that query among the base vectors.
    The files can be named however you like.
- Save all three files somewhere in the `fvec` directory in the root of the `jvector` repo (if it doesn't exist, create it). It's recommended to create at least one sub-folder with the name of the dataset and copy or move all three files there.
- Edit `MultiFileDatasource.java` to configure a new dataset and it's associated files:
    ```java
    put("cust-ds", new MultiFileDatasource("cust-ds",
            "/cust-ds/base.fvec",
            "/cust-ds/query.fvec",
            "/cust-ds/neighbors.ivec"));
    ```
    The file paths are resolved relative to the `fvec` directory. `cust-ds` is the name of the dataset and can be changed to whatever is appropriate.
- In `jvector-examples/yaml-configs/datasets.yml`, add an entry corresponding to your custom dataset. Comment out other datasets which you don't want to benchmark.
    ```yaml
    custom:
      - cust-ds
    ```

### Using HDF5 datasets

HDF5 datasets consist of a single file. The Hdf5Loader looks for three HDF5 datasets within the file, `train`, `test` and `neighbors`. These correspond to the base, query and neighbors vectors described above for fvec/ivec files.

To use an HDF5 dataset, edit `jvector-examples/yaml-configs/datasets.yml` to add an entry like the following:
```yaml
category:
  - dataset-name.hdf5
```

BenchYAML looks for hdf5 datasets with the name `dataset-name.hdf5` in the `hdf5` folder in the root of this repo. If the file doesn't exist, BenchYAML will attempt to automatically download the dataset from ann-benchmarks.com. To use a custom dataset, simply ensure that the dataset is available in the `hdf5` folder and edit `datasets.yml` accordingly.

## Setting benchmark parameters

Benchmark configurations are defined in `jvector-examples/yaml-configs`. There are three types of files:
- `datasets.yml` which controls which datasets will be used for running the benchmark.
- `default.yml` which defines the default parameter sets to be used for all datasets.
- `dataset-name.yml` which specifies the parameter sets for a single dataset.

### datasets.yml

This file specifies the datasets to be used when running `BenchYAML`. Datasets are grouped into categories. The categories can be arbitrarily chosen for convenience and are not currently considered by the benchmarking system.

### default.yml / \<dataset-name\>.yml

These files define the parameters to be used by `BenchYAML`. The settings in the `default.yml` file apply to all datasets, except ones which have a custom configuration defined in `<dataset-name>.yml`.

See `default.yml` for a list of all options.

Most parameters can be specified as an array. For these parameters, a separate benchmark is run for each value of the parameter. If multiple parameters are specified as arrays, a benchmark is run for each combination (i.e. taking the Cartesian product). For example:
```yaml
construction:
  M: [32, 64]
  ef: [100, 200]
```
will build and benchmark four graphs, one for each combination of M and ef in {(32, 100), (64, 100), (32, 200), (64, 200)}. This is useful when running a Grid search to identify the best performing parameters.


<!-- TODO Bench args -->

# Hosting Datasets

You can host and distribute your datasets remotely as long as they are available 
via HTTPS or S3. This guide tells you how to do this.

## Directory layout

All `.yaml` and `.yml` files under this directory tree are discovered automatically at startup.
This is configured in DataSets.java as a loader parameter.

```
datasets/
  public/
    jvector-public-datasets.yaml      # _includes the public S3 catalog
    example_datasets_config.yaml       # reference/template with all options documented
  custom/
    my-team-datasets.yaml              # (you create this) your own datasets
```

## Quick start

### Using public datasets

Public datasets work out of the box. The file `public/jvector-public-datasets.yaml` uses
`_include` to pull the dataset catalog from S3, and files are downloaded on first use:

```sh
# see what's available
curl -L https://jvector-datasets-public.s3.us-east-1.amazonaws.com/datasets-clean/catalog_entries.yaml
```

Downloaded files are cached locally in `dataset_cache/public/` by default.
Set the `DATASET_CACHE_DIR` environment variable to change this location.

### Adding your own local datasets

1. Create a `.yaml` file anywhere under this directory (e.g. `custom/my-datasets.yaml`).
2. Map each dataset name to its three files:

```yaml
my-dataset:
  base: /path/to/base_vectors.fvecs
  query: /path/to/query_vectors.fvecs
  gt: /path/to/ground_truth.ivecs
```

3. Add the appropriate settings to these files as well, so BenchYAML can use the datasets.
   - `jvector-examples/yaml-configs/dataset-metadata.yml`:
   - `jvector-examples/yaml-configs/datasets.yml`

### Hosting remote datasets

You can host datasets on any S3 bucket or HTTPS server. Each dataset needs three files
in fvecs/ivecs format (base vectors, query vectors, ground truth indices).

**Option A: Use `_include` to reference a remote catalog**

Create a thin local YAML that pulls entries from a remote `catalog_entries.yaml`:

```yaml
_defaults:
  cache_dir: ${DATASET_CACHE_DIR:-dataset_cache}/my-remote

_include:
  url: s3://my-bucket/datasets/catalog_entries.yaml
```

The remote catalog lists dataset entries in the same format. Its base path (the directory
containing the catalog file) is used as the default `base_url` for all included entries.

**Option B: Use `base_url` per entry or in `_defaults`**

```yaml
_defaults:
  base_url: s3://my-bucket/datasets/
  cache_dir: ${DATASET_CACHE_DIR:-dataset_cache}/my-remote

ada002-100k:
  base: ada_002_100k_base.fvecs
  query: ada_002_100k_query.fvecs
  gt: ada_002_100k_gt.ivecs
```

File paths are appended to `base_url` for downloading. Files in subdirectories work too
(e.g. `base: subdir/file.fvecs` downloads from `s3://my-bucket/datasets/subdir/file.fvecs`).

### Private datasets with secret paths

Use `${VAR}` env var expansion to keep secrets out of committed files:

```yaml
_defaults:
  base_url: s3://my-bucket/${DATASET_SECRET_HASH}/
  cache_dir: ${DATASET_CACHE_DIR:-dataset_cache}/private

dpr-1M:
  base: dpr/base.fvecs
  query: dpr/query.fvecs
  gt: dpr/gt.ivecs
```

Set `DATASET_SECRET_HASH` in your environment. The `${VAR:-default}` syntax provides a
fallback value when the variable is not set.

## Catalog file reference

### Required fields (per dataset entry)

| Field   | Description |
|---------|-------------|
| `base`  | Path to base vectors file (`.fvecs`) |
| `query` | Path to query vectors file (`.fvecs`) |
| `gt`    | Path to ground truth indices file (`.ivecs`) |

### Optional fields

| Field       | Description |
|-------------|-------------|
| `base_url`  | Remote URL (S3 or HTTPS) to download files from when not cached locally |
| `cache_dir` | Local directory for cached files (relative or absolute path) |

### Special entries

| Key          | Description |
|--------------|-------------|
| `_defaults`  | Default values folded into all dataset entries in the same file. Entry-level values take precedence. |
| `_include`   | Contains a `url` field pointing to a remote catalog. Remote entries are fetched and merged with local `_defaults`. |
| `_*`         | Any root key starting with `_` is excluded from dataset names. |

### Environment variables

- Field values support `${VAR}` and `${VAR:-default}` syntax (bash-style).
- `${VAR}` expands to the environment variable value; throws an error if not set.
- `${VAR:-default}` uses the default when the variable is not set (including `${VAR:-}` for empty string).
- The `DATASET_CACHE_DIR` environment variable sets a global default `cache_dir` when none is specified at the entry or `_defaults` level.

### Cache directory resolution order

1. `cache_dir` on the dataset entry
2. `cache_dir` in `_defaults`
3. `DATASET_CACHE_DIR` environment variable
4. The directory containing the catalog YAML file

### Supported transport protocols

- **S3** (`s3://bucket/path`) -- uses the AWS SDK with anonymous credentials
- **HTTPS** (`https://host/path`) -- uses Java's built-in HTTP client
- **Local files** -- no download; files are read directly from the resolved path

# Sharing a Dataset

To share a dataset on a hosted URL:

1. **Prepare your files** -- you need three files in fvecs/ivecs format:
   - `base_vectors.fvecs` -- the vectors to index
   - `query_vectors.fvecs` -- the vectors to search with
   - `ground_truth.ivecs` -- the known nearest neighbor indices for each query

2. **Upload them** to an S3 bucket or HTTPS-accessible location.

3. **Create a catalog file** (any `.yaml` file) listing the dataset:
   ```yaml
   _defaults:
     base_url: https://my-server.com/datasets/

   my-dataset:
     base: my_base_vectors.fvecs
     query: my_query_vectors.fvecs
     gt: my_ground_truth.ivecs
   ```

4. **Distribute the catalog file.** Recipients drop it into their
   `jvector-examples/datasets/` directory and the loader picks it up automatically.
   Files are downloaded and cached locally on first use.

For private datasets, use `${VAR}` in the `base_url` to keep secret paths out of the file:
```yaml
_defaults:
  base_url: s3://my-bucket/${SECRET_HASH}/
```

See [datasets.md](datasets.md) for the full configuration reference.

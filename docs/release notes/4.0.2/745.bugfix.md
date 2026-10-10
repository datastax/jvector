### Publish S3 Dataset Downloads Only After Completion

**Description**

Download S3 dataset files into private temporary files and atomically publish the
cache path after successful completion and content-length validation. This prevents
other benchmark processes from reading an in-progress download or losing a completed
cache file when another download attempt fails.

**How to Enable**

No configuration is required. This applies to S3 downloads made by the example
benchmark dataset loader; HTTP downloads already use atomic publication.

**Notes**

Normal failures remove their temporary files. A hard process termination may leave
an unreferenced temporary file, which is not accepted as cached data. Existing cache
files are not retroactively validated; remove any files known to come from an
interrupted download before reusing that cache.

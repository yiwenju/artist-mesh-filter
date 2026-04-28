# S3 Download Bottleneck Fix

## Problem

The pipeline processes ~4.1M mesh files from S3. Each file download uses
`subprocess.run(['aws', 's3', 'cp', ...])`, which spawns a full AWS CLI process
per file. The CLI is itself a Python program, so every call pays:

| Overhead source             | Cost        |
|-----------------------------|-------------|
| Fork + exec subprocess      | ~100 ms     |
| Python interpreter startup   | ~150 ms     |
| Import boto3/botocore/plugins| ~200 ms     |
| Credential resolution        | ~50–100 ms  |
| TLS handshake to S3          | ~50–100 ms  |
| **Total fixed overhead**     | **~500–1000 ms** |

The actual data transfer for a typical mesh file is only ~20–50 ms. So **90%+ of
download time was startup overhead**, and this overhead is per-file regardless of
file size.

With 16 workers, observed throughput: **~2.4 files/s** (vs theoretical ~10.7).

## Fix

Replaced the CLI subprocess with a **persistent `boto3` S3 client** per worker
process (`filter_objaverse.py`).

Each worker now initializes a single `boto3.client('s3')` in `_init_worker()`
and reuses it for every download. This means:

- **Zero process-spawn overhead** — no fork/exec per file
- **Connection reuse** — TCP/TLS connections stay alive via HTTP keep-alive
- **Credential caching** — credentials are resolved once at startup
- **Adaptive retries** — built-in retry with backoff on transient errors

The client is configured with `max_pool_connections=10` and
`retries={'max_attempts': 3, 'mode': 'adaptive'}`.

### Before

```python
proc = subprocess.run(
    ['aws', 's3', 'cp', s3_path, tmp_path, '--quiet'],
    capture_output=True, timeout=120,
)
```

### After

```python
bucket, key = _parse_s3_uri(s3_path)
_s3.download_file(bucket, key, tmp_path)
```

## Expected Impact

Per-file download time drops from ~0.5–1 s to ~0.03–0.1 s. With 16 workers and
a total per-file time of ~0.55 s (down from ~1.5 s), expected throughput is
**~25–30 files/s** — roughly a **10x improvement**.

## Further Optimizations (if needed)

1. **Increase workers to 32–48**: With download overhead eliminated, the
   bottleneck shifts to CPU (trimesh + feature extraction). More workers on
   multi-core nodes will scale linearly.

2. **Producer-consumer prefetch**: Dedicated download threads feeding a queue to
   processing workers, fully overlapping I/O with compute.

3. **S3 VPC gateway endpoint**: If the SLURM cluster runs on AWS, a VPC
   endpoint keeps S3 traffic on the AWS backbone instead of traversing the
   public internet.

4. **In-memory loading**: Stream files into `BytesIO` via
   `s3.download_fileobj()` and pass to `trimesh.load(file_obj, file_type=ext)`
   to skip temp file disk I/O. Works for GLB/STL/PLY; less reliable for OBJ
   (which references .mtl files).

## Dependencies

`boto3` and `botocore` — already installed on any system with the AWS CLI.
No new packages required.

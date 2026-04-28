# Pipeline Optimizations

Additional optimizations beyond the S3 boto3 fix (see `s3_fix_summary.md`).
All changes are on branch `fix/s3-boto3-persistent-client`.

---

## 1. Intermediate checkpointing within batches

**File:** `filter_objaverse.py`
**Impact:** Crash resilience — prevents losing up to 50K results on failure

### Problem

If a SLURM job crashed at file 49,000 of 50,000, all results were lost.
The skip-if-output-exists logic only worked at the batch level. Progress
was printed but never saved.

### Fix

Every 10,000 files, results are written to a `.checkpoint.parquet` file
alongside the output. On restart, the checkpoint is loaded, already-processed
paths are skipped, and processing resumes where it left off. The checkpoint
is deleted once the batch completes successfully.

```
batch_42.parquet              ← final output (written on completion)
batch_42.checkpoint.parquet   ← partial results (written every 10K, deleted on success)
```

---

## 2. Chunked future submission with backpressure

**File:** `filter_objaverse.py`
**Impact:** Lower memory usage, better scheduling

### Problem

All 50,000 futures were submitted at once:
```python
futures = {pool.submit(_process_one, s3p): s3p for s3p in s3_paths}
```

This created 50K `Future` objects in memory upfront with no flow control.

### Fix

Uses a sliding window: `num_workers * 4` futures are in flight at any time.
As futures complete, new ones are submitted from the iterator. This keeps
memory bounded and gives the scheduler better visibility into what's ready.

```python
# Submit initial batch
for s3p in itertools.islice(path_iter, submit_size):
    pending.add(pool.submit(_process_one, s3p))

# Refill as futures complete
while pending:
    done_futures, pending = wait(pending, return_when=FIRST_COMPLETED)
    ...
    for s3p in itertools.islice(path_iter, len(done_futures)):
        pending.add(pool.submit(_process_one, s3p))
```

---

## 3. Explicit temp directory

**File:** `filter_objaverse.py`
**Impact:** Prevents `/tmp` overflow on nodes with small tmpfs

### Problem

`tempfile.NamedTemporaryFile()` defaults to `/tmp`, which on many SLURM nodes
is a small tmpfs in RAM. With 16 workers downloading concurrently (some
ObjaverseXL files are 100MB+), this could fill up and cause cascading failures.

### Fix

Temp files now respect `$TMPDIR` (standard on SLURM), falling back to `/tmp`:

```python
TEMP_DIR = os.environ.get('TMPDIR', '/tmp')
# ...
tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=TEMP_DIR)
```

---

## 4. Vectorized UV island computation

**File:** `topology_features.py`
**Impact:** ~2-5x faster for high-poly meshes (100K+ faces)

### Problem

Two hot Python loops iterated over every UV coordinate and every face edge:

```python
# Loop 1: O(n_uv_coords) Python dict lookups
for row in uv_rounded:
    key = tuple(row)
    ...

# Loop 2: O(3 * n_faces) Python dict lookups + tuple creation
for fi in range(n_faces):
    for j in range(3):
        e = tuple(sorted([...]))
        ...
```

For a 200K-face mesh, these combined to ~800K+ Python-level iterations.

### Fix

**UV dedup:** Replaced the manual dict-based dedup with `np.unique`:
```python
_, uv_idx = np.unique(uv_rounded, axis=0, return_inverse=True)
```

**Edge matching:** Replaced the per-edge dict lookup with vectorized sort +
comparison. Edge keys are computed in numpy, sorted, and adjacent duplicates
identified with a mask. Only matched edges (which need union) hit Python:

```python
all_edges.sort(axis=1)
edge_keys = all_edges[:, 0] * n_uv_verts + all_edges[:, 1]
order = np.argsort(edge_keys)
mask = edge_keys_sorted[:-1] == edge_keys_sorted[1:]
for i in np.nonzero(mask)[0]:
    union(face_ids_sorted[i], face_ids_sorted[i + 1])
```

---

## 5. Cached face-vertex array and edge lengths

**File:** `topology_features.py`
**Impact:** Minor — eliminates redundant allocation for large meshes

### Problem

`mesh.vertices[mesh.faces]` (a `(n_faces, 3, 3)` float64 array) was computed
twice per mesh — once for aspect ratios and once for edge length CV. Edge
lengths (`e01`, `e12`, `e20`) were also computed twice.

### Fix

Compute `fv` and edge lengths once, reuse for both feature groups.

---

## Note: GPU allocation

The SLURM config requests `--gpus=1` on `p5en-shared`. The current workload
is CPU-bound (numpy/scipy/sklearn), but the vector math could potentially be
GPU-accelerated with CuPy if it becomes the bottleneck after the S3 fix. Left
as-is for Yiwen to evaluate — depends on cluster pricing, partition
availability, and whether a GPU classifier is planned for stage 2.

---

## Combined impact estimate

| Metric | Before | After (all changes) |
|--------|--------|---------------------|
| Throughput (16 workers) | ~2.4 files/s | ~25–30 files/s |
| Crash recovery | Lose entire batch | Lose at most 10K files |
| UV feature time (200K faces) | ~0.5s | ~0.1–0.2s |
| Memory (futures) | 50K objects | ~64 objects |
| `/tmp` overflow risk | Yes | No (`$TMPDIR` respected) |

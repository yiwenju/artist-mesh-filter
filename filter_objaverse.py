"""filter_objaverse.py — Stage 1 inference on ObjaverseXL from S3.

Designed for SLURM job array processing. Each job reads a batch file
of S3 paths, downloads each mesh to /tmp, classifies, deletes.

Uses multiprocessing within each job for throughput.

Usage:
    # Single batch (for SLURM array jobs):
    python3.11 filter_objaverse.py --batch /path/to/batch_1.txt --output /path/to/results/batch_1.parquet

    # Standalone (process all batches sequentially):
    python3.11 filter_objaverse.py --batch-dir /path/to/batch_lists --output-dir /path/to/results
"""

import tempfile
import os
import sys
import argparse
import time
import numpy as np
import joblib
import boto3
from botocore.config import Config as BotoConfig
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['PYOPENGL_PLATFORM'] = 'egl'

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASSIFIER_PATH, FEATURE_COLUMNS, MIN_FACES, MAX_FACES

MESH_EXTENSIONS = {'.glb', '.gltf', '.obj', '.stl', '.fbx', '.ply'}
NUM_WORKERS = 16
FLUSH_EVERY = 5000

# Globals loaded once per worker process via initializer
_clf = None
_s3 = None


def _init_worker():
    global _clf, _s3
    _clf = joblib.load(str(CLASSIFIER_PATH))
    _s3 = boto3.client(
        's3',
        config=BotoConfig(
            max_pool_connections=10,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
        ),
    )


def _parse_s3_uri(uri):
    """'s3://bucket/key' -> (bucket, key)"""
    no_scheme = uri[5:]
    bucket, _, key = no_scheme.partition('/')
    return bucket, key


def _process_one(s3_path):
    """Download, classify, delete. Returns result dict or None."""
    from topology_features import load_meshes_from_file, compute_topology_features, features_to_vector

    ext = os.path.splitext(s3_path)[1].lower()
    if ext not in MESH_EXTENSIONS:
        return None

    uid = Path(s3_path).stem

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        bucket, key = _parse_s3_uri(s3_path)
        _s3.download_file(bucket, key, tmp_path)

        if os.path.getsize(tmp_path) < 1024:
            return None

        meshes = load_meshes_from_file(tmp_path)
        if not meshes:
            return None

        best_prob = 0.0
        best_nf = 0
        best_nv = 0

        for mesh in meshes:
            nf = len(mesh.faces)
            if nf < MIN_FACES or nf > MAX_FACES:
                continue
            try:
                obj_path = tmp_path if ext == '.obj' else None
                feats = compute_topology_features(mesh, obj_path=obj_path)
                vec = features_to_vector(feats).reshape(1, -1)
                prob = float(_clf.predict_proba(vec)[0, 1])
                if prob > best_prob:
                    best_prob = prob
                    best_nf = nf
                    best_nv = len(mesh.vertices)
            except Exception:
                continue

        if best_nf == 0:
            return None

        return {
            'uid': uid,
            's3_path': s3_path,
            'artist_prob': round(best_prob, 4),
            'face_count': best_nf,
            'vertex_count': best_nv,
        }
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def process_batch(batch_file, output_path, num_workers=NUM_WORKERS):
    """Process one batch file of S3 paths."""
    with open(batch_file) as f:
        s3_paths = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(s3_paths)} files from {batch_file}")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results = []
    processed = 0
    accepted = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_process_one, s3p): s3p for s3p in s3_paths}

        for future in as_completed(futures):
            processed += 1
            try:
                result = future.result(timeout=180)
                if result is not None:
                    results.append(result)
                    accepted += 1
            except Exception:
                pass

            if processed % FLUSH_EVERY == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                print(f"  {processed}/{len(s3_paths)} processed, "
                      f"{accepted} accepted, {rate:.1f} files/s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {processed} processed, {accepted} accepted in {elapsed:.0f}s "
          f"({processed / elapsed:.1f} files/s)")

    if results:
        table = pa.Table.from_pylist(results)
        pq.write_table(table, str(output_path))
        print(f"Saved {len(results)} results to {output_path}")

        probs = np.array([r['artist_prob'] for r in results])
        for t in [0.3, 0.5, 0.7, 0.85, 0.9]:
            n = int(np.sum(probs >= t))
            print(f"  P >= {t}: {n:>6d} ({n / len(probs) * 100:.1f}%)")
    else:
        print("No results to save.")

    return len(results)


def main():
    parser = argparse.ArgumentParser(description="ObjaverseXL Stage 1 inference")
    parser.add_argument('--batch', help="Single batch file to process")
    parser.add_argument('--output', help="Output parquet path for single batch")
    parser.add_argument('--batch-dir', help="Directory of batch files (standalone mode)")
    parser.add_argument('--output-dir', help="Output directory for standalone mode")
    parser.add_argument('--workers', type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    if args.batch:
        output = args.output or args.batch.replace('.txt', '.parquet')
        process_batch(args.batch, output, num_workers=args.workers)

    elif args.batch_dir:
        batch_dir = Path(args.batch_dir)
        output_dir = Path(args.output_dir or str(batch_dir) + '_results')
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_files = sorted(batch_dir.glob('batch_*.txt'))
        print(f"Found {len(batch_files)} batch files")

        total_accepted = 0
        for bf in batch_files:
            out = output_dir / bf.name.replace('.txt', '.parquet')
            if out.exists():
                print(f"Skipping {bf.name} (already done)")
                continue
            n = process_batch(str(bf), str(out), num_workers=args.workers)
            total_accepted += n

        print(f"\nAll batches done. Total accepted: {total_accepted}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

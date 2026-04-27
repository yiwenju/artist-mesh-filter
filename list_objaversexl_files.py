"""list_objaversexl_files.py — Enumerate all mesh files across ObjaverseXL S3 sources.

Generates batch files for SLURM array processing.

Sources:
  - s3://mod3d-west/objaverse-complete/glbs/       (800K GLBs, original Objaverse)
  - s3://mod3d-west/objaverse-xl/thingiverse/       (STL files)
  - s3://mod3d-west/objaverse-xl/smithsonian/        (museum scans)
  - s3://mod3d-west/objaverse-xl/github/repos_final/ (mixed formats)

Usage:
    python3.11 list_objaversexl_files.py --output-dir /weka/home-jurwen/artist_mesh_filter/batch_lists --batch-size 50000
"""

import argparse
import subprocess
import os
from pathlib import Path

MESH_EXTENSIONS = {'.glb', '.gltf', '.obj', '.stl', '.fbx', '.ply'}

S3_SOURCES = [
    's3://mod3d-west/objaverse-complete/glbs/',
    's3://mod3d-west/objaverse-xl/thingiverse/',
    's3://mod3d-west/objaverse-xl/smithsonian/',
    's3://mod3d-west/objaverse-xl/github/repos_final/',
]

MIN_FILE_BYTES = 1024


def list_s3_recursive(s3_prefix):
    """List all files under an S3 prefix recursively. Yields (s3_path, size_bytes)."""
    proc = subprocess.Popen(
        ['aws', 's3', 'ls', s3_prefix, '--recursive'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        size = int(parts[2])
        key = parts[3]
        yield key, size
    proc.wait()


def is_mesh_file(key):
    ext = os.path.splitext(key)[1].lower()
    return ext in MESH_EXTENSIONS


def extract_bucket_prefix(s3_url):
    """Split s3://bucket/prefix/ into (bucket, prefix)."""
    without_scheme = s3_url[5:]
    bucket = without_scheme.split('/')[0]
    prefix = '/'.join(without_scheme.split('/')[1:])
    return bucket, prefix


def main():
    parser = argparse.ArgumentParser(description="List ObjaverseXL mesh files and split into batches")
    parser.add_argument('--output-dir', default='/weka/home-jurwen/artist_mesh_filter/batch_lists')
    parser.add_argument('--batch-size', type=int, default=50000)
    parser.add_argument('--sources', nargs='+', default=None,
                        help="Override S3 sources (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = args.sources or S3_SOURCES
    all_files_path = output_dir / 'all_files.txt'

    print(f"Listing mesh files from {len(sources)} sources...")
    total = 0
    skipped_ext = 0
    skipped_size = 0

    with open(all_files_path, 'w') as out:
        for source in sources:
            print(f"\n  Source: {source}", flush=True)
            bucket, _ = extract_bucket_prefix(source)
            source_count = 0

            for key, size in list_s3_recursive(source):
                if not is_mesh_file(key):
                    skipped_ext += 1
                    continue
                if size < MIN_FILE_BYTES:
                    skipped_size += 1
                    continue

                s3_path = f"s3://{bucket}/{key}"
                out.write(f"{s3_path}\n")
                total += 1
                source_count += 1

                if source_count % 100000 == 0:
                    print(f"    {source_count:,} files...", flush=True)

            print(f"    Total from this source: {source_count:,}")

    print(f"\nTotal mesh files: {total:,}")
    print(f"Skipped (wrong extension): {skipped_ext:,}")
    print(f"Skipped (too small): {skipped_size:,}")
    print(f"Master list: {all_files_path}")

    # Split into batch files
    print(f"\nSplitting into batches of {args.batch_size:,}...")
    batch_idx = 0
    current_batch = []

    with open(all_files_path) as f:
        for line in f:
            current_batch.append(line.strip())
            if len(current_batch) >= args.batch_size:
                batch_idx += 1
                batch_path = output_dir / f"batch_{batch_idx}.txt"
                with open(batch_path, 'w') as bf:
                    bf.write('\n'.join(current_batch) + '\n')
                current_batch = []

    if current_batch:
        batch_idx += 1
        batch_path = output_dir / f"batch_{batch_idx}.txt"
        with open(batch_path, 'w') as bf:
            bf.write('\n'.join(current_batch) + '\n')

    print(f"Created {batch_idx} batch files in {output_dir}")
    print(f"Use: sbatch --array=1-{batch_idx} filter_objaverse.slurm")


if __name__ == '__main__':
    main()

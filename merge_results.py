"""merge_results.py — Merge per-batch parquet files into final output.

Usage:
    python3.11 merge_results.py --input-dir /path/to/results --output /path/to/final.parquet [--threshold 0.85]
"""

import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge batch result parquets")
    parser.add_argument('--input-dir', required=True, help="Directory with per-batch parquet files")
    parser.add_argument('--output', required=True, help="Output merged parquet path")
    parser.add_argument('--threshold', type=float, default=0.85, help="Artist probability threshold for stats")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    parquet_files = sorted(input_dir.glob('*.parquet'))
    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    if not parquet_files:
        print("No parquet files found.")
        return

    tables = []
    for pf in parquet_files:
        tables.append(pq.read_table(str(pf)))

    merged = pa.concat_tables(tables)
    pq.write_table(merged, args.output)

    probs = merged.column('artist_prob').to_numpy()
    print(f"\nTotal meshes scored: {len(probs):,}")
    print(f"\nP(artist) distribution:")
    for t in [0.3, 0.5, 0.7, 0.85, 0.9]:
        n = int(np.sum(probs >= t))
        print(f"  P >= {t}: {n:>10,} ({n / len(probs) * 100:.1f}%)")

    n_accepted = int(np.sum(probs >= args.threshold))
    print(f"\nAt threshold {args.threshold}: {n_accepted:,} meshes accepted "
          f"({n_accepted / len(probs) * 100:.1f}%)")
    print(f"Saved: {args.output}")

    # Also save a filtered version with only accepted meshes
    filtered_path = args.output.replace('.parquet', f'_accepted_{args.threshold}.parquet')
    mask = probs >= args.threshold
    filtered = merged.filter(mask)
    pq.write_table(filtered, filtered_path)
    print(f"Filtered ({n_accepted:,} rows): {filtered_path}")


if __name__ == '__main__':
    main()

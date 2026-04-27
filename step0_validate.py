"""step0_validate.py — Profile ObjaverseXL BEFORE training.

Samples meshes from S3 (or local cache), computes features for all of them
(NO face-count filtering — validation must cover the full distribution),
and exports a CSV for human labeling.

Can run in parallel with training (Track B in the pipeline).
"""

import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import (
    OBJAVERSE_XL_S3, OBJAVERSE_XL_LOCAL, STEP0_SAMPLE_PATH,
    STEP0_MESH_DIR, FEATURE_COLUMNS, OUTPUT_DIR, VALIDATION_SAMPLE_SIZE,
)
from topology_features import load_meshes_from_file, compute_topology_features


def _list_s3_files(s3_prefix, extensions=('.glb', '.obj', '.gltf')):
    """List mesh files from S3 bucket."""
    print(f"Listing files from {s3_prefix} ...")
    result = subprocess.run(
        ['aws', 's3', 'ls', s3_prefix, '--recursive'],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  aws s3 ls failed: {result.stderr[:300]}")
        return []

    files = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            key = parts[3]
            if any(key.lower().endswith(ext) for ext in extensions):
                files.append(key)
    print(f"  Found {len(files)} mesh files")
    return files


def _download_s3_file(s3_key, local_dir):
    """Download a single file from S3 to local directory."""
    s3_url = f"{OBJAVERSE_XL_S3}/{s3_key}" if not s3_key.startswith('s3://') else s3_key
    # Derive bucket path correctly
    if not s3_url.startswith('s3://'):
        s3_url = f"{OBJAVERSE_XL_S3}/{s3_key}"

    local_path = local_dir / Path(s3_key).name
    if local_path.exists():
        return local_path

    try:
        subprocess.run(
            ['aws', 's3', 'cp', s3_url, str(local_path)],
            capture_output=True, timeout=120,
        )
        if local_path.exists():
            return local_path
    except Exception:
        pass
    return None


def sample_and_profile(n_sample=None):
    if n_sample is None:
        n_sample = VALIDATION_SAMPLE_SIZE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STEP0_MESH_DIR.mkdir(parents=True, exist_ok=True)

    # Try local first, fall back to S3
    local_files = []
    if OBJAVERSE_XL_LOCAL.exists():
        print(f"Using local ObjaverseXL: {OBJAVERSE_XL_LOCAL}")
        local_files = sorted(
            list(OBJAVERSE_XL_LOCAL.rglob('*.glb')) +
            list(OBJAVERSE_XL_LOCAL.rglob('*.obj')) +
            list(OBJAVERSE_XL_LOCAL.rglob('*.gltf'))
        )
        print(f"  Found {len(local_files)} local files")

    if not local_files:
        print("No local files found. Sampling from S3...")
        s3_keys = _list_s3_files(OBJAVERSE_XL_S3)
        if not s3_keys:
            print("ERROR: No files found in S3 either. Check credentials and path.")
            return

        rng = np.random.default_rng(42)
        sample_keys = rng.choice(
            s3_keys,
            size=min(n_sample * 5, len(s3_keys)),
            replace=False,
        )

        print(f"Downloading {len(sample_keys)} files from S3...")
        local_files = []
        for key in tqdm(sample_keys, desc="Downloading"):
            local = _download_s3_file(key, STEP0_MESH_DIR)
            if local is not None:
                local_files.append(local)
            if len(local_files) >= n_sample * 3:
                break

    # Subsample if too many
    if len(local_files) > n_sample * 5:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(local_files), size=n_sample * 5, replace=False)
        local_files = [local_files[i] for i in sorted(indices)]

    # Profile meshes — NO face-count filtering for validation
    records = []
    for fpath in tqdm(local_files, desc="Profiling"):
        if len(records) >= n_sample:
            break

        meshes = load_meshes_from_file(str(fpath))
        for mesh_idx, mesh in enumerate(meshes):
            if len(records) >= n_sample:
                break
            try:
                obj_path = str(fpath) if str(fpath).endswith('.obj') else None
                feats = compute_topology_features(mesh, obj_path=obj_path)
                uid = f"{fpath.stem}_{mesh_idx}" if len(meshes) > 1 else fpath.stem
                records.append({
                    'path': str(fpath),
                    'uid': uid,
                    'mesh_index': mesh_idx,
                    'face_count': len(mesh.faces),
                    'vertex_count': len(mesh.vertices),
                    'human_label': '',
                    **feats,
                })
            except Exception:
                continue

    df = pd.DataFrame(records)
    df.to_csv(str(STEP0_SAMPLE_PATH), index=False)
    print(f"\nSaved {len(df)} samples to {STEP0_SAMPLE_PATH}")

    print(f"\nFeature distributions (excluding sentinels):")
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].values
        valid = vals[vals > -0.5]
        if len(valid) > 0:
            print(f"  {col:35s} "
                  f"mean={np.mean(valid):8.3f}  "
                  f"std={np.std(valid):8.3f}  "
                  f"[{np.min(valid):7.3f}, {np.max(valid):7.3f}]")

    print(f"\n{'=' * 60}")
    print(f"NEXT STEPS:")
    print(f"  1. Optionally render wireframe thumbnails:")
    print(f"     python3.11 stage2_render.py --sample")
    print(f"  2. Open {STEP0_SAMPLE_PATH} in a spreadsheet")
    print(f"  3. Fill 'human_label' column: artist / not_artist / unsure")
    print(f"  4. This becomes your test set for validation (Step 5)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    import sys
    n = VALIDATION_SAMPLE_SIZE
    for arg in sys.argv[1:]:
        if arg.startswith('--n='):
            n = int(arg.split('=')[1])
    sample_and_profile(n_sample=n)

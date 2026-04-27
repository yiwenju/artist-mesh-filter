"""build_dataset.py — Assemble training data.

Key improvements over original guide:
- Dual positive representation: each OBJ positive generates both an original-topology
  row AND a triangulated row (simulating GLB export) to bridge the format gap.
- Single mesh load per file (no double-loading).
- Multi-mesh GLB handling (each sub-mesh is a separate sample).
- ShapeNet as additional real-negative source.
- UUID-based negative IDs (no collisions).
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    POSITIVE_DIRS, OBJAVERSE_SCANNED_DIR, OBJAVERSE_SCANNED_SAMPLE,
    TRAINING_DATA_PATH, SYNTH_MESH_DIR, FEATURE_COLUMNS,
    UV_DROPOUT_RATE, NEG_PER_MESH, OUTPUT_DIR,
)
from topology_features import (
    load_meshes_from_file, compute_topology_features,
    features_to_vector, triangulate_mesh,
)
from negative_synthesis import (
    available_strategies, generate_negative, add_auto_uv,
    strip_uvs, make_negative_uid,
)


def _collect_positive_files():
    """Gather all mesh files from positive directories."""
    files = []
    for source, src_dir in POSITIVE_DIRS.items():
        if not src_dir.exists():
            print(f"  WARNING: {src_dir} does not exist, skipping")
            continue
        found = sorted(
            list(src_dir.rglob('*.obj')) +
            list(src_dir.rglob('*.glb')) +
            list(src_dir.rglob('*.gltf'))
        )
        for f in found:
            files.append((source, f))
        print(f"  {source}: {len(found)} files")
    return files


def _process_positive(mesh, fpath, source, is_obj, do_uv_dropout):
    """
    Extract features from a single positive mesh.
    Returns (feature_vector, metadata_dict) or None on failure.
    """
    obj_path = str(fpath) if is_obj else None

    mesh_for_feat = mesh
    if do_uv_dropout:
        mesh_for_feat = strip_uvs(mesh)
        obj_path = None

    feats = compute_topology_features(mesh_for_feat, obj_path=obj_path)
    return features_to_vector(feats), {
        'source': source,
        'strategy': 'original',
        'path': str(fpath),
    }


def _process_positive_triangulated(mesh, fpath, source, do_uv_dropout):
    """
    Extract features from a triangulated copy of an OBJ positive.
    Simulates what the mesh looks like when loaded as GLB at inference.
    """
    tri_mesh = triangulate_mesh(mesh)
    mesh_for_feat = tri_mesh
    if do_uv_dropout:
        mesh_for_feat = strip_uvs(tri_mesh)

    feats = compute_topology_features(mesh_for_feat, obj_path=None)
    return features_to_vector(feats), {
        'source': source,
        'strategy': 'original_triangulated',
        'path': str(fpath),
    }


def build():
    SYNTH_MESH_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_features = []
    all_labels = []
    all_meta = []

    strategies = available_strategies()
    print(f"Available degradation strategies: {strategies}")

    # ---- POSITIVES ----
    print("\n=== Collecting positive files ===")
    pos_files = _collect_positive_files()
    print(f"Total positive files: {len(pos_files)}")

    # Resumption: check for checkpoint
    checkpoint_path = OUTPUT_DIR / 'build_checkpoint.npz'
    start_idx = 0
    if checkpoint_path.exists():
        ckpt = np.load(str(checkpoint_path), allow_pickle=True)
        all_features = list(ckpt['features'])
        all_labels = list(ckpt['labels'])
        all_meta = list(ckpt['meta'])
        start_idx = int(ckpt['next_idx'])
        print(f"  Resuming from checkpoint: idx={start_idx}, "
              f"{len(all_features)} rows loaded")

    for file_idx, (source, fpath) in enumerate(tqdm(pos_files, desc="Positives")):
        if file_idx < start_idx:
            continue
        meshes = load_meshes_from_file(str(fpath))
        if not meshes:
            continue

        is_obj = str(fpath).endswith('.obj')

        for mesh_idx, mesh in enumerate(meshes):
            do_uv_dropout = np.random.random() < UV_DROPOUT_RATE

            # Original positive
            try:
                vec, meta = _process_positive(mesh, fpath, source, is_obj, do_uv_dropout)
                all_features.append(vec)
                all_labels.append(1)
                all_meta.append(meta)
            except Exception:
                continue

            # Dual representation: triangulated copy for OBJ files
            if is_obj:
                try:
                    vec_tri, meta_tri = _process_positive_triangulated(
                        mesh, fpath, source, do_uv_dropout
                    )
                    all_features.append(vec_tri)
                    all_labels.append(1)
                    all_meta.append(meta_tri)
                except Exception:
                    pass

            # Synthetic negatives from this positive
            # Skip degradation for non-OBJ (glTF/GLB) -- too slow on multi-mesh scenes
            if not is_obj:
                continue
            chosen = np.random.choice(
                strategies,
                size=min(NEG_PER_MESH, len(strategies)),
                replace=False,
            )
            for strat in chosen:
                neg = generate_negative(mesh, str(fpath), strat)
                if neg is None or len(neg.faces) < 10:
                    continue

                r = np.random.random()
                if r < 0.1:
                    pass  # 10% — no UV
                else:
                    try:
                        neg = add_auto_uv(neg)
                    except Exception:
                        pass

                neg_uid = make_negative_uid(fpath.stem, strat)
                neg_path = SYNTH_MESH_DIR / f"{neg_uid}.obj"
                try:
                    neg.export(str(neg_path))
                except Exception:
                    continue

                try:
                    neg_feats = compute_topology_features(neg, obj_path=None)
                    all_features.append(features_to_vector(neg_feats))
                    all_labels.append(0)
                    all_meta.append({
                        'source': source,
                        'strategy': strat,
                        'path': str(neg_path),
                    })
                except Exception:
                    continue

        # Checkpoint every 500 files
        if (file_idx + 1) % 500 == 0:
            np.savez(
                str(checkpoint_path),
                features=np.array(all_features, dtype=np.float32),
                labels=np.array(all_labels, dtype=np.int32),
                meta=np.array(all_meta, dtype=object),
                next_idx=file_idx + 1,
            )
            print(f"  Checkpoint saved at idx={file_idx + 1}, "
                  f"{len(all_features)} rows")

    # Remove checkpoint after positives complete
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ---- OBJAVERSE SCANNED NEGATIVES (via Objaverse++ labels) ----
    if OBJAVERSE_SCANNED_DIR.exists():
        print(f"\n=== Objaverse scanned negatives ({OBJAVERSE_SCANNED_DIR}) ===")
        scanned_files = sorted(
            list(OBJAVERSE_SCANNED_DIR.rglob('*.glb')) +
            list(OBJAVERSE_SCANNED_DIR.rglob('*.obj'))
        )
        print(f"  Found {len(scanned_files)} scanned files")

        if len(scanned_files) > OBJAVERSE_SCANNED_SAMPLE:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(scanned_files), size=OBJAVERSE_SCANNED_SAMPLE, replace=False)
            scanned_files = [scanned_files[i] for i in sorted(indices)]

        for fpath in tqdm(scanned_files, desc="Scanned negatives"):
            meshes = load_meshes_from_file(str(fpath))
            for mesh in meshes:
                try:
                    obj_path = str(fpath) if str(fpath).endswith('.obj') else None
                    feats = compute_topology_features(mesh, obj_path=obj_path)
                    all_features.append(features_to_vector(feats))
                    all_labels.append(0)
                    all_meta.append({
                        'source': 'objaverse_scanned',
                        'strategy': 'real_scan',
                        'path': str(fpath),
                    })
                except Exception:
                    continue
    else:
        print(f"\n  WARNING: Scanned negatives dir not found: {OBJAVERSE_SCANNED_DIR}")
        print("  Run download_scanned_negatives.py first.")

    # ---- Save as Parquet ----
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    columns = {col: X[:, i] for i, col in enumerate(FEATURE_COLUMNS)}
    columns['label'] = y
    columns['source'] = [m['source'] for m in all_meta]
    columns['strategy'] = [m['strategy'] for m in all_meta]
    columns['mesh_path'] = [m['path'] for m in all_meta]

    table = pa.table(columns)
    pq.write_table(table, str(TRAINING_DATA_PATH))

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"\nSaved: {TRAINING_DATA_PATH}")
    print(f"  Total: {len(y)} | Pos: {n_pos} | Neg: {n_neg} | Ratio: 1:{n_neg / max(n_pos, 1):.1f}")

    from collections import Counter
    strat_counts = Counter(m['strategy'] for m in all_meta)
    for s, c in sorted(strat_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:30s} {c:>6d}")


if __name__ == '__main__':
    build()

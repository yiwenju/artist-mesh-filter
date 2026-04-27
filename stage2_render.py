"""stage2_render.py — Wireframe rendering via pyrender.

Only triggered if Stage 1 validation shows precision < 85% at recall >= 80%.
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import trimesh
import pyrender
import math
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq

from config import CANDIDATES_PATH, RENDERS_DIR, STEP0_SAMPLE_PATH


def render_mesh_wireframe(mesh_path, output_dir, n_views=8, image_size=224):
    """Render mesh with wireframe overlay from multiple viewpoints."""
    try:
        loaded = trimesh.load(str(mesh_path), process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values()
                      if isinstance(g, trimesh.Trimesh) and len(g.faces) >= 10]
            if not meshes:
                return False
            tm = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            tm = loaded
        else:
            return False
    except Exception:
        return False

    if tm is None or len(tm.faces) < 10:
        return False

    center = tm.vertices.mean(axis=0)
    tm.vertices -= center
    scale = np.max(np.linalg.norm(tm.vertices, axis=1))
    if scale > 0:
        tm.vertices /= scale

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(image_size, image_size)

    for i in range(n_views):
        azimuth = 2 * math.pi * i / n_views
        elevation = math.radians(30 if i % 2 == 0 else -10)

        radius = 2.5
        cx = radius * math.cos(elevation) * math.cos(azimuth)
        cy = radius * math.cos(elevation) * math.sin(azimuth)
        cz = radius * math.sin(elevation)
        eye = np.array([cx, cy, cz])

        forward = -eye / (np.linalg.norm(eye) + 1e-10)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
        right /= (np.linalg.norm(right) + 1e-10)
        up = np.cross(right, forward)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = eye

        scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.3, 0.3, 0.3])

        solid_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.85, 0.85, 0.85, 0.95], metallicFactor=0.0, roughnessFactor=0.8,
        )
        scene.add(pyrender.Mesh.from_trimesh(tm, material=solid_mat))

        wire_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.15, 0.15, 0.15, 1.0], metallicFactor=0.0, roughnessFactor=1.0,
        )
        scene.add(pyrender.Mesh.from_trimesh(tm, material=wire_mat, wireframe=True))

        scene.add(pyrender.PerspectiveCamera(yfov=math.radians(45)), pose=cam_pose)
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=cam_pose)

        color, _ = renderer.render(scene)
        Image.fromarray(color).save(f"{output_dir}/view_{i:02d}.png")

    renderer.delete()
    return True


def render_sample():
    """Render wireframes for Step 0 validation sample."""
    df = pd.read_csv(str(STEP0_SAMPLE_PATH))
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    success, fail = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rendering sample"):
        uid = row['uid']
        out_dir = RENDERS_DIR / str(uid)
        if (out_dir / 'view_07.png').exists():
            success += 1
            continue
        ok = render_mesh_wireframe(row['path'], str(out_dir))
        if ok:
            success += 1
        else:
            fail += 1
    print(f"Done: {success} success, {fail} failed")


def render_candidates(max_meshes=None):
    """Render wireframes for all Stage 1 candidates."""
    df = pq.read_table(str(CANDIDATES_PATH)).to_pandas()
    if max_meshes:
        df = df.nlargest(max_meshes, 'artist_prob')
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    success, fail = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rendering candidates"):
        uid = row['uid']
        out_dir = RENDERS_DIR / str(uid)
        if (out_dir / 'view_07.png').exists():
            success += 1
            continue
        ok = render_mesh_wireframe(row['path'], str(out_dir))
        if ok:
            success += 1
        else:
            fail += 1
    print(f"Done: {success} success, {fail} failed")


if __name__ == '__main__':
    import sys
    if '--sample' in sys.argv:
        render_sample()
    elif '--candidates' in sys.argv:
        max_n = None
        if '--max' in sys.argv:
            idx = sys.argv.index('--max')
            max_n = int(sys.argv[idx + 1])
        render_candidates(max_meshes=max_n)
    else:
        print("Usage:")
        print("  python3.11 stage2_render.py --sample              # render Step 0 samples")
        print("  python3.11 stage2_render.py --candidates          # render all candidates")
        print("  python3.11 stage2_render.py --candidates --max N  # render top N candidates")

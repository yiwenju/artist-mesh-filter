"""negative_synthesis.py — 5 degradation strategies + UV augmentation.

Gracefully handles missing optional dependencies: if open3d or pymeshlab
are unavailable, only the strategies that need them are disabled.
"""

import uuid
import numpy as np
import trimesh

# Lazy-import optional heavy deps; track availability
_HAS_OPEN3D = False
_HAS_PYMESHLAB = False
_HAS_XATLAS = False

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    pass

try:
    import pymeshlab
    _HAS_PYMESHLAB = True
except ImportError:
    pass

try:
    import xatlas as _xatlas
    _HAS_XATLAS = True
except ImportError:
    pass


def _check_xatlas():
    if not _HAS_XATLAS:
        raise RuntimeError(
            "xatlas is REQUIRED for UV augmentation on negatives.\n"
            "Install: pip install xatlas\n"
            "Without it, all negatives silently lose auto-UV and the classifier\n"
            "learns a UV-presence shortcut instead of topology features."
        )


# ================================================================
# Strategy 1: Poisson reconstruction (mimics photogrammetry scans)
# ================================================================

MAX_FACES_FOR_DEGRADATION = 50_000
MAX_FACES_SKIP_DEGRADATION = 500_000


def poisson_degrade(mesh):
    """Sample point cloud from surface, Poisson reconstruct.

    Produces the characteristic irregular-triangle wireframe pattern of
    photogrammetry meshes without artificial noise.
    Scales n_points and depth to mesh complexity to avoid hanging on huge meshes.
    """
    if not _HAS_OPEN3D:
        return None

    nf = len(mesh.faces)
    if nf < 500:
        n_points, depth = 20000, 7
    elif nf < 10000:
        n_points, depth = 50000, 8
    elif nf < 50000:
        n_points, depth = 100000, 9
    else:
        n_points, depth = 200000, 9

    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    recon, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    recon.remove_vertices_by_mask(densities < np.quantile(densities, 0.01))

    return trimesh.Trimesh(
        vertices=np.asarray(recon.vertices),
        faces=np.asarray(recon.triangles),
        process=False,
    )


# ================================================================
# Strategy 2: Marching cubes via voxelization (mimics AI-generated)
# ================================================================

def marching_cubes_degrade(mesh, pitch=None):
    """Voxelize with flood-fill, extract via marching cubes."""
    if pitch is None:
        bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        pitch = bbox_diag / np.random.choice([48, 64, 96])
    voxelized = mesh.voxelized(pitch=pitch).fill()
    mc = voxelized.marching_cubes
    # marching_cubes returns voxel-grid coords; transform back to world space
    mc.apply_transform(voxelized.transform)
    return mc


# ================================================================
# Strategy 3: Isotropic remeshing (mimics auto-remeshed models)
# ================================================================

def remesh_degrade(filepath):
    """Isotropic explicit remeshing — destroys intentional edge flow."""
    if not _HAS_PYMESHLAB:
        return None

    factor = np.random.uniform(30.0, 80.0)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(filepath))
    bbox_diag = ms.current_mesh().bounding_box().diagonal()
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.AbsoluteValue(bbox_diag / factor),
        iterations=5,
    )
    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)


# ================================================================
# Strategy 4: QEM decimation (mimics auto-decimated models)
# ================================================================

def decimate_degrade(filepath):
    """Aggressive quadric edge collapse."""
    if not _HAS_PYMESHLAB:
        return None

    ratio = np.random.uniform(0.1, 0.4)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(filepath))
    target = max(int(ms.current_mesh().face_number() * ratio), 20)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target,
        preserveboundary=False,
        preservenormal=False,
        preservetopology=False,
        qualitythr=0.5,
    )
    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)


# ================================================================
# Strategy 5: Voxelization (mimics Minecraft/voxel-style exports)
# ================================================================

def voxel_degrade(mesh):
    """Low-resolution voxelization -> axis-aligned box mesh."""
    resolution = np.random.choice([16, 24, 32])
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    pitch = bbox_diag / resolution
    voxelized = mesh.voxelized(pitch=pitch).fill()
    return voxelized.as_boxes()


# ================================================================
# Strategy dispatch
# ================================================================

ALL_STRATEGIES = ['marching_cubes', 'voxel', 'decimate', 'remesh']


def available_strategies():
    """Return fast in-process strategies only.

    Poisson is excluded: too slow per mesh and causes OOM via subprocess forks.
    Photogrammetry-like negatives come from Objaverse scanned meshes instead.
    """
    avail = ['marching_cubes', 'voxel']
    if _HAS_PYMESHLAB:
        avail.extend(['decimate', 'remesh'])
    return avail


def _cap_mesh_size(mesh):
    """Decimate mesh if too large, to prevent degradation from hanging."""
    if len(mesh.faces) <= MAX_FACES_FOR_DEGRADATION:
        return mesh
    # pymeshlab is more reliable than trimesh for decimation
    if _HAS_PYMESHLAB:
        try:
            ms = pymeshlab.MeshSet()
            m = pymeshlab.Mesh(mesh.vertices, mesh.faces)
            ms.add_mesh(m)
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=MAX_FACES_FOR_DEGRADATION,
                qualitythr=0.5,
            )
            out = ms.current_mesh()
            return trimesh.Trimesh(
                vertices=out.vertex_matrix(),
                faces=out.face_matrix(),
                process=False,
            )
        except Exception:
            pass
    # Fallback: subsample faces
    rng = np.random.default_rng(0)
    indices = rng.choice(len(mesh.faces), size=MAX_FACES_FOR_DEGRADATION, replace=False)
    return trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[indices],
        process=True,
    )


def generate_negative(mesh, filepath, strategy):
    """
    Generate one degraded mesh from an artist mesh.
    All strategies run in-process (no subprocess overhead).
    Skips meshes that are too large (feature extraction still works on them).

    Returns trimesh.Trimesh or None on failure.
    """
    if len(mesh.faces) > MAX_FACES_SKIP_DEGRADATION:
        return None
    capped = _cap_mesh_size(mesh)
    try:
        if strategy == 'marching_cubes':
            return marching_cubes_degrade(capped)
        elif strategy == 'voxel':
            return voxel_degrade(capped)
        elif strategy == 'remesh':
            return remesh_degrade(filepath)
        elif strategy == 'decimate':
            return decimate_degrade(filepath)
    except Exception:
        return None
    return None


def make_negative_uid(stem, strategy):
    """Generate a unique ID for a synthetic negative (no collisions)."""
    return f"{stem}_{strategy}_{uuid.uuid4().hex[:8]}"


# ================================================================
# UV augmentation utilities
# ================================================================

def add_auto_uv(mesh):
    """
    Apply xatlas auto-UV to a mesh.
    Produces characteristic bad UV pattern: many small fragmented islands.
    """
    _check_xatlas()
    import xatlas

    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, faces)
    atlas.generate()
    vmapping, new_faces, new_uvs = atlas[0]

    result = trimesh.Trimesh(vertices=verts[vmapping], faces=new_faces, process=False)
    result.visual = trimesh.visual.TextureVisuals(
        uv=new_uvs,
        material=trimesh.visual.material.SimpleMaterial(),
    )
    return result


def strip_uvs(mesh):
    """Remove UV data from mesh. Used for positive UV-dropout augmentation."""
    stripped = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=False,
    )
    stripped.visual = trimesh.visual.ColorVisuals(mesh=stripped)
    return stripped

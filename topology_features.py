"""topology_features.py — 18-feature topology extractor.

Handles multi-mesh GLB files (each sub-mesh treated separately),
uses SENTINEL for format-dependent and missing-data columns,
and provides a triangulate helper for dual positive representation.
"""

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from config import SENTINEL, FEATURE_COLUMNS, SENTINEL_COLUMNS


# ================================================================
# Mesh loading — handles multi-mesh GLB/glTF
# ================================================================

def load_meshes_from_file(filepath):
    """
    Load meshes from any format. Returns list of trimesh.Trimesh.
    Multi-mesh GLB/glTF files yield one entry per sub-mesh.
    """
    try:
        loaded = trimesh.load(str(filepath), process=False)
    except Exception:
        return []

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            g for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.faces) >= 10
        ]
        return meshes if meshes else []
    elif isinstance(loaded, trimesh.Trimesh):
        return [loaded] if len(loaded.faces) >= 10 else []
    return []


def load_mesh_safe(filepath):
    """Load a single mesh (legacy compat). Returns first mesh or None."""
    meshes = load_meshes_from_file(filepath)
    return meshes[0] if meshes else None


# ================================================================
# Triangulation for dual positive representation
# ================================================================

def triangulate_mesh(mesh):
    """
    Force-triangulate a mesh (simulates GLB export behavior).
    Returns a new trimesh with all faces as triangles.
    """
    return trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=True,
    )


# ================================================================
# Quad detection (OBJ only — other formats triangulate on export)
# ================================================================

def detect_quads_from_obj(filepath):
    """Parse OBJ face lines to count quads before triangulation."""
    tris, quads = 0, 0
    try:
        with open(str(filepath), 'r') as f:
            for line in f:
                if line.startswith('f '):
                    n = len(line.strip().split()) - 1
                    if n == 3:
                        tris += 1
                    elif n >= 4:
                        quads += 1
    except Exception:
        return SENTINEL
    total = tris + quads
    return quads / total if total > 0 else 0.0


# ================================================================
# UV extraction
# ================================================================

def _extract_uv(mesh):
    """Safely extract UV array from trimesh. Returns None if absent."""
    if mesh.visual is None:
        return None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
        return mesh.visual.uv
    if hasattr(mesh.visual, 'to_texture'):
        try:
            tex = mesh.visual.to_texture()
            if hasattr(tex, 'uv') and tex.uv is not None and len(tex.uv) > 0:
                return tex.uv
        except Exception:
            pass
    return None


# ================================================================
# UV island computation via Union-Find
# ================================================================

def compute_uv_island_features(mesh):
    """
    Compute UV island statistics using Union-Find on UV-space face adjacency.
    Returns dict with 3 features. All SENTINEL if no UV data.
    """
    uv = _extract_uv(mesh)
    n_faces = len(mesh.faces)

    if uv is None or n_faces == 0:
        return {
            'uv_island_count': SENTINEL,
            'uv_faces_per_island': SENTINEL,
            'uv_island_size_entropy': SENTINEL,
        }

    uv_rounded = np.round(uv, decimals=6)
    unique_map = {}
    uv_idx = []
    for row in uv_rounded:
        key = tuple(row)
        if key not in unique_map:
            unique_map[key] = len(unique_map)
        uv_idx.append(unique_map[key])
    uv_idx = np.array(uv_idx)

    if len(uv_idx) == n_faces * 3:
        face_uv = uv_idx.reshape(n_faces, 3)
    elif len(uv_idx) == len(mesh.vertices):
        face_uv = uv_idx[mesh.faces]
    else:
        return {
            'uv_island_count': SENTINEL,
            'uv_faces_per_island': SENTINEL,
            'uv_island_size_entropy': SENTINEL,
        }

    parent = list(range(n_faces))
    uf_rank = [0] * n_faces

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if uf_rank[ra] < uf_rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if uf_rank[ra] == uf_rank[rb]:
            uf_rank[ra] += 1

    uv_edge_to_face = {}
    for fi in range(n_faces):
        for j in range(3):
            e = tuple(sorted([int(face_uv[fi, j]), int(face_uv[fi, (j + 1) % 3])]))
            if e in uv_edge_to_face:
                union(fi, uv_edge_to_face[e])
            else:
                uv_edge_to_face[e] = fi

    island_map = {}
    for fi in range(n_faces):
        root = find(fi)
        island_map[root] = island_map.get(root, 0) + 1

    island_sizes = np.array(list(island_map.values()), dtype=float)
    n_islands = len(island_sizes)

    entropy = 0.0
    if n_islands > 1:
        probs = island_sizes / island_sizes.sum()
        raw = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_ent = np.log2(n_islands)
        entropy = raw / max_ent if max_ent > 0 else 0.0

    return {
        'uv_island_count': float(n_islands),
        'uv_faces_per_island': float(n_faces / max(n_islands, 1)),
        'uv_island_size_entropy': entropy,
    }


# ================================================================
# Main feature computation — 18 features
# ================================================================

def compute_topology_features(mesh, obj_path=None):
    """
    Compute all 18 topology features for a single mesh.

    Args:
        mesh: trimesh.Trimesh loaded with process=False
        obj_path: str path to .obj file for quad detection.
                  Pass None for non-OBJ formats -> quad_ratio gets SENTINEL.
    Returns:
        dict mapping feature names to float values.
    """
    feat = {}
    n_faces = len(mesh.faces)
    n_verts = len(mesh.vertices)

    # 1. log_face_count
    feat['log_face_count'] = float(np.log1p(n_faces))

    # 2-4. Valence statistics
    edges = mesh.edges_unique
    if n_verts > 0 and len(edges) > 0:
        row = np.concatenate([edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0]])
        adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(n_verts, n_verts))
        valences = np.array(adj.sum(axis=1)).flatten().astype(int)
        valences = valences[valences > 0]
        counts = np.bincount(valences)
        probs = counts[counts > 0] / counts.sum()
        feat['valence_entropy'] = float(-np.sum(probs * np.log2(probs + 1e-10)))
        feat['pct_valence_4'] = float(np.mean(valences == 4))
        feat['valence_std'] = float(np.std(valences))
    else:
        feat['valence_entropy'] = SENTINEL
        feat['pct_valence_4'] = SENTINEL
        feat['valence_std'] = SENTINEL

    # 5. quad_ratio (SENTINEL for non-OBJ)
    if obj_path is not None and str(obj_path).endswith('.obj'):
        feat['quad_ratio'] = detect_quads_from_obj(str(obj_path))
    else:
        feat['quad_ratio'] = SENTINEL

    # 6-8. Face aspect ratio
    if n_faces > 0:
        fv = mesh.vertices[mesh.faces]
        e01 = np.linalg.norm(fv[:, 1] - fv[:, 0], axis=1)
        e12 = np.linalg.norm(fv[:, 2] - fv[:, 1], axis=1)
        e20 = np.linalg.norm(fv[:, 0] - fv[:, 2], axis=1)
        longest = np.maximum(e01, np.maximum(e12, e20))
        shortest = np.minimum(e01, np.minimum(e12, e20))
        ar = longest / (shortest + 1e-10)
        feat['aspect_ratio_mean'] = float(np.mean(ar))
        feat['aspect_ratio_p95'] = float(np.percentile(ar, 95))
        feat['pct_sliver_faces'] = float(np.mean(ar > 10))
    else:
        feat['aspect_ratio_mean'] = 0.0
        feat['aspect_ratio_p95'] = 0.0
        feat['pct_sliver_faces'] = 0.0

    # 9-10. Edge topology (numpy-vectorized)
    if n_faces > 0:
        faces_arr = mesh.faces
        he = np.concatenate([
            np.sort(np.stack([faces_arr[:, 0], faces_arr[:, 1]], axis=1), axis=1),
            np.sort(np.stack([faces_arr[:, 1], faces_arr[:, 2]], axis=1), axis=1),
            np.sort(np.stack([faces_arr[:, 2], faces_arr[:, 0]], axis=1), axis=1),
        ], axis=0)
        edge_ids = he[:, 0].astype(np.int64) * (n_verts + 1) + he[:, 1].astype(np.int64)
        _, edge_counts = np.unique(edge_ids, return_counts=True)
        total_e = len(edge_counts)
        feat['boundary_edge_ratio'] = float(np.sum(edge_counts == 1)) / max(total_e, 1)
        feat['non_manifold_edge_ratio'] = float(np.sum(edge_counts > 2)) / max(total_e, 1)
    else:
        feat['boundary_edge_ratio'] = 0.0
        feat['non_manifold_edge_ratio'] = 0.0

    # 11. Connected components (via scipy label, not mesh.split which is very slow)
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        face_adj = mesh.face_adjacency
        if len(face_adj) > 0:
            data = np.ones(len(face_adj) * 2)
            rows = np.concatenate([face_adj[:, 0], face_adj[:, 1]])
            cols = np.concatenate([face_adj[:, 1], face_adj[:, 0]])
            graph = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
            n_comp, _ = connected_components(graph, directed=False)
        else:
            n_comp = n_faces
    except Exception:
        n_comp = 1
    feat['n_components'] = float(n_comp)

    # 12. Edge length CV
    if n_faces > 0:
        fv = mesh.vertices[mesh.faces]
        all_len = np.concatenate([
            np.linalg.norm(fv[:, 1] - fv[:, 0], axis=1),
            np.linalg.norm(fv[:, 2] - fv[:, 1], axis=1),
            np.linalg.norm(fv[:, 0] - fv[:, 2], axis=1),
        ])
        feat['edge_length_cv'] = float(np.std(all_len) / (np.mean(all_len) + 1e-10))
    else:
        feat['edge_length_cv'] = 0.0

    # 13-14. Dihedral angle statistics
    try:
        angles = mesh.face_adjacency_angles
        if len(angles) > 0:
            feat['dihedral_angle_std'] = float(np.std(angles))
            feat['dihedral_angle_mean'] = float(np.mean(angles))
        else:
            feat['dihedral_angle_std'] = 0.0
            feat['dihedral_angle_mean'] = 0.0
    except Exception:
        feat['dihedral_angle_std'] = 0.0
        feat['dihedral_angle_mean'] = 0.0

    # 15. Euler characteristic normalized
    V = n_verts
    E_count = len(mesh.edges_unique) if len(edges) > 0 else 0
    F_count = n_faces
    chi = V - E_count + F_count
    feat['euler_char_normalized'] = float(chi / max(F_count, 1))

    # 16-18. UV island features
    feat.update(compute_uv_island_features(mesh))

    return feat


def features_to_vector(features):
    """Convert feature dict to numpy array in canonical column order."""
    return np.array([
        features.get(col, SENTINEL if col in SENTINEL_COLUMNS else 0.0)
        for col in FEATURE_COLUMNS
    ])

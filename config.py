"""config.py — All paths and constants."""

from pathlib import Path

PYTHON = "python3.11"

# ============================================================
# Dataset paths
# ============================================================

POSITIVE_DIRS = {
    'toys4k': Path('/weka/home-jurwen/toys4k/toys4k_obj_files/'),
    'sketchfab': Path('/weka/home-jurwen/sketchfab/models/'),
}

# Evermotion already has ~4K synthetic negatives from previous runs.
# Add back when ready:
# 'evermotion_am': Path('/weka/home-jurwen/evermotion/AM_obj/'),
# 'evermotion_ai': Path('/weka/home-jurwen/evermotion/AI_obj/'),

# Objaverse scanned meshes as real negative source (via Objaverse++ labels)
OBJAVERSE_SCANNED_DIR = Path('/weka/home-jurwen/objaverse_scanned/')
OBJAVERSE_SCANNED_SAMPLE = 5000

# ObjaverseXL S3 bucket for profiling / inference
OBJAVERSE_XL_S3 = 's3://mod3d-west/objaverse-xl'
OBJAVERSE_XL_LOCAL = Path('/weka/home-jurwen/objaverse_xl/')

# ============================================================
# Output paths
# ============================================================

OUTPUT_DIR = Path('/weka/home-jurwen/artist_mesh_filter/')

TRAINING_DATA_PATH = OUTPUT_DIR / 'training_data.parquet'
SYNTH_MESH_DIR = OUTPUT_DIR / 'synthetic_negatives/'
CLASSIFIER_PATH = OUTPUT_DIR / 'topology_classifier.pkl'
CANDIDATES_PATH = OUTPUT_DIR / 'objaverse_candidates.parquet'
STEP0_SAMPLE_PATH = OUTPUT_DIR / 'step0_sample.csv'
STEP0_MESH_DIR = OUTPUT_DIR / 'step0_meshes/'
RENDERS_DIR = OUTPUT_DIR / 'wireframe_renders/'
STAGE2_MODEL_PATH = OUTPUT_DIR / 'stage2_classifier.pth'

# ============================================================
# Feature extraction
# ============================================================

SENTINEL = -1.0

FEATURE_COLUMNS = [
    'log_face_count',
    'valence_entropy',
    'pct_valence_4',
    'valence_std',
    'quad_ratio',
    'aspect_ratio_mean',
    'aspect_ratio_p95',
    'pct_sliver_faces',
    'boundary_edge_ratio',
    'non_manifold_edge_ratio',
    'n_components',
    'edge_length_cv',
    'dihedral_angle_std',
    'dihedral_angle_mean',
    'euler_char_normalized',
    'uv_island_count',
    'uv_faces_per_island',
    'uv_island_size_entropy',
]

SENTINEL_COLUMNS = {
    'quad_ratio',
    'uv_island_count',
    'uv_faces_per_island',
    'uv_island_size_entropy',
    'valence_entropy',
    'pct_valence_4',
    'valence_std',
}

# ============================================================
# Training
# ============================================================

UV_DROPOUT_RATE = 0.7
NEG_PER_MESH = 3
NEG_AUTO_UV_RATE = 0.9
VALIDATION_SAMPLE_SIZE = 500

# ============================================================
# Inference
# ============================================================

MIN_FACES = 100
MAX_FACES = 200_000
MIN_FILE_BYTES = 1024
PARQUET_CHUNK_SIZE = 50_000

#!/bin/bash

#SBATCH --job-name=artist-mesh-filter
#SBATCH --account=mod3d
#SBATCH --partition=p5en-shared
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --time=6:00:00
#SBATCH --output=logs/filter.%A_%a.out
#SBATCH --error=logs/filter.%A_%a.err
#SBATCH --array=1-83

mkdir -p logs

echo "Job started at $(date)"
echo "Node: $HOSTNAME | Job: $SLURM_JOB_ID | Array: $SLURM_ARRAY_TASK_ID"

# Paths
SCRIPT_DIR="/admin/home-jurwen/labs/artist_mesh_filter"
BATCH_DIR="/weka/home-jurwen/artist_mesh_filter/batch_lists"
RESULTS_DIR="/weka/home-jurwen/artist_mesh_filter/batch_results"

BATCH_FILE="${BATCH_DIR}/batch_${SLURM_ARRAY_TASK_ID}.txt"
OUTPUT_FILE="${RESULTS_DIR}/batch_${SLURM_ARRAY_TASK_ID}.parquet"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$BATCH_FILE" ]; then
    echo "Error: Batch file $BATCH_FILE not found."
    exit 1
fi

# Skip if already completed
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output already exists: $OUTPUT_FILE. Skipping."
    exit 0
fi

echo "Processing batch file: $BATCH_FILE"
echo "Output: $OUTPUT_FILE"
echo "Files in batch: $(wc -l < "$BATCH_FILE")"

cd "$SCRIPT_DIR"

python3.11 filter_objaverse.py \
    --batch "$BATCH_FILE" \
    --output "$OUTPUT_FILE" \
    --workers 16 \
    || { echo "Error: Processing failed"; exit 1; }

echo "Job completed at $(date)"

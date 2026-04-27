"""download_scanned_negatives.py — Download scanned meshes from Objaverse for negatives.

Iterates through S3 shards, matches GLB filenames against Objaverse++
scanned UIDs, randomly collects up to N meshes.
"""

import json
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm

S3_BASE = "s3://mod3d-west/objaverse-complete/glbs"
SCANNED_UIDS_PATH = "/weka/home-jurwen/objaverse_plusplus/scanned_uids.json"
OUTPUT_DIR = Path("/weka/home-jurwen/objaverse_scanned/")
TARGET_COUNT = 5000
SEED = 42


def list_shard(shard_name):
    """List all GLB filenames in a shard."""
    result = subprocess.run(
        ["aws", "s3", "ls", f"{S3_BASE}/{shard_name}/"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return []
    uids = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[3].endswith(".glb"):
            uid = parts[3].replace(".glb", "")
            uids.append(uid)
    return uids


def download_glb(shard, uid, dest_dir):
    """Download a single GLB from S3."""
    s3_path = f"{S3_BASE}/{shard}/{uid}.glb"
    local_path = dest_dir / f"{uid}.glb"
    if local_path.exists():
        return local_path
    try:
        subprocess.run(
            ["aws", "s3", "cp", s3_path, str(local_path)],
            capture_output=True, timeout=120,
        )
        if local_path.exists():
            return local_path
    except Exception:
        pass
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SCANNED_UIDS_PATH) as f:
        scanned_uids = set(json.load(f))
    print(f"Scanned UIDs loaded: {len(scanned_uids)}")

    # List all shards
    result = subprocess.run(
        ["aws", "s3", "ls", f"{S3_BASE}/"],
        capture_output=True, text=True, timeout=60,
    )
    shards = []
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split()
        if parts and parts[-1].startswith("PRE "):
            shards.append(parts[-1].rstrip("/"))
        elif parts and parts[-1].endswith("/"):
            shards.append(parts[-1].rstrip("/"))
    # Fix parsing: PRE shows as separate token
    shards = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("PRE "):
            shard = line[4:].rstrip("/")
            shards.append(shard)
    print(f"Found {len(shards)} shards")

    # Pass 1: iterate shards, collect all scanned UIDs with their shard
    print("\nPass 1: Scanning shards for scanned UIDs...")
    scanned_in_s3 = []

    for shard in tqdm(shards, desc="Scanning shards"):
        shard_uids = list_shard(shard)
        for uid in shard_uids:
            if uid in scanned_uids:
                scanned_in_s3.append((shard, uid))

    print(f"\nFound {len(scanned_in_s3)} scanned GLBs in S3")

    # Random sample
    rng = np.random.default_rng(SEED)
    if len(scanned_in_s3) > TARGET_COUNT:
        indices = rng.choice(len(scanned_in_s3), size=TARGET_COUNT, replace=False)
        selected = [scanned_in_s3[i] for i in sorted(indices)]
    else:
        selected = scanned_in_s3
    print(f"Selected {len(selected)} for download")

    # Pass 2: download
    success, fail = 0, 0
    for shard, uid in tqdm(selected, desc="Downloading"):
        path = download_glb(shard, uid, OUTPUT_DIR)
        if path:
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} downloaded, {fail} failed")
    print(f"Saved to: {OUTPUT_DIR}")

    # Save manifest
    manifest = [{"uid": uid, "shard": shard, "path": str(OUTPUT_DIR / f"{uid}.glb")}
                for shard, uid in selected]
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()

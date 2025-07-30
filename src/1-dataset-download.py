import os
import requests
import tarfile
import glob
import shutil
import pandas as pd

# URLs for the dataset parts
DATASET_DIR = "crag_dataset"
PARTS = [
    "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v4.tar.bz2.part1",
    "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v4.tar.bz2.part2",
    "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v4.tar.bz2.part3",
    "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_3_dev_v4.tar.bz2.part4",
]

os.makedirs(DATASET_DIR, exist_ok=True)

# Step 1: Download the parts
for i, url in enumerate(PARTS, 1):
    part_path = os.path.join(DATASET_DIR, f"crag_task_3_dev_v4.tar.bz2.part{i}")
    if not os.path.exists(part_path):
        print(f"[+] Downloading: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(part_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    else:
        print(f"[✓] Already downloaded: {part_path}")

# Step 2: Merge into .tar.bz2
merged_path = os.path.join(DATASET_DIR, "crag_task_3_dev_v4.tar.bz2")
if not os.path.exists(merged_path):
    print(f"[+] Merging into {merged_path}")
    with open(merged_path, "wb") as wfd:
        for i in range(1, 5):
            part_path = os.path.join(DATASET_DIR, f"crag_task_3_dev_v4.tar.bz2.part{i}")
            with open(part_path, "rb") as f:
                shutil.copyfileobj(f, wfd)
else:
    print(f"[✓] Already merged: {merged_path}")

# Step 3: Extract the merged .tar.bz2 file
extracted_dir = os.path.join(DATASET_DIR, "extracted")
if not os.path.exists(extracted_dir):
    print(f"[+] Extracting to: {extracted_dir}")
    with tarfile.open(merged_path, "r:bz2") as tar:
        tar.extractall(extracted_dir)
else:
    print(f"[✓] Already extracted: {extracted_dir}")

# Step 4: Convert all .json/.jsonl files to .parquet
parquet_dir = os.path.join(DATASET_DIR, "parquet")
os.makedirs(parquet_dir, exist_ok=True)
for file in glob.glob(f"{extracted_dir}/**/*.json*", recursive=True):
    filename = os.path.basename(file)
    parquet_path = os.path.join(parquet_dir, filename.replace(".jsonl", "").replace(".json", "") + ".parquet")

    print(f"[+] Converting {filename} to Parquet")
    try:
        if file.endswith(".jsonl"):
            df = pd.read_json(file, lines=True)
        else:
            df = pd.read_json(file)

        try:
            df.to_parquet(parquet_path, index=False)
        except Exception as e:
            print(f"⚠️ Conversion failed due to: {e} — converting all columns to strings")
            df = df.astype(str)
            df.to_parquet(parquet_path, index=False)

        os.remove(file)
    except Exception as e:
        print(f"❌ Failed to process {file}: {e}")



import pyarrow.parquet as pq
import pyarrow as pa
import glob
import os

final_parquet_path = os.path.join(DATASET_DIR, "crag_full_dataset.parquet")

print(f"[+] Merging all .parquet files incrementally")

parquet_files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))

writer = None
for parquet_file in parquet_files:
    try:
        table = pq.read_table(parquet_file)
        if writer is None:
            writer = pq.ParquetWriter(final_parquet_path, table.schema)
        elif table.schema != writer.schema:
            print(f"⚠️ Skipping {parquet_file}: schema mismatch")
            continue
        writer.write_table(table)
        print(f"[✓] Added: {os.path.basename(parquet_file)}")
    except Exception as e:
        print(f"⚠️ Failed to append {parquet_file}: {e}")

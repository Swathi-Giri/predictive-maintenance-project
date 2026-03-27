"""
download_data.py — Download the NASA C-MAPSS Turbofan Engine Degradation Dataset.

HOW TO USE:
    python download_data.py

This will download and extract the dataset into data/raw/
"""

import urllib.request
import zipfile
import os
import sys


DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
RAW_DIR = os.path.join("data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "cmapss.zip")


def download_dataset():
    """Download and extract the NASA C-MAPSS dataset."""

    os.makedirs(RAW_DIR, exist_ok=True)

    # Check if already downloaded
    if os.path.exists(os.path.join(RAW_DIR, "train_FD001.txt")):
        print("✅ Dataset already exists in data/raw/")
        return

    print(f"📥 Downloading NASA C-MAPSS dataset...")
    print(f"   URL: {DATASET_URL}")
    print(f"   This is about 3 MB — should take a few seconds.\n")

    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("✅ Download complete!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("   1. Go to: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print("   2. Download the dataset")
        print("   3. Extract the files into the data/raw/ folder")
        print("   4. You should have: train_FD001.txt, test_FD001.txt, RUL_FD001.txt")
        sys.exit(1)

    # Extract
    print("📦 Extracting files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    print("✅ Extraction complete!")

    # Clean up zip
    os.remove(ZIP_PATH)

    # Verify files exist
    expected_files = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    found = []
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f in expected_files:
                # Move to raw dir if in subfolder
                src = os.path.join(root, f)
                dst = os.path.join(RAW_DIR, f)
                if src != dst:
                    os.rename(src, dst)
                found.append(f)

    print(f"\n📁 Files in data/raw/:")
    for f in sorted(os.listdir(RAW_DIR)):
        size = os.path.getsize(os.path.join(RAW_DIR, f))
        print(f"   {f} ({size:,} bytes)")

    missing = set(expected_files) - set(found)
    if missing:
        print(f"\n⚠️  Missing files: {missing}")
        print("   Check if they are in a subfolder inside data/raw/")
    else:
        print("\n✅ All required files present! You're ready to go.")
        print("   Next step: python src/train.py")


if __name__ == "__main__":
    download_dataset()

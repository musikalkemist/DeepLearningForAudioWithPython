"""
GTZAN Dataset Setup Utility
----------------------------------
This script automates the acquisition and organization of the GTZAN dataset 
for the TSOAI (The Sound of AI) courses.

Functionality:
1. Downloads the latest dataset version directly from Hugging Face.
2. Extracts the compressed archive into the local project structure.
3. Sanitizes the dataset by removing hidden MacOS metadata artifacts
   (._ files) to prevent processing errors in librosa.
4. Performs automatic cleanup of temporary download files.
"""

import os
import requests
import tarfile
from pathlib import Path

URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
# Anchors everything to the directory of the current file
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "GTZAN_dataset"  # Root folder
DATASET_FILE = BASE_DIR / "genres.tar.gz" # Zip file

def clean_macos_artifacts(folder_path):
    """Recursively removes hidden MacOS metadata files (starting with ._)"""
    print(f"\n3️⃣  Cleaning MacOS metadata artifacts in {folder_path} ...")
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith("._"):
                full_path = os.path.join(root, file)
                try:
                    os.remove(full_path)
                    count += 1
                except OSError as e:
                    print(f"   Failed to remove {file}: {e}")
    
    if count > 0:
        print(f"   🧹 Removed {count} hidden '._' ghost files.")
    else:
        print("   ✨ No artifacts found. Directory is clean.")


def download_direct():    
    # This calculates the full path based on where you run the script from
    absolute_path = os.path.abspath(DATASET_PATH)
    
    print()
    print("="*80)
    print("Starting automated dataset downloader script for GTZAN for TSOAI courses!")
    print("="*80)

    # 1. Download
    print(f"\n1️⃣  Downloading dataset from Hugging Face.…\n⏳ Please wait until it's finished, it could take a while :)")
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status() # Check for download errors
        with open(DATASET_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk: f.write(chunk)
        print("☑️  Completed dataset download as: ", DATASET_FILE)
    except Exception as e:
        print(f"❌ Error downloading file: {e}")
        return

    # 2. Extract
    print(f"\n2️⃣  Extracting files into destination folder …\n   {absolute_path}")
    try:
        with tarfile.open(DATASET_FILE, "r:gz") as tar:
            tar.extractall(path=DATASET_PATH)
    except Exception as e:
        print(f"❌ Error extracting file: {e}")
        return
            
    print(f"\n3️⃣  Finishisng cleaning process …")
    # 3. Cleanup MacOS artifacts
    clean_macos_artifacts(DATASET_PATH)

    # 4. Remove zip file
    if os.path.exists(DATASET_FILE):
        os.remove(DATASET_FILE)

    print("\n","="*80)
    print("✅ Dataset was downloaded sucessfully.\n\n")

if __name__ == "__main__":
    download_direct()
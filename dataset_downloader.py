"""
GTZAN Dataset Setup Utility
----------------------------------
This script automates the acquisition and organization of the GTZAN dataset 
for the TSOAI (The Sound of AI) courses. 

Functionality:
1. Downloads the latest dataset version from Huggingface.

"""

import os
import requests
import tarfile

def download_direct():
    url = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
    target_folder = "./GTZAN_dataset" # Root folder
    filename = "genres.tar.gz"
    # This calculates the full path based on where you run the script from
    absolute_path = os.path.abspath(target_folder)

    print()
    print("="*80)
    print("Starting automated dataset downloader script for GTZAN for TSOAI courses!")
    print("="*80)

    # 1. Download
    print(f"\n1️⃣  Downloading dataset from Hugging Face.…\n⏳ Please wait until it's finished, it could take a while :)")
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk: f.write(chunk)
    print("☑️  Completed dataset download as: ", filename)
            
    # 2. Extract
    print(f"\n2️⃣  Extracting files into destination folder …\n   {absolute_path}")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=target_folder)
            
    print("="*80)
    print("✅ Success! Dataset is ready for training.\n\n")
    
    # Cleanup zip file
    os.remove(filename)

if __name__ == "__main__":
    download_direct()
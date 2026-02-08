# How to Download the GTZAN Dataset

This project requires the **GTZAN Genre Collection** dataset. You can download it automatically using the provided Python script (recommended for TSOAI courses) or manually.

## Option 1: Automatic Download (Recommended)

This method uses the `dataset_downloader.py` script to fetch the data directly from Hugging Face and organize the folders according to the project requirements. **No API keys are required.**

### 1. Install Requirements
This script uses standard Python libraries, so usually no extra installation is needed. If `requests` is missing:
```bash
pip install requests
```

### 2. Run the Script
Run the script to download, extract, and clean up the compressed files automatically:
```bash
python dataset_downloader.py
```

### What this script does
1. **Downloads** the dataset (`genres.tar.gz`) directly from the Marsyas/GTZAN Hugging Face repository.
2. **Creates Directory:** It creates a `./GTZAN_dataset` folder in your project root.
3. **Extracts:** Unzips the contents directly into that folder.
4. **Cleanup:** Automatically removes the `.tar.gz` file after extraction to save space.

---
## Option 2: Manual Download
If the script fails or you prefer to manage files yourself:

### 1. Download
1. Visit the Hugging Face file repository: [https://huggingface.co/datasets/marsyas/gtzan/tree/main/data](https://www.google.com/search?q=https://huggingface.co/datasets/marsyas/gtzan/tree/main/data)
2. Download the file named `genres.tar.gz`.

### 2. Extract and Organize
1. Create a folder named `GTZAN_dataset` in your project root.
2. Extract the contents of `genres.tar.gz` into that folder.

### 3. Verify Directory Structure
Your project directory should look exactly like this for the code to run correctly:
```text
DLforAudio/
├── GTZAN_dataset/                 
│   └── genres/         
│       ├── blues/            <-- For every genre, there is a folder.
│       │   └── blues.00000.wav 
│       └── ...
└── dataset_downloader.py
```

### 4. Path Configuration
Ensure your preprocessing scripts point to the correct folder location:
```python
# preprocess.py (or your config file)
DATASET_PATH = "GTZAN_dataset/genres/"
```
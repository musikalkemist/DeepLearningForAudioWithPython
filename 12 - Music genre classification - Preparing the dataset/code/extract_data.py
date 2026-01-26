# Based on the original implementation by Valerio.
# Includes improvements over the video tutorial: automatic dataset downloading, robust error handling, and an enhanced CLI.
# NOTE: The deprecated original source code is available on the "legacy" branch.

import json
import os
import math
import librosa
import subprocess
import time
import warnings

ROOT = "../../"
DATASET_PATH = f"{ROOT}GTZAN_dataset/genres" # Change it to your local path if needed
JSON_PATH = f"{ROOT}GTZAN_dataset/data_10.json"
DOWNLOADER_PATH = f"{ROOT}dataset_downloader.py"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# Ignore specific warnings from librosa/audioread
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    corrupt_count = 0 # Memory of corrupted files

    print("")
    print("="*80)
    print("🔄 STARTING DATA EXTRACTION PROCESS")
    print("="*80)
    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = os.path.split(dirpath)[-1]
            data["mapping"].append(semantic_label)
            print("\n- Processing: {}".format(semantic_label))

            process_count = 0 # Memory of successful files
            # process all audio files in genre sub-dir
            for f in filenames:
                # Ignore hidden files like .DS_Store (MacOS)
                if f.startswith('.'):
                    continue

                # load audio file
                file_path = os.path.join(dirpath, f)
                
                # Solution for corrupted files
                try:
                    # signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print(f"\r{' ' * 100}\r", end="") 
                    print(f"  ❗️ Skipping corrupted file {file_path}\n")
                    corrupt_count +=1
                    continue

                # process all segments of audio file
                for d in range(num_segments):
                    process_count +=1

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        if (process_count <= 10) or (process_count % 10 == 0):
                            updated_count = process_count
                        if (d % 2 == 0) or (d + 1 == 10):
                            print("\r  [{:04d} chunks ready]  {}, segment:{:02d}  ".format(round(updated_count,2), f, d+1), end="", flush=True)

    # Ensure the directory exists (prevents "FileNotFoundError")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # save MFCCs to json file
    try:
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)
        print(f"\n\n☑️  Successfully saved data to {json_path}\n")
        
    except OSError as e:
        # Catch file system errors (permissions, disk full, bad path)
        print(f"❌ FILE ERROR: Could not open or write to {json_path}\nDetails: {e}")
    except TypeError as e:
        # Catch JSON errors (e.g., trying to save a NumPy array directly)
        print(f"❌ DATA ERROR: The dictionary contains data that JSON cannot save.\nDetails: {e}")
    except Exception as e:
        # Catch anything else unexpected
        print(f"❌ UNEXPECTED ERROR: {e}")


    if corrupt_count:
        print(f"⚠️  ALERT: {corrupt_count} files were corrupted and have been skipped.")
    print("="*80)
    print("✅ Success! Dataset is ready for training.\n\n")
        
if __name__ == "__main__":
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n\n⚠️  ALERT: Dataset not found at {DATASET_PATH}")
        print("Attempting to run automated downloader… [Press Ctrl+C to cancel]")
        for n in range(5, 0, -1):
            print(f"\rStarting in {n} seconds...", end="")
            time.sleep(1)
        else:
            print("\rStarting download...       \n")

        # Check if the downloader script exists before trying to run it
        if os.path.exists(DOWNLOADER_PATH):
            subprocess.run(["python", DOWNLOADER_PATH], check=True)
        else:
            print(f"❌ ERROR: Could not find downloader script at {DOWNLOADER_PATH}")
            exit(1) # Stop execution if data is missing and downloader isn't found

    # Proceed with MFCC extraction
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
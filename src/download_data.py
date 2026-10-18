"""
Step 1: download the Amharic part of the AfriSenti dataset.

Source: AfriSenti-SemEval 2023 shared task (Muhammad et al., 2023), licence CC BY 4.0.
https://github.com/afrisenti-semeval/afrisent-semeval-2023/tree/main/data/amh

Run:  python src/download_data.py
"""
import os
import urllib.request

from utils import DATA_DIR

BASE_URL = "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data/amh/"

os.makedirs(DATA_DIR, exist_ok=True)
for split in ["train", "dev", "test"]:
    url = BASE_URL + f"{split}.tsv"
    out_path = os.path.join(DATA_DIR, f"{split}.tsv")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, out_path)
print("Done. Files saved in", DATA_DIR)

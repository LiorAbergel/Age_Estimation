"""
Download the HHD-Age dataset from Kaggle.

Usage:
    python download_dataset.py

Requires the 'kagglehub' package:
    pip install kagglehub

On first run, you will be prompted to authenticate with your Kaggle credentials.
"""

import os
import re
import shutil

import pandas as pd

KAGGLE_DATASET = "liorabergel/hhd-age"
KAGGLE_CSV_NAME = "AgeAnnotations_and_Split.csv"
TARGET_CSV_NAME = "NewAgeSplit.csv"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _prepare_csv(data_dir):
    """Rename the Kaggle CSV and add the WriterNumber column if missing."""
    target_csv = os.path.join(data_dir, TARGET_CSV_NAME)
    source_csv = os.path.join(data_dir, KAGGLE_CSV_NAME)

    # If target already exists with WriterNumber, nothing to do
    if os.path.isfile(target_csv):
        df = pd.read_csv(target_csv, nrows=1)
        if "WriterNumber" in df.columns:
            return

    # Determine which CSV to read
    if os.path.isfile(source_csv):
        df = pd.read_csv(source_csv)
    elif os.path.isfile(target_csv):
        df = pd.read_csv(target_csv)
    else:
        return

    # Add WriterNumber derived from filename (e.g. w11_F_2_form1.tif -> 11)
    if "WriterNumber" not in df.columns:
        df["WriterNumber"] = df["File"].apply(
            lambda f: int(re.match(r"w(\d+)", f).group(1))
        )

    df.to_csv(target_csv, index=False)

    # Remove the original if it was renamed
    if os.path.isfile(source_csv) and source_csv != target_csv:
        os.remove(source_csv)


def ensure_dataset(data_dir=None):
    """Download the HHD-Age dataset if not already present.

    Args:
        data_dir: Target directory for the dataset. Defaults to ./data
                  relative to this script's location.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    # Resolve relative paths against the repo root (this script's directory)
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)

    csv_path = os.path.join(data_dir, TARGET_CSV_NAME)
    train_dir = os.path.join(data_dir, "train")

    if os.path.isfile(csv_path) and os.path.isdir(train_dir):
        # Ensure WriterNumber column exists even if data was already downloaded
        _prepare_csv(data_dir)
        return data_dir

    print(f"Dataset not found at {data_dir}. Downloading from Kaggle...")

    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "kagglehub is required to download the dataset. "
            "Install it with: pip install kagglehub"
        )

    downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Downloaded to: {downloaded_path}")

    os.makedirs(data_dir, exist_ok=True)

    # Copy contents from the downloaded cache to the target data directory
    for item in os.listdir(downloaded_path):
        src = os.path.join(downloaded_path, item)
        dst = os.path.join(data_dir, item)
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Post-process: rename CSV and add WriterNumber column
    _prepare_csv(data_dir)

    print(f"Dataset ready at {data_dir}")
    return data_dir


if __name__ == "__main__":
    ensure_dataset()

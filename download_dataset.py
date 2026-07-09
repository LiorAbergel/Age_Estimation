"""
Download the HHD-Age dataset from Zenodo.

The dataset is hosted as a single archive on the Zenodo record published by
Rabaev et al. (record 14996257). It is fully open, so no account or API
credentials are required.

Usage:
    python download_dataset.py

Uses only the Python standard library plus pandas.
"""

import os
import re
import shutil
import tempfile
import urllib.request
import zipfile

import pandas as pd

# Zenodo record 14996257 -> single file AgeAnnotations_and_Split.zip (~109 MB).
ZENODO_RECORD_ID = "14996257"
ZENODO_ZIP_NAME = "AgeAnnotations_and_Split.zip"
ZENODO_ZIP_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{ZENODO_ZIP_NAME}?download=1"
)

SOURCE_CSV_NAME = "AgeAnnotations_and_Split.csv"
TARGET_CSV_NAME = "NewAgeSplit.csv"
SPLITS = ("train", "val", "test")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _prepare_csv(data_dir):
    """Normalise the labels CSV to the format the experiments expect.

    Produces ``NewAgeSplit.csv`` with columns ``File, AgeGroup, Set, Age,
    WriterNumber`` where ``Set`` is lowercase (``train``/``val``/``test``) and
    ``WriterNumber`` is derived from the filename (e.g. ``w11_F_2_form1.tif`` ->
    ``11``). Renames the Zenodo-provided ``AgeAnnotations_and_Split.csv`` if that
    is what is present.
    """
    target_csv = os.path.join(data_dir, TARGET_CSV_NAME)
    source_csv = os.path.join(data_dir, SOURCE_CSV_NAME)

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

    # Normalise the split column to lowercase (the CNN/ViT pipelines match
    # ``Set == "train"/"val"/"test"`` case-sensitively).
    if "Set" in df.columns:
        df["Set"] = df["Set"].astype(str).str.lower()

    # Add WriterNumber derived from filename (e.g. w11_F_2_form1.tif -> 11)
    if "WriterNumber" not in df.columns:
        df["WriterNumber"] = df["File"].apply(
            lambda f: int(re.match(r"w(\d+)", f).group(1))
        )

    df.to_csv(target_csv, index=False)

    # Remove the original if it was renamed
    if os.path.isfile(source_csv) and source_csv != target_csv:
        os.remove(source_csv)


def _report_progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(f"\r  downloading {ZENODO_ZIP_NAME}: {pct:3d}% ({mb:.1f}/{total_mb:.1f} MB)",
          end="", flush=True)


def _find_source_root(extract_dir):
    """Locate the directory inside the extracted archive that holds the CSV.

    The Zenodo archive may extract either directly or nested inside a top-level
    folder, so search recursively for ``AgeAnnotations_and_Split.csv``.
    """
    for root, _dirs, files in os.walk(extract_dir):
        if SOURCE_CSV_NAME in files or TARGET_CSV_NAME in files:
            return root
    raise RuntimeError(
        f"Could not find {SOURCE_CSV_NAME} inside the downloaded archive."
    )


def _copy_split_images(src_root, data_dir):
    """Copy the train/val/test image folders, skipping non-image junk files."""
    for split in SPLITS:
        src_split = os.path.join(src_root, split)
        if not os.path.isdir(src_split):
            raise RuntimeError(
                f"Expected split folder '{split}' not found in the archive at {src_root}."
            )
        dst_split = os.path.join(data_dir, split)
        os.makedirs(dst_split, exist_ok=True)
        for name in os.listdir(src_split):
            if not name.lower().endswith(".tif"):
                continue  # skip desktop.ini and other non-image artefacts
            dst = os.path.join(dst_split, name)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(src_split, name), dst)


def ensure_dataset(data_dir=None):
    """Download the HHD-Age dataset from Zenodo if not already present.

    Args:
        data_dir: Target directory for the dataset. Defaults to ./data
                  relative to this script's location.

    Produces the canonical layout expected by every experiment::

        <data_dir>/NewAgeSplit.csv
        <data_dir>/train/  <data_dir>/val/  <data_dir>/test/
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    # Resolve relative paths against the repo root (this script's directory)
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)

    csv_path = os.path.join(data_dir, TARGET_CSV_NAME)

    def _split_populated(split):
        d = os.path.join(data_dir, split)
        return os.path.isdir(d) and any(f.lower().endswith(".tif") for f in os.listdir(d))

    have_splits = all(_split_populated(s) for s in SPLITS)

    if os.path.isfile(csv_path) and have_splits:
        # Ensure WriterNumber column exists even if data was already downloaded
        _prepare_csv(data_dir)
        return data_dir

    print(f"Dataset not found at {data_dir}. Downloading from Zenodo...")
    os.makedirs(data_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, ZENODO_ZIP_NAME)
        urllib.request.urlretrieve(ZENODO_ZIP_URL, zip_path, _report_progress)
        print()  # newline after the progress bar

        extract_dir = os.path.join(tmp, "extracted")
        print("  extracting archive...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        src_root = _find_source_root(extract_dir)

        # Copy and normalise the labels CSV.
        src_csv = os.path.join(src_root, SOURCE_CSV_NAME)
        if not os.path.isfile(src_csv):
            src_csv = os.path.join(src_root, TARGET_CSV_NAME)
        shutil.copy2(src_csv, os.path.join(data_dir, SOURCE_CSV_NAME))

        # Copy the image splits (excluding junk files such as desktop.ini).
        _copy_split_images(src_root, data_dir)

    # Post-process: rename CSV, lowercase Set, add WriterNumber column.
    _prepare_csv(data_dir)

    print(f"Dataset ready at {data_dir}")
    return data_dir


if __name__ == "__main__":
    ensure_dataset()

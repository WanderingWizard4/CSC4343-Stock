"""
setup_data.py
=============
Extracts all us_ohlc1m_YYYY-MM.tar files downloaded from OneDrive
and organizes them into the folder structure expected by StockDataLoader.

Expected output structure:
    extracted_files/
        2000/
            2000-01/
                AAPL.csv
                MSFT.csv
                ...
            2000-02/
                ...
        2001/
            ...

USAGE
-----
1. Download all .tar files from David's OneDrive into a single folder
   (e.g. a folder called "downloads" next to this script)
2. Set TAR_DIR below to that folder path
3. Set OUTPUT_DIR to where you want the extracted files
4. Run:  python setup_data.py
"""

import os
import tarfile
import re
from pathlib import Path

# =============================================================
# CONFIGURE THESE TWO PATHS BEFORE RUNNING
# =============================================================

# Folder where you downloaded all the .tar files from OneDrive
TAR_DIR = "./downloads"

# Where extracted files should go — must match DATA_PATH in volatility_lstm.py
# and stock_data_loader.py (i.e. the "extracted_files" folder)
OUTPUT_DIR = "../OHLC 1 minute data/extracted_files"

# =============================================================


def extract_all(tar_dir: str, output_dir: str) -> None:
    tar_dir    = Path(tar_dir)
    output_dir = Path(output_dir)

    # Find all .tar files matching the naming pattern
    tar_files = sorted(tar_dir.glob("us_ohlc1m_*.tar"))

    if not tar_files:
        print(f"No .tar files found in '{tar_dir}'. "
              f"Make sure you downloaded them from OneDrive into that folder.")
        return

    print(f"Found {len(tar_files)} .tar files to extract.\n")

    success = 0
    skipped = 0

    for tar_path in tar_files:
        # Parse YYYY-MM from filename, e.g. us_ohlc1m_2003-08.tar → 2003, 08
        match = re.search(r"(\d{4})-(\d{2})", tar_path.name)
        if not match:
            print(f"  SKIP — couldn't parse year/month from: {tar_path.name}")
            skipped += 1
            continue

        year  = match.group(1)
        month = match.group(2)

        # Build destination path: extracted_files/YYYY/YYYY-MM/
        dest = output_dir / year / f"{year}-{month}"
        dest.mkdir(parents=True, exist_ok=True)

        # Skip if already extracted (folder is non-empty)
        if any(dest.iterdir()):
            print(f"  SKIP (already extracted) — {tar_path.name} → {dest}")
            skipped += 1
            continue

        print(f"  Extracting {tar_path.name} → {dest} ...", end=" ", flush=True)
        try:
            with tarfile.open(tar_path, "r") as tf:
                # Extract all members, stripping any leading path components
                # so CSVs land directly in dest/ rather than a subdirectory
                for member in tf.getmembers():
                    member.name = Path(member.name).name  # strip subdirs
                    if member.name:  # skip empty names (directory entries)
                        tf.extract(member, path=dest)
            print("done")
            success += 1
        except Exception as e:
            print(f"ERROR — {e}")
            skipped += 1

    print(f"\n{'='*50}")
    print(f"  Extracted : {success} files")
    print(f"  Skipped   : {skipped} files")
    print(f"  Output    : {output_dir.resolve()}")
    print(f"{'='*50}")
    print("\nYou're ready to run volatility_lstm.py!")


if __name__ == "__main__":
    extract_all(TAR_DIR, OUTPUT_DIR)

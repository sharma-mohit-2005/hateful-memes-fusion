"""Download the Hateful Memes dataset from the Hugging Face mirror into data/hateful_memes/.

Usage:  python src/download_data.py
Result: data/hateful_memes/{train,dev_seen,dev_unseen,test_seen,test_unseen}.jsonl + img/*.png  (~3.5 GB)

If the mirror is ever unavailable, download the dataset manually (e.g. Kaggle) and arrange it in the
same layout: the jsonl files must have fields id, img ("img/xxxxx.png"), text, label.

How it works (overview):
  1. Read command-line options (which Hugging Face repo, how many parallel download threads).
  2. Create the target folder data/hateful_memes/.
  3. Use huggingface_hub.snapshot_download to fetch only the files we need (jsonl + images).
  4. Count the downloaded images and list the jsonl files as a quick sanity check.
"""
# argparse: standard library module for reading command-line options like --workers 8.
import argparse

# snapshot_download: downloads (a filtered part of) a whole Hugging Face repository to a local folder,
# with resume support and parallel downloads.
from huggingface_hub import snapshot_download

# RAW_DIR = data/hateful_memes, defined once in common.py so all scripts agree on the location.
from common import RAW_DIR


# ---------------------------------------------------------------------------------------------
# main
# Purpose: download the Hateful Memes dataset (about 12k memes: image + caption text + label).
# Inputs:  command-line flags --repo (Hugging Face dataset id) and --workers (parallel threads).
# Output:  files written to data/hateful_memes/; prints a short summary.
# Viva tip: Q: What is in the dataset? A: Facebook AI's Hateful Memes: each meme = image + overlaid
#           text + binary label (1 = hateful). It includes "benign confounders" so that neither
#           text alone nor image alone is enough - the model must combine both (multimodal).
# Viva tip: Q: Why a script instead of manual download? A: Reproducibility - anyone can rebuild the
#           exact same data folder with one command.
# ---------------------------------------------------------------------------------------------
def main():
    # Create the command-line parser.
    ap = argparse.ArgumentParser()
    # Which Hugging Face dataset repo to download from (a public mirror of the original dataset).
    ap.add_argument("--repo", default="neuralcatcher/hateful_memes")
    # Number of parallel download threads; more threads = faster for many small image files.
    ap.add_argument("--workers", type=int, default=16)
    # Parse the actual arguments given on the command line (or use the defaults).
    args = ap.parse_args()

    # Make sure the destination folder exists (create parents too, no error if it already exists).
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Tell the user what is happening, since the download is large and slow.
    print(f"Downloading {args.repo} -> {RAW_DIR} (12k images, takes 5-20 min)")
    # Download the repository snapshot.
    snapshot_download(
        # The dataset's id on Hugging Face Hub.
        repo_id=args.repo,
        # It is a dataset repo (not a model repo), so the right URL prefix is used.
        repo_type="dataset",
        # Save the files directly into data/hateful_memes/ (not into the hidden HF cache).
        local_dir=RAW_DIR,
        # Only download the label files (jsonl), the images, and the licence/readme - skip anything else.
        allow_patterns=["*.jsonl", "img/*", "LICENSE.txt", "README.md"],
        # Use the requested number of parallel download threads.
        max_workers=args.workers,
    )
    # Sanity check 1: count how many PNG images arrived in img/.
    n_img = len(list((RAW_DIR / "img").glob("*.png")))
    # Sanity check 2: list the names of all jsonl annotation files (train, dev_seen, test_seen ...).
    jsonls = sorted(p.name for p in RAW_DIR.glob("*.jsonl"))
    # Print the summary so the user can confirm the download is complete.
    print(f"Done. {n_img} images, jsonl files: {jsonls}")


# Run main() only when this file is executed directly (python src/download_data.py),
# not when it is imported by another script.
if __name__ == "__main__":
    main()

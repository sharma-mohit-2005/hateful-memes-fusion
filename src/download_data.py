"""Download the Hateful Memes dataset from Hugging Face mirrors into data/hateful_memes/.

Usage:  python src/download_data.py          (safe to re-run: files already downloaded are skipped)
Result: data/hateful_memes/{train,dev_seen,dev_unseen,test_seen,test_unseen}.jsonl + img/*.png  (~4.5 GB)

If the mirrors are ever unavailable, download the dataset manually (e.g. Kaggle) and arrange it in the
same layout: the jsonl files must have fields id, img ("img/xxxxx.png"), text, label.

How it works (overview):
  1. Read command-line options (which Hugging Face repos, how many parallel download threads).
  2. Create the target folder data/hateful_memes/.
  3. Download the original label files (jsonl) and the images from the main mirror.
  4. The main mirror is missing ~2.5k of the 12,140 images, so every image referenced by the jsonl
     files that is still missing is fetched from a second mirror that has all of them.
     We take ONLY the images from the second mirror: its train.jsonl contains ~4.4k extra memes that are
     not part of the original dataset, and using them would make our results incomparable to the paper.
  5. Count the images and report any that are still missing (prepare_splits.py skips those memes).
"""
# argparse: standard library module for reading command-line options like --workers 8.
import argparse
# json: read the jsonl label files to find out which images we need.
import json
# ThreadPoolExecutor: download many small image files in parallel threads.
from concurrent.futures import ThreadPoolExecutor

# snapshot_download: downloads (a filtered part of) a whole Hugging Face repository to a local folder,
# with resume support and parallel downloads. hf_hub_download: downloads one single file.
from huggingface_hub import hf_hub_download, snapshot_download
# tqdm: progress bar for the second (fill-in) download.
from tqdm import tqdm

# RAW_DIR = data/hateful_memes, defined once in common.py so all scripts agree on the location.
from common import RAW_DIR


# ---------------------------------------------------------------------------------------------
# needed_images
# Purpose: list every image path referenced by the downloaded jsonl files.
# Output:  sorted list of relative paths like "img/01235.png".
# Viva tip: Q: Why not just download every image of the second mirror? A: It also contains images of
#           ~4.4k extra memes we do not use; fetching only what the original jsonl files reference
#           saves time and disk space.
# ---------------------------------------------------------------------------------------------
def needed_images():
    # A set removes duplicates (some memes appear in more than one split, e.g. dev_seen / dev_unseen).
    imgs = set()
    # Every jsonl file in the data folder (train, dev_seen, dev_unseen, test_seen, test_unseen).
    for path in RAW_DIR.glob("*.jsonl"):
        # One JSON object per non-empty line; its "img" field is the relative image path.
        imgs |= {json.loads(line)["img"] for line in open(path, encoding="utf-8") if line.strip()}
    return sorted(imgs)


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
    # Main mirror: original label files (jsonl) and most of the images.
    ap.add_argument("--repo", default="neuralcatcher/hateful_memes")
    # Second mirror: has all 12,140 images; used only to fill in images missing from the main mirror.
    ap.add_argument("--image_repo", default="limjiayi/hateful_memes_expanded")
    # Number of parallel download threads; more threads = faster for many small image files.
    ap.add_argument("--workers", type=int, default=16)
    # Parse the actual arguments given on the command line (or use the defaults).
    args = ap.parse_args()

    # Make sure the destination folder exists (create parents too, no error if it already exists).
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Tell the user what is happening, since the download is large and slow.
    print(f"Step 1/2: downloading labels + images from {args.repo} -> {RAW_DIR} (takes 5-20 min)")
    # Download the repository snapshot (files that already exist locally are skipped).
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

    # Images referenced by the label files that did not arrive from the main mirror.
    missing = [p for p in needed_images() if not (RAW_DIR / p).exists()]
    print(f"Step 2/2: fetching {len(missing)} missing images from {args.image_repo}")

    # Download one image into the same data folder (same relative path "img/xxxxx.png").
    def fetch(rel_path):
        try:
            hf_hub_download(args.image_repo, rel_path, repo_type="dataset", local_dir=RAW_DIR)
        except Exception as e:  # one failed image must not stop the whole download
            print(f"  could not download {rel_path}: {e}")

    # Run the downloads in parallel threads; list(...) waits for all of them and drives the progress bar.
    with ThreadPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(fetch, missing), total=len(missing), desc="missing images"))

    # Sanity check: every referenced image should now exist.
    needed = needed_images()
    still_missing = [p for p in needed if not (RAW_DIR / p).exists()]
    # List the names of all jsonl annotation files (train, dev_seen, test_seen ...).
    jsonls = sorted(p.name for p in RAW_DIR.glob("*.jsonl"))
    # Print the summary so the user can confirm the download is complete.
    print(f"Done. {len(needed) - len(still_missing)}/{len(needed)} images present, jsonl files: {jsonls}")
    if still_missing:
        print(f"WARNING: {len(still_missing)} images still missing - re-run this script; "
              f"prepare_splits.py will skip those memes.")


# Run main() only when this file is executed directly (python src/download_data.py),
# not when it is imported by another script.
if __name__ == "__main__":
    main()

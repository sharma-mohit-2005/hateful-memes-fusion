"""Modality reliance test (research angle C) for every CLIP fusion-head run (C_* runs).

At test time we corrupt one modality and measure how much AUROC drops:
  original        untouched
  shuffle_text    each meme gets a random other meme's text
  shuffle_image   each meme gets a random other meme's image
  mean_text       text embedding replaced by the train-set mean (text "removed")
  mean_image      image embedding replaced by the train-set mean
A model that truly fuses both modalities should drop a lot in BOTH shuffle conditions.

Usage: python src/analysis/modality_ablation.py [--split test]
Output: results/modality_ablation_<split>.csv/.md

How it works (overview):
- We do NOT retrain anything. We load each already-trained fusion head (one per run and seed) and feed it
  the same frozen CLIP features as before, except that one modality is broken on purpose.
- Logic of the test: if the model's AUROC does NOT drop when the text is shuffled, then the model was not
  using the text at all (it ignores it). A big drop means the model depends on that modality.
- Shuffling keeps the distribution of the features realistic (they are real embeddings, just from the wrong
  meme), so it removes only the meme-specific information. Mean-replacement gives every meme the same
  "average" embedding, which is like deleting the modality with a neutral value.
- Shuffling is random, so we repeat it several times (--repeats) and average, to get a stable number.
- Results are reported as mean ± std over the training seeds of each run.
"""
# argparse: reads command-line options (--split, --repeats).
import argparse
# json: reads each run's saved config.json (model hyper-parameters).
import json
# sys: lets us edit the import search path.
import sys
# Path: convenient, OS-independent file paths.
from pathlib import Path

# Put the "src" folder (parent of "analysis") at the front of the import path so that our own modules
# (common, fusion_heads, train_head) can be imported. Must come before those imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# numpy: mean and std of the scores.
import numpy as np
# pandas: builds the final results table.
import pandas as pd
# torch: tensors, loading model weights, random permutations.
import torch
# AUROC: area under the ROC curve; threshold-free, 0.5 = random guessing, 1.0 = perfect ranking.
from sklearn.metrics import roc_auc_score

# Project helpers: result folders, device selection (GPU/CPU), feature loader, list of seeds per run.
from common import RESULTS_DIR, RUNS_DIR, get_device, load_features, run_seeds
# FusionHead: the small trainable network on top of frozen CLIP image+text embeddings.
from fusion_heads import FusionHead
# predict_probs: runs the model on (image, text) feature tensors and returns p(hateful) per meme.
from train_head import predict_probs

# The five test conditions, in the order they appear as columns in the output table.
CONDITIONS = ["original", "shuffle_text", "shuffle_image", "mean_text", "mean_image"]


# ---------------------------------------------------------------------------------------------------------
# main
# Purpose : for every CLIP fusion-head run (folders named C_*), load each trained seed, evaluate AUROC under
#           the five conditions, and write a table of mean ± std over seeds.
# Inputs  : command-line --split (which split to test on) and --repeats (number of random shuffles per seed)
# Outputs : results/modality_ablation_<split>.csv and .md, plus the table printed to the terminal.
# Viva tip: Q: How do you know if your fusion model really uses the text? A: Shuffle the text across memes at
#           test time. If AUROC stays the same, the text was ignored; if it falls toward 0.5, it was used.
# Viva tip: Q: Why use the TRAIN mean for mean-replacement, not the test mean? A: The test set must not leak
#           into the procedure; the train mean is a fixed, "neutral" input the model has seen around.
# Viva tip: Q: Why average several shuffles? A: One random permutation could be lucky or unlucky; the average
#           of 5 gives a more stable estimate.
# ---------------------------------------------------------------------------------------------------------
def main():
    # Create the argument parser.
    ap = argparse.ArgumentParser()
    # Which split to evaluate on (default "test").
    ap.add_argument("--split", default="test")
    # How many random shuffles to average for each seed (default 5).
    ap.add_argument("--repeats", type=int, default=5, help="random shuffles averaged per seed")
    # Parse the command line.
    args = ap.parse_args()
    # Pick the GPU if available, otherwise the CPU.
    device = get_device()

    # One output row per run will be collected here.
    rows = []
    # Loop over all run folders whose name starts with "C_" (the frozen-CLIP fusion-head runs), sorted.
    for run_path in sorted(RUNS_DIR.glob("C_*")):
        # Run name, e.g. "C_clip_l14_cross".
        run = run_path.name
        # Dictionary: condition -> list of AUROC values (one per seed).
        scores = {c: [] for c in CONDITIONS}
        # Evaluate every seed that this run was trained with.
        for seed in run_seeds(run):
            # Folder of this seed, e.g. .../C_clip_l14_cross/seed0.
            d = run_path / f"seed{seed}"
            # Skip seeds whose trained weights are missing (e.g. unfinished training).
            if not (d / "model.pt").exists():
                continue
            # Load the training config so we can rebuild the exact same model architecture.
            cfg = json.loads((d / "config.json").read_text())
            # Rebuild the fusion head with the saved settings: fusion type (e.g. concat / cross-attention),
            # input embedding size, projection size, cross-attention size, hidden size and dropout.
            # .to(device) moves the model to GPU/CPU.
            model = FusionHead(cfg["fusion"], cfg["in_dim"], proj_dim=cfg["proj_dim"], cross_dim=cfg["cross_dim"],
                               hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
            # Load the trained weights into the model. map_location makes it work even if the weights were
            # saved on a GPU and we are now on a CPU.
            model.load_state_dict(torch.load(d / "model.pt", map_location=device))
            # Frozen CLIP features of the evaluation split (same feature set the model was trained on).
            f = load_features(cfg["features"], args.split)
            # Features of the training split, only used to compute the mean embeddings.
            tr = load_features(cfg["features"], "train")
            # Image embeddings of the eval split as a tensor, shape (N, D).
            img = torch.tensor(f["img"], device=device)
            # Text embeddings of the eval split as a tensor, shape (N, D).
            txt = torch.tensor(f["txt"], device=device)
            # True labels (0 benign / 1 hateful), shape (N,).
            y = f["label"]
            # Train-set mean image embedding, shape (D,), copied (broadcast) to every row -> shape (N, D).
            # expand_as does not copy memory; it just views the same vector N times.
            mean_img = torch.tensor(tr["img"].mean(0), device=device).expand_as(img)
            # Same for the text: every meme gets the average training text embedding.
            mean_txt = torch.tensor(tr["txt"].mean(0), device=device).expand_as(txt)
            # Small helper: given image features i and text features t, predict probabilities with the model
            # and return the AUROC against the true labels y. The labels y are never changed.
            auc = lambda i, t: roc_auc_score(y, predict_probs(model, i, t))

            # Baseline: both modalities untouched.
            scores["original"].append(auc(img, txt))
            # Text "removed": every meme gets the mean text, the real image is kept.
            scores["mean_text"].append(auc(img, mean_txt))
            # Image "removed": every meme gets the mean image, the real text is kept.
            scores["mean_image"].append(auc(mean_img, txt))
            # Random number generator on CPU seeded with the run seed, so the shuffles are reproducible.
            g = torch.Generator(device="cpu").manual_seed(seed)
            # Lists of AUROCs for the repeated text-shuffles (st) and image-shuffles (si).
            st, si = [], []
            # Repeat the random shuffle several times.
            for _ in range(args.repeats):
                # A random permutation of 0..N-1 (made on CPU with our generator), then moved to the device.
                perm = torch.randperm(len(y), generator=g).to(device)
                # Shuffle text: meme k keeps its image and label but gets the text of meme perm[k].
                st.append(auc(img, txt[perm]))
                # Shuffle image: meme k keeps its text and label but gets the image of meme perm[k].
                si.append(auc(img[perm], txt))
            # Store the average over the repeats as this seed's score.
            scores["shuffle_text"].append(np.mean(st))
            scores["shuffle_image"].append(np.mean(si))
        # Only add a table row if at least one seed was evaluated.
        if scores["original"]:
            # Start the row with the run name.
            row = {"run": run}
            # For each condition write "mean ± std" over seeds (std shows how much seeds vary).
            for c in CONDITIONS:
                row[c] = f"{np.mean(scores[c]):.3f} ± {np.std(scores[c]):.3f}"
            # Add the row to the table.
            rows.append(row)

    # Build the results DataFrame (one row per run, one column per condition).
    res = pd.DataFrame(rows)
    # Base output path without extension: results/modality_ablation_<split>.
    out = RESULTS_DIR / f"modality_ablation_{args.split}"
    # Save as CSV.
    res.to_csv(out.with_suffix(".csv"), index=False)
    # Save as a Markdown table for the report.
    out.with_suffix(".md").write_text(res.to_markdown(index=False), encoding="utf-8")
    # Print the table to the terminal.
    print(res.to_string(index=False))


# Run main() only when this file is executed directly as a script.
if __name__ == "__main__":
    main()

"""Most confident mistakes of a model (for the qualitative error analysis in the report).

Usage: python src/analysis/error_examples.py --run C_clip_l14_cross --split test --k 6
Output: results/errors/<run>_<split>.csv (all errors, sorted) and <run>_<split>.png (top FP / FN)
Tip: label ~40 errors by hand in the csv (e.g. sarcasm / needs world knowledge / text-in-image / label noise)
and report the category counts.

How it works (overview):
- Load the predicted probabilities p(hateful) of one run for every seed and average them per meme.
  Averaging over seeds gives a more stable prediction (a small ensemble) and the errors we show are not
  caused by one unlucky seed.
- A meme is an error if the prediction at threshold 0.5 disagrees with the label.
    false positive (FP): label 0 (benign) but predicted hateful (prob >= 0.5)
    false negative (FN): label 1 (hateful) but predicted benign  (prob < 0.5)
- "Confidence" of a mistake = |prob - label|. Near 1 means the model was very sure and very wrong.
  We sort errors by this value, save them all to a CSV (with an empty "category" column to fill by hand),
  and plot the top k FPs and top k FNs.
"""
# argparse: reads command-line options (--run, --split, --k).
import argparse
# sys: lets us change the import search path.
import sys
# Path: OS-independent file paths.
from pathlib import Path

# Add the "src" folder to the import path so that "common" can be imported (must be before that import).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# matplotlib: plotting library.
import matplotlib

# "Agg" backend: save plots to files only, no window (works on servers without a display).
matplotlib.use("Agg")
# pyplot plotting API (imported after the backend is chosen).
import matplotlib.pyplot as plt
# pandas: tables of predictions and errors.
import pandas as pd
# PIL Image: opens meme images for the figure.
from PIL import Image

# Project helpers: original image folder, results folder, prediction/split loaders, seed list of a run.
from common import RAW_DIR, RESULTS_DIR, load_preds, load_split, run_seeds


# ---------------------------------------------------------------------------------------------------------
# main
# Purpose : find the most confident false positives and false negatives of one run on one split.
# Inputs  : command-line --run (run name, required), --split (default test), --k (examples per row, default 6)
# Outputs : results/errors/<run>_<split>.csv (all errors, most confident first) and
#           results/errors/<run>_<split>.png (top k FP in row 1, top k FN in row 2)
# Viva tip: Q: What is the difference between a false positive and a false negative here? A: FP = a benign
#           meme wrongly flagged as hateful; FN = a hateful meme the model missed. For moderation, FNs let
#           hate through, FPs censor harmless content.
# Viva tip: Q: Why average probabilities over seeds? A: It reduces randomness from training (like a small
#           ensemble), so the errors shown are systematic mistakes of the model type, not seed noise.
# ---------------------------------------------------------------------------------------------------------
def main():
    # Create the argument parser.
    ap = argparse.ArgumentParser()
    # Name of the run folder to analyse (e.g. C_clip_l14_cross); must be given.
    ap.add_argument("--run", required=True)
    # Which split to analyse (default "test").
    ap.add_argument("--split", default="test")
    # How many example images to show per error type.
    ap.add_argument("--k", type=int, default=6)
    # Parse the command line.
    args = ap.parse_args()

    # Load the prediction table (id, prob, label) of every seed and stack them into one long table.
    probs = pd.concat([load_preds(args.run, s, args.split) for s in run_seeds(args.run)])
    # Group by meme id: average the probability over seeds; the label is the same in every seed, so take
    # the first one. Result: one row per meme.
    probs = probs.groupby("id", as_index=False).agg(prob=("prob", "mean"), label=("label", "first"))
    # Join the averaged probability onto the split table (which has img path, text and label) by meme id.
    df = load_split(args.split).merge(probs[["id", "prob"]], on="id")
    # Distance between prediction and truth: near 0 = confident and right, near 1 = confident and wrong.
    df["error"] = (df["prob"] - df["label"]).abs()
    # Keep only misclassified memes: predicted hateful (prob >= 0.5) is not equal to actually hateful.
    # Sort so that the most confident mistakes come first.
    errors = df[(df["prob"] >= 0.5) != (df["label"] == 1)].sort_values("error", ascending=False)
    # For an error, a true label 0 means we predicted hateful -> false positive;
    # a true label 1 means we predicted benign -> false negative.
    errors["kind"] = errors["label"].map({0: "false_positive", 1: "false_negative"})
    # Empty column for manual labelling of the error reason (sarcasm, world knowledge, label noise, ...).
    errors["category"] = ""  # fill by hand

    # Output folder results/errors.
    out = RESULTS_DIR / "errors"
    # Create it if it does not exist.
    out.mkdir(parents=True, exist_ok=True)
    # Save all errors (sorted) to CSV for the manual analysis.
    errors.to_csv(out / f"{args.run}_{args.split}.csv", index=False)

    # Top k most confident false positives (errors are already sorted by confidence).
    fp = errors[errors.kind == "false_positive"].head(args.k)
    # Top k most confident false negatives.
    fn = errors[errors.kind == "false_negative"].head(args.k)
    # Figure grid: 2 rows (FP, FN) x k columns; squeeze=False keeps axes 2-D even when k = 1.
    fig, axes = plt.subplots(2, args.k, figsize=(3 * args.k, 7), squeeze=False)
    # Row 0 shows false positives, row 1 shows false negatives, each with a descriptive name.
    for row, sub, name in [(0, fp, "FP (benign, predicted hateful)"), (1, fn, "FN (hateful, missed)")]:
        # First hide the axes of every cell, so empty cells (fewer than k errors) stay blank.
        for ax in axes[row]:
            ax.axis("off")
        # Fill the cells of this row with the error examples, one per column.
        for ax, (_, r) in zip(axes[row], sub.iterrows()):
            # Open the original meme image, convert to RGB and draw it.
            ax.imshow(Image.open(RAW_DIR / r["img"]).convert("RGB"))
            # Title: error type on the first line and the averaged p(hateful) on the second line.
            ax.set_title(f"{name}\np={r['prob']:.2f}", fontsize=7)
    # Fix spacing between subplots.
    fig.tight_layout()
    # Save the figure as PNG.
    fig.savefig(out / f"{args.run}_{args.split}.png", dpi=110)
    # Print a short summary: total errors, how many FP and FN, and where the files are.
    print(f"{len(errors)} errors ({(errors.kind == 'false_positive').sum()} FP, "
          f"{(errors.kind == 'false_negative').sum()} FN) -> {out}")


# Run main() only when the file is executed directly.
if __name__ == "__main__":
    main()

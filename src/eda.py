"""Exploratory data analysis: label balance, text length, repeated texts, sample memes.

Usage: python src/eda.py      -> results/figures/eda_*.png + printed stats (use them in the Dataset section)
WARNING: the sample grid shows real (possibly offensive) memes.

How it works (overview):
  1. Load train + all evaluation splits.
  2. Print/save a table per split: size, % hateful, mean and max words per caption.
  3. Count captions that are reused on several memes, and how many appear with BOTH labels
     (the dataset's "benign confounders" - proof that text alone is not enough).
  4. Plot label distribution and caption-length histograms -> eda_distribution.png.
  5. Plot 4 non-hateful + 4 hateful example memes -> eda_samples.png.
"""
# matplotlib: plotting library.
import matplotlib

# "Agg" backend draws plots to files only (no window); needed on servers / without a display.
# It must be set BEFORE pyplot is imported.
matplotlib.use("Agg")
# pyplot: the convenient plotting interface (figures, axes, bar/hist plots).
import matplotlib.pyplot as plt
# pandas: tables and grouping for the statistics.
import pandas as pd
# PIL Image: to open the meme PNG files for the sample grid.
from PIL import Image

# Shared paths and loaders from common.py.
from common import FIG_DIR, RAW_DIR, eval_splits, load_split


# ---------------------------------------------------------------------------------------------
# main
# Purpose: compute dataset statistics and save EDA figures for the report.
# Inputs:  the prepared splits in data/splits/.
# Output:  results/eda_stats.csv, results/figures/eda_distribution.png, eda_samples.png, printed stats.
# Viva tip: Q: Why do EDA before training? A: To know class balance (about 1/3 hateful -> accuracy is
#           misleading, so we use AUROC), text lengths (choose BERT max_length) and dataset quirks.
# Viva tip: Q: What are benign confounders? A: Same caption with a different image (or vice versa) that
#           flips the label; they force a model to use BOTH modalities, so unimodal models do poorly.
# ---------------------------------------------------------------------------------------------
def main():
    # Make sure results/figures/ exists before saving plots there.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Dict comprehension: {"train": df, "val": df, "test": df, ...} for train plus every eval split.
    splits = {s: load_split(s) for s in ["train"] + eval_splits()}

    # Collect one statistics row (a dict) per split.
    rows = []
    # Loop over each split name and its DataFrame.
    for name, df in splits.items():
        # Number of words per caption: split text on whitespace, then take the list length -> Series [N].
        words = df["text"].str.split().str.len()
        # Size, % hateful (mean of 0/1 labels x 100), average and maximum caption length in words.
        rows.append({"split": name, "n": len(df), "hateful_%": round(100 * df["label"].mean(), 1),
                     "mean_words": round(words.mean(), 1), "max_words": int(words.max())})
    # Turn the list of dicts into a table (one row per split).
    stats = pd.DataFrame(rows)
    # Print the table without the row index, for the console / report.
    print(stats.to_string(index=False))
    # Save it as results/eda_stats.csv (FIG_DIR.parent = results/).
    stats.to_csv(FIG_DIR.parent / "eda_stats.csv", index=False)

    # Further analysis is done on the training set only.
    train = splits["train"]
    # Same caption reused on different images is how the "benign confounders" were built.
    # Group memes by normalised caption (lower-case, trimmed spaces) and keep only groups with >1 meme,
    # i.e. all memes whose text is shared with at least one other meme.
    dup = train.groupby(train["text"].str.lower().str.strip()).filter(lambda g: len(g) > 1)
    # For each shared caption count distinct labels; .eq(2) is True when the caption appears as both
    # hateful and not hateful; .sum() counts how many such captions there are.
    mixed = dup.groupby(dup["text"].str.lower().str.strip())["label"].nunique().eq(2).sum()
    # Report both numbers: they show that the same text can be hateful or not depending on the image.
    print(f"\ntrain: {len(dup)} memes share their text with another meme; "
          f"{mixed} texts appear with BOTH labels (text-confounders)")

    # Figure 1: one row, two side-by-side plots (11 x 3.5 inches).
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    # Count label 0 and 1 in each split; .T transposes so rows = splits, columns = labels.
    counts = pd.DataFrame({s: d["label"].value_counts().sort_index() for s, d in splits.items()}).T
    # Give the columns readable names (label 0 -> "not hateful", label 1 -> "hateful").
    counts.columns = ["not hateful", "hateful"]
    # Grouped bar chart of class counts per split on the left plot; rot=0 keeps x labels horizontal.
    counts.plot.bar(ax=axes[0], rot=0, title="Label distribution")
    # For each class, plot a histogram of caption length on the right plot.
    for lbl, name in [(0, "not hateful"), (1, "hateful")]:
        # Select train memes of this class, count words, 40 bins; alpha=0.6 so both histograms show.
        axes[1].hist(train[train.label == lbl]["text"].str.split().str.len(), bins=40, alpha=0.6, label=name)
    # Title and x-axis label for the histogram.
    axes[1].set(title="Words per meme (train)", xlabel="words")
    # Show the legend (which colour is which class).
    axes[1].legend()
    # Automatically adjust spacing so titles/labels do not overlap.
    fig.tight_layout()
    # Save the figure as PNG at 150 dots per inch (good quality for a report).
    fig.savefig(FIG_DIR / "eda_distribution.png", dpi=150)

    # Pick 4 random non-hateful and 4 random hateful training memes; random_state=0 -> same picks
    # every run (reproducible); pd.concat stacks the two groups into one table of 8 rows.
    sample = pd.concat([train[train.label == 0].sample(4, random_state=0),
                        train[train.label == 1].sample(4, random_state=0)])
    # Figure 2: a 2 x 4 grid of image panels.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    # axes.ravel() flattens the 2x4 grid to 8 axes; zip pairs each axis with one sampled meme row.
    for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
        # Open the meme image (path relative to the raw data folder), convert to 3-channel RGB, draw it.
        ax.imshow(Image.open(RAW_DIR / r["img"]).convert("RGB"))
        # Title: the label and the first 60 characters of the caption (small font to fit).
        ax.set_title(f"label={r['label']}\n{r['text'][:60]}", fontsize=8)
        # Hide axis ticks/frame - they are meaningless for images.
        ax.axis("off")
    # Tidy spacing between panels.
    fig.tight_layout()
    # Save the sample grid at 120 dpi (smaller file, still readable).
    fig.savefig(FIG_DIR / "eda_samples.png", dpi=120)
    # Tell the user where the figures went.
    print(f"Figures saved to {FIG_DIR}")


# Run main() only when this file is executed directly, not when imported.
if __name__ == "__main__":
    main()

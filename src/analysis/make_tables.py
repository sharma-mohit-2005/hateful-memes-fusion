"""Aggregate every run into the main results table (mean ± std over seeds) + a bar chart.

Usage: python src/analysis/make_tables.py
Output: results/summary.csv, results/summary.md (paste into README/report), results/figures/test_auroc.png

How it works (overview):
- Every training run (text-only BERT, image-only ResNet, early/late fusion, CLIP fusion heads, ...) has a
  folder in results/runs with one sub-folder per random seed. Each seed folder has a metrics.json with the
  AUROC and accuracy on every evaluated split.
- For each run we collect the metrics of all seeds and report "mean ± std" across seeds.
- Why several seeds? Weight initialisation, data order and dropout are random, so one run can be lucky or
  unlucky. The mean gives a fair estimate; the std shows how stable the model is. Two models whose
  difference is smaller than their std are not clearly different.
- Finally we draw a horizontal bar chart of test AUROC (mean with std error bars), sorted from worst to best,
  with a dotted line at 0.5 (random guessing).
"""
# json: reads each seed's metrics.json file.
import json
# sys: lets us change the import search path.
import sys
# Path: OS-independent file paths.
from pathlib import Path

# Add the "src" folder to the import path so that "common" can be imported (must be before that import).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# matplotlib: plotting library.
import matplotlib

# "Agg" backend writes plots to files without opening a window; works on servers with no display.
matplotlib.use("Agg")
# pyplot plotting API (imported after choosing the backend).
import matplotlib.pyplot as plt
# numpy: mean and standard deviation.
import numpy as np
# pandas: builds and saves the summary table.
import pandas as pd

# Project folders: FIG_DIR for figures, RESULTS_DIR for tables, RUNS_DIR containing one folder per run.
from common import FIG_DIR, RESULTS_DIR, RUNS_DIR


# ---------------------------------------------------------------------------------------------------------
# main
# Purpose : read metrics.json of every seed of every run, build the summary table (mean ± std over seeds)
#           and a bar chart of test AUROC.
# Inputs  : none (reads everything under results/runs)
# Outputs : results/summary.csv, results/summary.md and results/figures/test_auroc.png
# Viva tip: Q: Why is AUROC the main metric and not accuracy? A: AUROC does not depend on a fixed threshold
#           and is less affected by class imbalance; it measures how well hateful memes are ranked above
#           benign ones. It is also the official metric of the Hateful Memes challenge.
# Viva tip: Q: What does "± std" mean in your table? A: The standard deviation of the metric over the
#           different random seeds of the same configuration, i.e. how much the result varies by chance.
# ---------------------------------------------------------------------------------------------------------
def main():
    # rows = table rows (one per run); plot = (run name, mean test AUROC, std test AUROC) for the chart.
    rows, plot = [], []
    # Loop over every run folder in alphabetical order (files are ignored, only directories).
    for run_path in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        # Load metrics.json of every seed folder (seed0, seed1, ...) of this run into a list of dicts.
        # Each dict looks like {"dev": {"auroc": ..., "acc": ...}, "test": {...}}.
        metrics = [json.loads(m.read_text()) for m in sorted(run_path.glob("seed*/metrics.json"))]
        # Skip runs that have no finished seeds.
        if not metrics:
            continue
        # Start the row with the run name and the number of seeds found.
        row = {"run": run_path.name, "seeds": len(metrics)}
        # Loop over the split names stored in the first seed's metrics (e.g. "dev", "test").
        for split in metrics[0]:
            # For both metrics: AUROC and accuracy.
            for m in ["auroc", "acc"]:
                # Collect this metric for this split from every seed -> array of length num_seeds.
                vals = np.array([x[split][m] for x in metrics])
                # Store as text "mean ± std" with 4 decimals, in a column like "test_auroc".
                row[f"{split}_{m}"] = f"{vals.mean():.4f} ± {vals.std():.4f}"
                # Remember test AUROC numbers separately for the bar chart.
                if split == "test" and m == "auroc":
                    plot.append((run_path.name, vals.mean(), vals.std()))
        # Add the finished row to the table.
        rows.append(row)
    # If no run had any metrics, stop the script with a clear message.
    if not rows:
        raise SystemExit("no runs found in results/runs")

    # Build the summary table: one row per run, one column per split/metric.
    df = pd.DataFrame(rows)
    # Save as CSV.
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    # Save as a Markdown table that can be pasted into the README/report.
    (RESULTS_DIR / "summary.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    # Print the table to the terminal.
    print(df.to_string(index=False))

    # Draw the chart only if at least one run has test results.
    if plot:
        # Sort runs by mean test AUROC (ascending), so the best model ends up at the top of a barh chart.
        plot.sort(key=lambda r: r[1])
        # Unzip the list of tuples into three tuples: names, means and stds.
        names, means, stds = zip(*plot)
        # Make sure the figure folder exists.
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        # Figure height grows with the number of runs (0.4 inch per bar + margin) so labels do not overlap.
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1.5))
        # Horizontal bars of mean AUROC; xerr draws the std over seeds as error bars with small caps.
        ax.barh(names, means, xerr=stds, color="#4C72B0", capsize=3)
        # Dotted grey vertical line at 0.5 = AUROC of random guessing, as a reference.
        ax.axvline(0.5, ls=":", c="grey")
        # Start the x-axis at 0.45 (zoom in on the interesting range) and leave some space after the best.
        ax.set_xlim(0.45, max(means) + 0.05)
        # Label the x-axis.
        ax.set_xlabel("test AUROC (mean ± std over seeds)")
        # Fix margins so the long run names are not cut off.
        fig.tight_layout()
        # Save the chart as PNG with high resolution (150 dpi) for the report.
        fig.savefig(FIG_DIR / "test_auroc.png", dpi=150)


# Run main() only when the file is executed directly.
if __name__ == "__main__":
    main()

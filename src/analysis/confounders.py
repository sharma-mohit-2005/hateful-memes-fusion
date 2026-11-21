"""Benign-confounder error analysis (research angle B).

1. Finds confounder pairs inside an eval split:
     text-confounder : same text, different image, different label
     image-confounder: (near-)same image, different text, different label
   Image similarity uses CLIP image embeddings. Best with the inpainted features (overlay text removed),
   because then two memes with the same base picture are almost identical.
2. For every run, computes on those pairs:
     pair_acc   both memes of the pair classified correctly (threshold 0.5)
     pair_rank  the hateful meme gets a higher probability than its benign twin (threshold-free)
   and normal AUROC/acc on confounder memes vs. the rest.
3. Saves a figure of example pairs with model probabilities.

Usage: python src/analysis/confounders.py --split test --sim_features clip_l14_inpainted --show_run C_clip_l14_cross
Outputs: results/confounders/{<split>_pairs.csv, results_<split>.csv/.md, pairs_<split>.png}

How it works (overview):
- What is a "benign confounder"? Meta built the Hateful Memes dataset so that for many hateful memes there is
  a "twin" meme that is NOT hateful. The twin either keeps the same text and swaps the image, or keeps the
  same image and swaps the text. Changing just one modality flips the label from hateful to benign.
- Why does the dataset do this? To defeat unimodal models. A text-only model sees identical input for both
  memes of a text-confounder pair, so it MUST give both the same probability and cannot get both right.
  The same holds for an image-only model on image-confounder pairs. Only a model that combines both
  modalities can separate the twins.
- Step 1 (find_pairs): group memes by normalized text to find same-text pairs; use cosine similarity of
  CLIP image embeddings to find near-identical-image pairs. A pair is a confounder if the labels differ.
- Step 2 (pair_scores + main loop): for every trained run and seed, measure pair_acc and pair_rank on the
  confounder pairs, plus AUROC/accuracy on confounder memes and accuracy on all other memes.
  Report mean ± std over seeds.
- Step 3: draw a figure with a few example pairs side by side, optionally showing one model's p(hate).
"""
# argparse: reads command-line options such as --split and --sim_threshold.
import argparse
# sys: needed so we can edit sys.path (the list of folders Python searches for imports).
import sys
# Path: an object-oriented, OS-independent way to build and handle file paths.
from pathlib import Path

# Add the "src" folder (two levels up from this file: analysis -> src) to the import path.
# This lets us import our shared helper module "common" which lives in src/.
# It MUST come before "from common import ..." below, otherwise Python cannot find "common".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# matplotlib is the plotting library.
import matplotlib

# Use the "Agg" backend: it draws plots into image files only, with no window.
# This is needed on servers / remote GPUs that have no display; otherwise plotting could crash.
matplotlib.use("Agg")
# pyplot gives the familiar plt.subplots / imshow / savefig API (imported AFTER choosing the backend).
import matplotlib.pyplot as plt
# numpy: fast arrays and maths (normalizing embeddings, matrix product, mean/std).
import numpy as np
# pandas: DataFrames (tables) for the memes, the pairs and the result table.
import pandas as pd
# PIL Image: opens the meme image files so we can show them in the figure.
from PIL import Image
# accuracy_score: fraction of correct predictions; roc_auc_score: AUROC, our main metric.
from sklearn.metrics import accuracy_score, roc_auc_score

# Shared project helpers: folder paths and loaders for data splits, CLIP features and saved predictions.
# FEAT_DIR = folder of extracted features, RAW_DIR = original images, RESULTS_DIR = output folder,
# RUNS_DIR = one sub-folder per trained model run; run_seeds(run) lists the seeds trained for that run.
from common import FEAT_DIR, RAW_DIR, RESULTS_DIR, RUNS_DIR, load_features, load_preds, load_split, run_seeds


# ---------------------------------------------------------------------------------------------------------
# find_pairs
# Purpose : find all candidate pairs of memes that share the same text OR have (almost) the same image,
#           and mark which of them are benign confounders (the two memes have different labels).
# Inputs  : df            - DataFrame of one split with columns id, img, text, label (row i <-> embedding i)
#           img_emb       - numpy array of shape (N, D): one CLIP image embedding per meme
#           sim_threshold - cosine similarity above which two images count as "the same picture"
# Output  : DataFrame with one row per pair: type ("text"/"image"), sim, id_a/id_b, label_a/label_b,
#           text_a/text_b and a boolean column "confounder" (True if the labels differ).
# Viva tip: Q: Why normalize the text before comparing? A: Captions differ in case, punctuation and
#           spaces; without normalization true duplicates would be missed.
# Viva tip: Q: How do you compute cosine similarity for all pairs at once? A: Divide every embedding by its
#           L2 norm, then the matrix product emb @ emb.T gives all pairwise cosines.
# Viva tip: Q: Why np.triu(sim, 1)? A: The similarity matrix is symmetric and the diagonal is 1 (self);
#           keeping only the upper triangle above the diagonal takes each pair exactly once.
# ---------------------------------------------------------------------------------------------------------
def find_pairs(df, img_emb, sim_threshold):
    # Normalize every caption: lowercase -> remove everything except letters, digits and spaces ->
    # split on whitespace and re-join with single spaces (removes extra/leading/trailing spaces).
    # Result: "Look at THIS!!  " and "look at this" become the same string "look at this".
    norm_text = df["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.split().str.join(" ")
    # List that will collect tuples (type, index_a, index_b, similarity).
    pairs = []
    # Add the normalized text as a temporary column "nt" and group the memes by it.
    # Every group with 2+ memes = memes that share the same caption (text-confounder candidates).
    for nt, g in df.assign(nt=norm_text).groupby("nt"):
        # Skip memes whose normalized text is empty (e.g. no text / only symbols); they are not real matches.
        if not nt:
            continue
        # Row indices of all memes in this same-text group.
        idx = g.index.tolist()
        # Two nested loops with b > a: generate every unordered pair inside the group exactly once.
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                # Store as a "text" pair; similarity is 1.0 because the (normalized) texts are identical.
                pairs.append(("text", idx[a], idx[b], 1.0))

    # L2-normalize every image embedding (divide each row by its length) -> every vector has length 1.
    # keepdims=True keeps shape (N, 1) so the division broadcasts over the D columns.
    emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    # For unit vectors the dot product equals cosine similarity. emb (N, D) @ emb.T (D, N) -> sim (N, N),
    # where sim[i, j] = cosine similarity between image i and image j (1 = identical direction).
    sim = emb @ emb.T
    # np.triu(sim, 1) keeps only entries above the main diagonal (j > i) and sets the rest to 0.
    # This removes self-pairs (diagonal) and duplicates (sim is symmetric, so (i,j) and (j,i) are equal).
    # np.where(... >= threshold) returns the row indices ia and column indices ib of very similar images.
    ia, ib = np.where(np.triu(sim, 1) >= sim_threshold)
    # Go through each pair of near-identical images.
    for a, b in zip(ia, ib):
        # Only keep it as an "image" pair if the texts are different; same image + same text would just be
        # a duplicate meme (and same-text pairs were already added above).
        if norm_text[a] != norm_text[b]:
            # Store with its actual cosine similarity so we can inspect/tune the threshold later.
            pairs.append(("image", a, b, float(sim[a, b])))

    # Turn the list of tuples into a DataFrame; a and b are row positions in df.
    out = pd.DataFrame(pairs, columns=["type", "a", "b", "sim"])
    # Look up the meme ids of both members using the positions a and b (.values = plain numpy array).
    out["id_a"], out["id_b"] = df["id"].values[out["a"]], df["id"].values[out["b"]]
    # Look up both labels (0 = benign, 1 = hateful).
    out["label_a"], out["label_b"] = df["label"].values[out["a"]], df["label"].values[out["b"]]
    # Look up both raw captions (useful when reading the CSV by hand).
    out["text_a"], out["text_b"] = df["text"].values[out["a"]], df["text"].values[out["b"]]
    # A pair is a benign confounder only if the labels differ: one modality changed and the label flipped.
    out["confounder"] = out["label_a"] != out["label_b"]
    # Drop the internal position columns; ids are what we need from here on.
    return out.drop(columns=["a", "b"])


# ---------------------------------------------------------------------------------------------------------
# pair_scores
# Purpose : score one model's predictions on the confounder pairs with two pair-level metrics.
# Inputs  : pairs - DataFrame from find_pairs (we only use rows where confounder is True)
#           preds - DataFrame of one run+seed predictions with columns id, prob (p(hateful)), label
# Output  : tuple (pair_acc, pair_rank), both fractions between 0 and 1.
#           pair_acc  = fraction of pairs where BOTH memes are classified correctly at threshold 0.5.
#           pair_rank = fraction of pairs where the hateful meme gets a higher probability than its benign
#                       twin. It does not need a threshold (like AUROC restricted to each pair).
# Viva tip: Q: Why should a text-only model be near chance on text-confounders? A: Both memes have the same
#           text, so it gives both the same probability: pair_rank is 0 (no strict ">") and at most one of
#           the two can be correct at 0.5, so pair_acc is 0.
# Viva tip: Q: Why report pair_rank as well as pair_acc? A: pair_acc depends on the 0.5 threshold, which may
#           be badly calibrated; pair_rank only asks whether the model ranks the twins in the right order.
# ---------------------------------------------------------------------------------------------------------
def pair_scores(pairs, preds):
    # Make a Series: meme id -> predicted probability of "hateful", so we can look up by id quickly.
    p = preds.set_index("id")["prob"]
    # Keep only true confounder pairs (labels differ); same-label pairs say nothing about modality use.
    conf = pairs[pairs["confounder"]]
    # Probabilities for meme a and meme b of every pair, as numpy arrays of length = number of pairs.
    pa, pb = p.loc[conf["id_a"]].values, p.loc[conf["id_b"]].values
    # True labels for meme a and meme b of every pair.
    la, lb = conf["label_a"].values, conf["label_b"].values
    # Prediction is "hateful" if prob >= 0.5; compare to the label (True == 1, False == 0).
    # The pair counts as correct only if BOTH memes are right (logical AND with &).
    both_correct = ((pa >= 0.5) == la) & ((pb >= 0.5) == lb)
    # For each pair: if a is the hateful one, check pa > pb; otherwise (b is hateful) check pb > pa.
    # np.where picks element-wise between the two boolean arrays.
    hateful_higher = np.where(la == 1, pa > pb, pb > pa)
    # Mean of a boolean array = fraction of True values -> pair_acc and pair_rank.
    return both_correct.mean(), hateful_higher.mean()


# ---------------------------------------------------------------------------------------------------------
# main
# Purpose : run the whole confounder analysis: mine pairs, score every saved run on them, write the result
#           tables, and save a figure with example pairs.
# Inputs  : command-line arguments --split, --sim_features, --sim_threshold, --show_run
# Outputs : results/confounders/<split>_pairs.csv, results_<split>.csv/.md and pairs_<split>.png
# Viva tip: Q: Why use the inpainted CLIP features to find image pairs? A: The overlay caption is part of the
#           pixels. If it stays, two memes with the same background but different text look different to
#           CLIP. After removing the text, same-background memes become almost identical, so pairs are found.
# Viva tip: Q: Why mean ± std over seeds? A: Training randomness changes results; several seeds show if a
#           difference between models is real or just noise.
# ---------------------------------------------------------------------------------------------------------
def main():
    # Create the command-line argument parser.
    ap = argparse.ArgumentParser()
    # Which data split to analyse (e.g. "test" or "dev"); default "test".
    ap.add_argument("--split", default="test")
    # Name of the feature set whose IMAGE embeddings are used to measure image similarity.
    ap.add_argument("--sim_features", default="clip_l14_inpainted",
                    help="feature tag used for image similarity (falls back to clip_l14)")
    # Cosine similarity cut-off for "same image"; 0.93 is strict so only near-duplicates count.
    ap.add_argument("--sim_threshold", type=float, default=0.93)
    # Optional: name of one run whose probabilities are written on the example figure.
    ap.add_argument("--show_run", default=None, help="run whose probabilities are printed on the figure")
    # Parse the arguments actually given on the command line.
    args = ap.parse_args()

    # Output folder results/confounders.
    out_dir = RESULTS_DIR / "confounders"
    # Create it (and parents) if missing; exist_ok=True means no error if it already exists.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load the split as a DataFrame (id, img, text, label).
    df = load_split(args.split)
    # Use the requested feature set if its folder exists; otherwise fall back to the normal clip_l14
    # features (for example when the inpainting step has not been run yet).
    tag = args.sim_features if (FEAT_DIR / args.sim_features).exists() else "clip_l14"
    # Load the saved features for this split (a dict with "ids", "img", "txt", ...).
    feats = load_features(tag, args.split)
    # Safety check: the feature rows must be in exactly the same order as the DataFrame rows,
    # otherwise row i of the embeddings would belong to a different meme than row i of df.
    assert (feats["ids"].astype(str) == df["id"].values).all()
    # Mine all text/image pairs using the image embeddings, shape (N, D).
    pairs = find_pairs(df, feats["img"], args.sim_threshold)
    # Save every pair (confounder or not) so we can inspect them by hand.
    pairs.to_csv(out_dir / f"{args.split}_pairs.csv", index=False)
    # Keep only the real confounder pairs (labels differ).
    conf = pairs[pairs["confounder"]]
    # Print a summary: how many pairs in total and how many confounders of each type.
    print(f"[{args.split}] similarity from '{tag}', thr={args.sim_threshold}: {len(pairs)} pairs, "
          f"{len(conf)} confounder pairs ({(conf.type == 'text').sum()} text, {(conf.type == 'image').sum()} image)")
    # Reminder to check pairs manually: a bad threshold can create false image pairs or miss real ones.
    print("-> open the pairs csv / figure and sanity-check a few image pairs; tune --sim_threshold if needed")

    # Set of all meme ids that belong to at least one confounder pair (union of side a and side b).
    conf_ids = set(conf["id_a"]) | set(conf["id_b"])
    # One result row per run will be collected here.
    rows = []
    # Loop over every run folder in RUNS_DIR (sorted by name so the table order is stable).
    for run in sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir()):
        # Per-seed metric lists for this run.
        stats = []
        # Each run was trained with several random seeds; evaluate each seed separately.
        for seed in run_seeds(run):
            # Load the saved predictions (id, prob, label) of this run/seed on this split.
            try:
                preds = load_preds(run, seed, args.split)
            # Some runs may not have predictions for this split; just skip that seed.
            except FileNotFoundError:
                continue
            # Boolean mask: True for memes that are part of a confounder pair.
            in_conf = preds["id"].isin(conf_ids)
            # The two pair metrics for this seed.
            pacc, prank = pair_scores(pairs, preds)
            # Predictions on confounder memes only.
            sub = preds[in_conf]
            # Predictions on all other (non-confounder) memes, for comparison.
            rest = preds[~in_conf]
            # Store 5 numbers: pair_acc, pair_rank, AUROC on confounder memes, accuracy (threshold 0.5) on
            # confounder memes, and accuracy on non-confounder memes. If conf_acc << nonconf_acc, the
            # confounders are the hard part of the dataset.
            stats.append([pacc, prank, roc_auc_score(sub["label"], sub["prob"]),
                          accuracy_score(sub["label"], sub["prob"] >= 0.5),
                          accuracy_score(rest["label"], rest["prob"] >= 0.5)])
        # Only add a row if at least one seed had predictions.
        if stats:
            # Convert to array of shape (num_seeds, 5) so we can take mean/std per column.
            s = np.array(stats)
            # Start the row with the run name and how many seeds were used.
            row = {"run": run, "seeds": len(s)}
            # For each of the 5 metrics (column k), write "mean ± std" across seeds as a text cell.
            for k, name in enumerate(["pair_acc", "pair_rank", "conf_auroc", "conf_acc", "nonconf_acc"]):
                row[name] = f"{s[:, k].mean():.3f} ± {s[:, k].std():.3f}"
            # Add this run to the table.
            rows.append(row)
    # Build the result table: one row per run.
    res = pd.DataFrame(rows)
    # Save as CSV (for further processing).
    res.to_csv(out_dir / f"results_{args.split}.csv", index=False)
    # Save as a Markdown table (to paste into the report/README).
    (out_dir / f"results_{args.split}.md").write_text(res.to_markdown(index=False), encoding="utf-8")
    # Also print the table in the terminal.
    print(res.to_string(index=False))

    # Pick up to 3 example pairs of each type ("text" and "image") for the figure.
    show = conf.groupby("type").head(3)
    # Probabilities to print on the figure (None = do not print any).
    probs = None
    # If a run was given, load the predictions of its first seed as a Series id -> prob.
    if args.show_run:
        probs = load_preds(args.show_run, run_seeds(args.show_run)[0], args.split).set_index("id")["prob"]
    # Grid: one row per example pair, 2 columns (meme a on the left, meme b on the right).
    # squeeze=False keeps axes as a 2-D array even if there is only one row.
    fig, axes = plt.subplots(len(show), 2, figsize=(8, 3.6 * len(show)), squeeze=False)
    # Series: meme id -> relative image path, to find each meme's image file.
    img_of = df.set_index("id")["img"]
    # For each figure row, take the matching pair r.
    for row_ax, (_, r) in zip(axes, show.iterrows()):
        # Left axis shows side "a", right axis shows side "b".
        for ax, side in zip(row_ax, "ab"):
            # Meme id of this side.
            mid = r[f"id_{side}"]
            # Open the original image and convert to RGB (some files may be RGBA or grayscale), then draw it.
            ax.imshow(Image.open(RAW_DIR / img_of[mid]).convert("RGB"))
            # Title shows the pair type and the true label of this meme.
            title = f"{r['type']}-confounder | label={r[f'label_{side}']}"
            # If we have model probabilities, also show the predicted p(hateful).
            if probs is not None:
                title += f" | p(hate)={probs[mid]:.2f}"
            # Small font so the title fits above the image.
            ax.set_title(title, fontsize=8)
            # Hide axis ticks/labels; they are meaningless for a picture.
            ax.axis("off")
    # Adjust spacing so titles and images do not overlap.
    fig.tight_layout()
    # Save the figure as a PNG file.
    fig.savefig(out_dir / f"pairs_{args.split}.png", dpi=110)
    # Tell the user where the outputs are.
    print(f"saved to {out_dir}")


# Run main() only when this file is executed as a script (not when it is imported by another module).
if __name__ == "__main__":
    main()

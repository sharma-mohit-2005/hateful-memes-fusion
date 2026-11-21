"""M4 late fusion: combine the probabilities of the text-only and image-only models.

  late_avg       p = 0.5 * p_text + 0.5 * p_image
  late_weighted  p = w * p_text + (1 - w) * p_image, w tuned on val AUROC

Usage: python src/late_fusion.py --text_run M1_bert_text --image_run M2_resnet_image
Can also fuse CLIP unimodal heads: --text_run C_clip_text_only --image_run C_clip_image_only --prefix C_late

How it works (overview):
  1. No new neural network is trained. We reuse the saved predictions (probability of "hateful") of an
     already-trained text model and an already-trained image model, for the same seed.
  2. For each split we join the two prediction tables on the meme id, so each row has p_text and p_image.
  3. late_avg: simple average with equal weights (w = 0.5).
  4. late_weighted: try w = 0.00, 0.05, ..., 1.00 (grid search, 21 values) and pick the w with the
     highest AUROC on the VALIDATION set. The same w is then applied to the test set.
  5. Both variants are saved like any other model run, so the same evaluation scripts can compare them.
  Late fusion is a "decision-level" fusion: each modality decides separately, and we only combine the
  final scores. It cannot learn interactions between text and image (e.g. a harmless caption that
  becomes hateful only with a specific picture), which is the main weakness compared to early fusion.
"""
# argparse: read command-line options (which runs to fuse and how to name the output).
import argparse

# numpy: create the grid of weights and find the best one (argmax).
import numpy as np
# roc_auc_score: computes AUROC, the main metric of the Hateful Memes challenge.
from sklearn.metrics import roc_auc_score

# Shared helpers: list of eval splits, loading saved predictions, listing seeds of a run, saving results.
from common import eval_splits, load_preds, run_seeds, save_run


# ----------------------------------------------------------------------------------------------------
# main
# Purpose : for every seed, fuse text-model and image-model probabilities with (a) equal weights and
#           (b) a weight tuned on validation AUROC, and save both as new runs.
# Inputs  : --text_run, --image_run (names of saved runs), --prefix (name of the output runs).
# Outputs : results/runs/<prefix>_avg/seed<k>/ and results/runs/<prefix>_weighted/seed<k>/.
# Viva tip: Q: Why tune w on validation and not on test? A: Test must stay untouched; tuning on test
#           would make the test score optimistic (information leakage).
# Viva tip: Q: Why average probabilities and not logits? A: Both models output calibrated-ish values in
#           0-1 on the same scale, so a weighted average is simple and interpretable; w shows how much
#           the fused model relies on text vs image.
# Viva tip: Q: Difference from early fusion? A: Late fusion combines final decisions of two separate
#           models; early fusion combines features and trains one classifier jointly.
# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Run name of the text-only model whose predictions we fuse (default: M1 BERT).
    ap.add_argument("--text_run", default="M1_bert_text")
    # Run name of the image-only model (default: M2 ResNet-50).
    ap.add_argument("--image_run", default="M2_resnet_image")
    # Prefix for the output run names, e.g. M4_late_avg and M4_late_weighted.
    ap.add_argument("--prefix", default="M4_late")
    args = ap.parse_args()

    # Only fuse seeds that exist for BOTH runs (set intersection); sorted for a stable order.
    # Seed k of the text model is paired with seed k of the image model.
    seeds = sorted(set(run_seeds(args.text_run)) & set(run_seeds(args.image_run)))
    for seed in seeds:
        # merged[split] will be a table with columns id, label, prob_text, prob_image.
        merged = {}
        for split in eval_splits():
            # Saved predictions of the text model for this split (columns id, prob, label).
            t = load_preds(args.text_run, seed, split)
            # Saved predictions of the image model for this split.
            i = load_preds(args.image_run, seed, split)
            # Inner join on meme id so each row lines up the two models' predictions for the SAME meme.
            # Only id and prob are taken from the image table (label already comes from the text table);
            # the duplicate "prob" columns get suffixes -> prob_text and prob_image.
            merged[split] = t.merge(i[["id", "prob"]], on="id", suffixes=("_text", "_image"))

        # Validation split is used to choose the weight.
        val = merged["val"]
        # Grid of candidate text weights: 0.00, 0.05, ..., 1.00 (21 values).
        # w = 0 means image only, w = 1 means text only.
        grid = np.linspace(0, 1, 21)
        # Grid search: validation AUROC of the fused probability for every candidate w.
        aucs = [roc_auc_score(val["label"], w * val["prob_text"] + (1 - w) * val["prob_image"]) for w in grid]
        # Weight with the highest validation AUROC (argmax gives its index in the grid).
        best_w = float(grid[int(np.argmax(aucs))])

        # Produce two fused models: the plain average and the val-tuned weighted average.
        for name, w in [("avg", 0.5), ("weighted", best_w)]:
            # preds[split] = fused predictions for that split.
            preds = {}
            for split, df in merged.items():
                # Keep the id and true label columns (copy so the merged table is not modified).
                out = df[["id", "label"]].copy()
                # Fused probability = convex combination of the two probabilities (stays in 0-1).
                out["prob"] = w * df["prob_text"] + (1 - w) * df["prob_image"]
                preds[split] = out
            # Save in the same format as trained models (predictions + metrics + config incl. w_text),
            # so the comparison/aggregation scripts treat late fusion like any other model.
            save_run(f"{args.prefix}_{name}", seed, preds,
                     {"text_run": args.text_run, "image_run": args.image_run, "w_text": w})
        # Report the chosen weight: > 0.5 means the fused model trusts the text model more.
        print(f"seed{seed}: best text weight on val = {best_w:.2f}")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()

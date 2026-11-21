"""M5: CLIP linear probe - logistic regression on [normalized image emb ; normalized text emb].
The regularization strength C is chosen on val AUROC. Deterministic, so saved as seed0 only.

Usage: python src/linear_probe.py [--features clip_l14]

How it works:
  A "linear probe" tests how much useful information is already in the frozen CLIP features by
  training only a LINEAR classifier on top of them (no hidden layers, no feature interaction).
  1. Load cached CLIP features (img [N,768], txt [N,768]) for train and the eval splits.
  2. For each mode - image only, text only, concat [img ; txt] - build the input matrix X from
     L2-normalized embeddings.
  3. Fit sklearn LogisticRegression for several values of C (inverse regularization strength) and keep
     the one with the best validation AUROC.
  4. Predict P(hateful) on every eval split with the chosen model and save the run.
  It is the simplest CLIP baseline; the MLP fusion heads (train_head.py) must beat it to justify
  their extra complexity.
"""
# argparse: command-line option --features.
import argparse

# numpy: normalization and concatenation of feature matrices.
import numpy as np
# pandas: prediction tables (id, prob, label).
import pandas as pd
# Logistic regression: linear model p = sigmoid(w.x + b), trained with L2 regularization by default.
from sklearn.linear_model import LogisticRegression
# AUROC metric used to pick the best C on validation.
from sklearn.metrics import roc_auc_score

# Project helpers: eval split names, cached feature loader, run saver.
from common import eval_splits, load_features, save_run


# ---------------------------------------------------------------------------------------------------
# design
# Purpose : Build the design matrix X (inputs to logistic regression) for one split and one mode.
# Inputs  : feats = npz with "img" [N,768], "txt" [N,768]; mode = "image" | "text" | "concat".
# Outputs : X of shape [N,768] (image/text) or [N,1536] (concat).
# Viva tip: Q: Why L2-normalize? A: CLIP compares vectors by cosine similarity, so only direction
#           matters; unit length puts image and text on the same scale before concatenation.
# Viva tip: Q: Can a linear probe on concat model image-text interaction? A: No - it is a weighted sum
#           of image features plus a weighted sum of text features; there is no multiplication between
#           them. That is why we also try product / cross fusions in fusion_heads.py.
# ---------------------------------------------------------------------------------------------------
def design(feats, mode):
    # Helper: divide each row by its Euclidean (L2) norm -> every row has length 1. keepdims keeps [N,1]
    # so the division broadcasts over all 768 columns.
    norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    # Pick which normalized blocks to use for this mode (a list of arrays).
    parts = {"image": [norm(feats["img"])], "text": [norm(feats["txt"])],
             "concat": [norm(feats["img"]), norm(feats["txt"])]}[mode]
    # Join the blocks column-wise: [N,768] for one block, [N,1536] for concat.
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------------------------------
# main
# Purpose : For each mode, tune C on validation, then save predictions of the best logistic regression.
# Inputs  : --features (which cached feature folder).
# Outputs : runs M5_linprobe_<features>_<mode> (seed 0) with predictions and the chosen C.
# Viva tip: Q: What is C? A: The INVERSE of regularization strength. Small C = strong L2 penalty =
#           simpler model (may underfit); large C = weak penalty (may overfit). We pick it on val.
# Viva tip: Q: Why only one seed? A: Logistic regression with the lbfgs solver is a convex problem with a
#           unique optimum, so the result is deterministic - more seeds would give identical numbers.
# Viva tip: Q: Why choose C on val, not test? A: Choosing on test would leak test information and
#           give an optimistic, unfair score; test is only used for the final report.
# ---------------------------------------------------------------------------------------------------
def main():
    # Command-line parser.
    ap = argparse.ArgumentParser()
    # Which cached feature folder to use (e.g. clip_l14).
    ap.add_argument("--features", default="clip_l14")
    # Parse the command line.
    args = ap.parse_args()

    # Training features (img, txt, label, ids).
    train = load_features(args.features, "train")
    # Features of every evaluation split, e.g. {"val": ..., "test": ...}.
    evals = {s: load_features(args.features, s) for s in eval_splits()}
    # Train a separate probe for image-only, text-only and concatenated features.
    for mode in ["image", "text", "concat"]:
        # Training design matrix for this mode.
        X = design(train, mode)
        # Will hold (best val AUROC, its C, its fitted classifier).
        best = None
        # Grid of C values on a roughly log scale (x3 steps) from strong to weak regularization.
        for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
            # Fit logistic regression; max_iter=3000 gives the solver enough iterations to converge.
            clf = LogisticRegression(C=C, max_iter=3000).fit(X, train["label"])
            # Val AUROC: predict_proba returns [N,2] = [P(class 0), P(class 1)]; [:, 1] = P(hateful).
            auc = roc_auc_score(evals["val"]["label"], clf.predict_proba(design(evals["val"], mode))[:, 1])
            # Keep this C if it is the first one or beats the best so far.
            if best is None or auc > best[0]:
                best = (auc, C, clf)
        # Unpack the winner (val AUROC itself is not needed here).
        _, C, clf = best
        # Prediction table per eval split: id, P(hateful) from the best model, true label.
        preds = {s: pd.DataFrame({"id": f["ids"].astype(str), "prob": clf.predict_proba(design(f, mode))[:, 1],
                                  "label": f["label"].astype(int)}) for s, f in evals.items()}
        # Save as run "M5_linprobe_<features>_<mode>", seed 0, recording the chosen C in the config.
        save_run(f"M5_linprobe_{args.features}_{mode}", 0, preds, {"features": args.features, "mode": mode, "C": C})


# Run main() only when executed as a script.
if __name__ == "__main__":
    main()

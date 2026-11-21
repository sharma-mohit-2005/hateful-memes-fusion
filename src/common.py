"""Shared paths, data loading, metrics and run-saving helpers used by every script.

How it works (overview):
  * Defines all project folder paths in ONE place, so every script reads/writes the same locations.
  * set_seed / get_device: reproducibility and choosing GPU vs CPU.
  * load_split / eval_splits: read the normalized train/val/test jsonl files made by prepare_splits.py.
  * compute_metrics: AUROC (main metric), accuracy and F1 for binary "hateful vs not hateful".
  * run_dir / save_run / load_preds / run_seeds: save and reload every model's predictions per seed,
    so later analysis scripts (tables, late fusion, plots) never need to retrain a model.
  * load_features: load cached frozen-CLIP embeddings saved by extract_clip.py.
"""
# json: to write metrics/config dictionaries as human-readable .json files.
import json
# random: Python's built-in random number generator (we seed it for reproducibility).
import random
# Path: object-oriented, OS-independent file paths (works on Windows and Linux with "/").
from pathlib import Path

# numpy: fast arrays; used for labels/probabilities and for loading cached features (.npz).
import numpy as np
# pandas: DataFrames (tables) for the dataset rows (id, img, text, label) and predictions.
import pandas as pd
# torch: PyTorch deep learning library (tensors, GPU, random seeds for weight init / dropout).
import torch
# sklearn metrics: standard, well-tested implementations of accuracy, F1 and ROC-AUC.
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ROOT = project folder. __file__ is this file (src/common.py); .resolve() makes it an absolute path;
# .parents[1] goes up two levels (src/ -> project root). This makes paths work from any working dir.
ROOT = Path(__file__).resolve().parents[1]
# All data lives under <project>/data.
DATA_DIR = ROOT / "data"
# Raw dataset as downloaded from Hugging Face (original jsonl files + img/ folder of PNG memes).
RAW_DIR = DATA_DIR / "hateful_memes"      # downloaded dataset (jsonl files + img/)
# Clean, normalized splits we actually train/evaluate on (created by prepare_splits.py).
SPLIT_DIR = DATA_DIR / "splits"           # normalized train/val/test jsonl (made by prepare_splits.py)
# Images with the overlaid caption text removed, so the image model cannot "read" the text.
INPAINT_DIR = DATA_DIR / "inpainted"      # memes with overlay text removed (made by inpaint_text.py)
# Cached CLIP embeddings: computing CLIP once and reusing it saves a lot of GPU time.
FEAT_DIR = ROOT / "features"              # cached CLIP embeddings
# Everything produced by experiments (predictions, metrics, figures).
RESULTS_DIR = ROOT / "results"
# results/runs/<model_name>/seed<k>/ : one folder per model and per random seed.
RUNS_DIR = RESULTS_DIR / "runs"           # one folder per model / seed
# results/figures/ : plots for the report (EDA, ROC curves, etc.).
FIG_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------------------------
# set_seed
# Purpose: fix every random number generator so that a run can be repeated with the same result.
# Input:   seed (int), e.g. 0, 1, 2.
# Output:  nothing (changes global random state).
# Viva tip: Q: Why do we fix random seeds? A: Weight initialisation, data shuffling and dropout are
#           random; fixing the seed makes experiments reproducible and lets us compare models fairly.
# Viva tip: Q: Why run several seeds? A: One seed can be lucky/unlucky; mean +/- std over seeds
#           shows whether a difference between models is real or just noise.
# ---------------------------------------------------------------------------------------------
def set_seed(seed: int):
    # Seed Python's own random module (used e.g. by some shuffling / augmentation code).
    random.seed(seed)
    # Seed NumPy's global generator (used by numpy-based sampling, sklearn helpers, etc.).
    np.random.seed(seed)
    # Seed PyTorch on the CPU (weight initialisation, dropout masks, DataLoader shuffling).
    torch.manual_seed(seed)
    # Seed PyTorch on ALL GPUs (each GPU has its own generator); safe to call even with no GPU.
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------------------------
# get_device
# Purpose: pick where tensors/models run.
# Input:   none.   Output: torch.device("cuda") if an NVIDIA GPU is available, else "cpu".
# Viva tip: Q: Why use a GPU? A: Neural nets are mostly big matrix multiplications, which a GPU does
#           in parallel, so training/inference is many times faster than on a CPU.
# ---------------------------------------------------------------------------------------------
def get_device():
    # Conditional expression: "cuda" when CUDA is available, otherwise fall back to "cpu".
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------
# load_split
# Purpose: read one prepared split (train / val / test / test_unseen) into a DataFrame.
# Input:   name (str) of the split.
# Output:  pandas DataFrame with columns id, img, text, label (one row per meme).
# Viva tip: Q: Why jsonl format? A: JSON Lines = one JSON object per line; easy to stream, append and
#           read with pandas, and it is the original format of the Hateful Memes dataset.
# ---------------------------------------------------------------------------------------------
def load_split(name: str) -> pd.DataFrame:
    # Build the file path, e.g. data/splits/train.jsonl.
    path = SPLIT_DIR / f"{name}.jsonl"
    # Fail early with a helpful message if the user forgot to run prepare_splits.py.
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python src/prepare_splits.py` first")
    # lines=True -> read JSON Lines. dtype id=str keeps leading zeros (e.g. "01235"), which matter
    # because ids are also used in image file names and to join predictions of different models.
    return pd.read_json(path, lines=True, dtype={"id": str})


# ---------------------------------------------------------------------------------------------
# eval_splits
# Purpose: list all labelled evaluation splits (everything except train), in a fixed order.
# Input:   none.   Output: list of names like ["val", "test", "test_unseen"].
# Viva tip: Q: Why keep val and test separate? A: val is used for model selection and tuning;
#           test is touched only once at the end, so the test score is an unbiased estimate.
# ---------------------------------------------------------------------------------------------
def eval_splits() -> list:
    """Every labelled split except train (val, test, and test_unseen if available)."""
    # Find every .jsonl in data/splits, take its file name without extension (.stem), skip "train".
    names = sorted(p.stem for p in SPLIT_DIR.glob("*.jsonl") if p.stem != "train")
    # Desired priority: val first, then test, anything else (e.g. test_unseen) after.
    order = {"val": 0, "test": 1}
    # Sort by that priority; unknown names get 2 so they come last (ties stay alphabetical).
    return sorted(names, key=lambda n: order.get(n, 2))


# ---------------------------------------------------------------------------------------------
# compute_metrics
# Purpose: score predictions for the binary task (1 = hateful, 0 = not hateful).
# Inputs:  labels - true 0/1 labels, shape [N]; probs - predicted P(hateful), shape [N], in [0, 1];
#          threshold - probability above which we call a meme hateful (default 0.5).
# Output:  dict {"auroc": float, "acc": float, "f1": float}.
# Viva tip: Q: Why AUROC as the main metric? A: It is threshold-independent (measures how well the
#           model RANKS hateful above non-hateful memes) and is robust to class imbalance; it is also
#           the official metric of the Hateful Memes challenge. AUROC 0.5 = random, 1.0 = perfect.
# Viva tip: Q: What is F1? A: Harmonic mean of precision and recall for the hateful class; unlike
#           accuracy it is not inflated by simply predicting the majority (non-hateful) class.
# ---------------------------------------------------------------------------------------------
def compute_metrics(labels, probs, threshold: float = 0.5) -> dict:
    # Convert lists / pandas Series into numpy arrays so vectorised operations work.
    labels = np.asarray(labels)
    # Same for the predicted probabilities, shape [N].
    probs = np.asarray(probs)
    # Turn probabilities into hard 0/1 decisions: True where prob >= threshold, then cast to int.
    preds = (probs >= threshold).astype(int)
    # Return plain Python floats (not numpy types) so the dict can be saved with json.dumps.
    return {
        # ROC-AUC uses the raw probabilities (ranking), NOT the thresholded predictions.
        "auroc": float(roc_auc_score(labels, probs)),
        # Accuracy = fraction of memes where the hard prediction equals the true label.
        "acc": float(accuracy_score(labels, preds)),
        # F1 for the positive (hateful) class; zero_division=0 avoids a crash/warning when the model
        # predicts no positives at all (precision would be 0/0).
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


# ---------------------------------------------------------------------------------------------
# run_dir
# Purpose: get (and create if needed) the output folder for one model + seed.
# Inputs:  run_name (str), e.g. "bert_text"; seed (int).
# Output:  Path like results/runs/bert_text/seed0.
# Viva tip: Q: Why one folder per seed? A: So results of different seeds never overwrite each other
#           and we can later average them (mean +/- std).
# ---------------------------------------------------------------------------------------------
def run_dir(run_name: str, seed: int) -> Path:
    # Compose the folder path.
    d = RUNS_DIR / run_name / f"seed{seed}"
    # parents=True creates missing parent folders; exist_ok=True means no error if it already exists.
    d.mkdir(parents=True, exist_ok=True)
    # Give the folder back to the caller.
    return d


# ---------------------------------------------------------------------------------------------
# save_run
# Purpose: save predictions, metrics and hyper-parameters of one trained model / seed.
# Inputs:  run_name, seed; preds = {split_name: DataFrame with columns id, prob, label};
#          config = dict of hyper-parameters (lr, epochs, model name ...).
# Output:  metrics dict {split: {"auroc", "acc", "f1"}}; also writes files to disk and prints a line.
# Viva tip: Q: Why save per-meme probabilities, not just the score? A: Later steps (late fusion,
#           ensembling, ROC curves, error analysis) can then reuse them without retraining models.
# ---------------------------------------------------------------------------------------------
def save_run(run_name: str, seed: int, preds: dict, config: dict) -> dict:
    """preds: {split: DataFrame(id, prob, label)}. Writes preds_<split>.csv, metrics.json, config.json."""
    # Output folder for this model/seed (created if missing).
    d = run_dir(run_name, seed)
    # Will hold metrics for each split, e.g. {"val": {...}, "test": {...}}.
    metrics = {}
    # Loop over every evaluated split and its prediction table.
    for split, df in preds.items():
        # Save the raw per-meme predictions as CSV (index=False: do not write the row numbers).
        df.to_csv(d / f"preds_{split}.csv", index=False)
        # Compute AUROC / accuracy / F1 from the true labels and predicted probabilities.
        metrics[split] = compute_metrics(df["label"], df["prob"])
    # Save all metrics as pretty-printed JSON (indent=2 makes it human-readable).
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Save hyper-parameters for reproducibility; default=str converts non-JSON types (e.g. Path) to text.
    (d / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    # Build a one-line summary like "val: AUROC=0.7012 Acc=0.6400  test: AUROC=...".
    summary = "  ".join(f"{s}: AUROC={m['auroc']:.4f} Acc={m['acc']:.4f}" for s, m in metrics.items())
    # Print it so the training log shows the result immediately.
    print(f"[{run_name} seed{seed}] {summary}")
    # Return metrics so the calling script can also use them.
    return metrics


# ---------------------------------------------------------------------------------------------
# load_preds
# Purpose: read back the saved predictions of one model / seed / split.
# Inputs:  run_name, seed, split.   Output: DataFrame with columns id, prob, label.
# Viva tip: Q: How does late fusion use this? A: It loads the text-model and image-model probabilities
#           for the same meme ids and combines them (e.g. a weighted average tuned on val).
# ---------------------------------------------------------------------------------------------
def load_preds(run_name: str, seed: int, split: str) -> pd.DataFrame:
    # dtype id=str keeps ids as strings with leading zeros, so rows from different models join correctly.
    return pd.read_csv(RUNS_DIR / run_name / f"seed{seed}" / f"preds_{split}.csv", dtype={"id": str})


# ---------------------------------------------------------------------------------------------
# run_seeds
# Purpose: find which seeds have been run for a model.
# Input:   run_name.   Output: sorted list of ints, e.g. [0, 1, 2].
# Viva tip: Q: How do you report results over seeds? A: Mean and standard deviation of the metric
#           across these seeds.
# ---------------------------------------------------------------------------------------------
def run_seeds(run_name: str) -> list:
    # Folders are named "seed0", "seed1" ...; p.name[4:] cuts off "seed" leaving the number as text,
    # int(...) converts it, and sorted(...) returns them in increasing order.
    return sorted(int(p.name[4:]) for p in (RUNS_DIR / run_name).glob("seed*"))


# ---------------------------------------------------------------------------------------------
# load_features
# Purpose: load pre-computed (cached) frozen CLIP embeddings for one split.
# Inputs:  tag (str) - which feature set, e.g. original vs inpainted images / CLIP variant;
#          split (str) - "train", "val", "test" ...
# Output:  dict {"ids": [N], "img": [N, D], "txt": [N, D], "label": [N]} of numpy arrays,
#          where D is the CLIP embedding size (e.g. 512 or 768).
# Viva tip: Q: Why cache CLIP features? A: CLIP is frozen (its weights are not trained), so its output
#           for a meme never changes; computing it once and reusing it makes training the small fusion
#           heads very fast and cheap.
# ---------------------------------------------------------------------------------------------
def load_features(tag: str, split: str) -> dict:
    """Cached CLIP features: {ids, img, txt, label} as numpy arrays."""
    # Path like features/<tag>/train.npz (.npz = compressed bundle of several numpy arrays).
    path = FEAT_DIR / tag / f"{split}.npz"
    # Helpful error if extract_clip.py has not been run for this tag yet.
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python src/extract_clip.py --tag {tag}` first")
    # Open the .npz; allow_pickle=True is needed because the ids array holds Python string objects.
    data = np.load(path, allow_pickle=True)
    # Turn the lazy NpzFile into a normal dict {array_name: numpy array} (loads every array into memory).
    return {k: data[k] for k in data.files}

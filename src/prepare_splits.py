"""Normalize the raw jsonl files into data/splits/{train,val,test[,test_unseen]}.jsonl.

Split policy (used everywhere downstream, mention it in the report):
  * if test_seen has labels  -> train = train, val = dev_seen, test = test_seen, test_unseen = extra test
  * otherwise                -> 10% of train (stratified) = val, dev_seen = test
Model selection / threshold / fusion weights are tuned ONLY on val; test is touched once at the end.

How it works (overview):
  1. read() loads each raw jsonl file and cleans it (ids as 5-digit strings, missing text -> "").
  2. labelled() checks whether a file really has 0/1 labels (public test files sometimes do not).
  3. main() picks the split policy above, keeps only the needed columns, removes duplicate ids,
     and writes the clean splits to data/splits/, printing size and % hateful for each.
"""
# pandas: tables (DataFrames) for reading, cleaning and writing the jsonl files.
import pandas as pd
# train_test_split: sklearn helper to randomly split data, with optional stratification.
from sklearn.model_selection import train_test_split

# RAW_DIR = data/hateful_memes (downloaded files); SPLIT_DIR = data/splits (our clean output).
from common import RAW_DIR, SPLIT_DIR


# ---------------------------------------------------------------------------------------------
# read
# Purpose: load one raw jsonl file and clean it.
# Input:   name (str), e.g. "train", "dev_seen", "test_seen".
# Output:  DataFrame (id, img, text, label ...) or None if the file does not exist.
# Viva tip: Q: Why zfill(5) on ids? A: Ids like 1235 must become "01235" so they match the image
#           file names and are identical across all files when we join predictions by id.
# ---------------------------------------------------------------------------------------------
def read(name):
    # Path to the raw file, e.g. data/hateful_memes/train.jsonl.
    path = RAW_DIR / f"{name}.jsonl"
    # Some files (e.g. test_unseen) may be missing in a given download; return None instead of crashing.
    if not path.exists():
        return None
    # Read JSON Lines (one meme per line); read id as text so it is not turned into a number.
    df = pd.read_json(path, lines=True, dtype={"id": str})
    # Normalise ids: make sure they are strings and left-pad with zeros to 5 characters.
    df["id"] = df["id"].astype(str).str.zfill(5)
    # Replace missing captions with "" and force string type, so the tokenizer never receives NaN.
    df["text"] = df["text"].fillna("").astype(str)
    # Return the cleaned table.
    return df


# ---------------------------------------------------------------------------------------------
# labelled
# Purpose: check whether a split has usable ground-truth labels.
# Input:   df (DataFrame or None).
# Output:  True if df exists, has a "label" column and every label is 0 or 1; otherwise False.
# Viva tip: Q: Why is this check needed? A: In some releases the test files have no labels (or -1
#           placeholders); we must not evaluate on fake labels, so we fall back to another policy.
# ---------------------------------------------------------------------------------------------
def labelled(df):
    # Short-circuit "and": stops at the first False, so df["label"] is only touched if it exists.
    return df is not None and "label" in df and df["label"].isin([0, 1]).all()


# ---------------------------------------------------------------------------------------------
# main
# Purpose: build the final train / val / test (and optional test_unseen) splits and save them.
# Inputs:  raw jsonl files in data/hateful_memes/.   Output: jsonl files in data/splits/.
# Viva tip: Q: Why a stratified split? A: Stratify keeps the same hateful/non-hateful ratio in train
#           and val (~35-37% hateful); with a purely random split val could be unbalanced by chance.
# Viva tip: Q: Why tune only on val? A: If we tuned on test, the test score would be optimistic
#           (information leakage); test must stay unseen until the final evaluation.
# ---------------------------------------------------------------------------------------------
def main():
    # Load the training set and the official "seen" development set.
    train, dev_seen = read("train"), read("dev_seen")
    # Load the official test sets (they may be missing or unlabelled).
    test_seen, test_unseen = read("test_seen"), read("test_unseen")
    # train and dev_seen are essential; stop with a clear message if they are missing.
    if train is None or dev_seen is None:
        raise SystemExit(f"train.jsonl / dev_seen.jsonl not found in {RAW_DIR} - run download_data.py")

    # Policy A: the official test set has real labels -> use the official splits as they are.
    if labelled(test_seen):
        # dev_seen is our validation set (for tuning); test_seen is the held-out test set.
        splits = {"train": train, "val": dev_seen, "test": test_seen}
        # If test_unseen is also labelled, keep it as an extra, harder test set (new unseen memes).
        if labelled(test_unseen):
            splits["test_unseen"] = test_unseen
    # Policy B: no labelled test -> carve our own validation set out of train.
    else:
        # 90% train / 10% val; stratify=label keeps the class ratio equal; random_state=42 makes the
        # split identical every time the script runs (reproducible).
        tr, va = train_test_split(train, test_size=0.1, stratify=train["label"], random_state=42)
        # dev_seen (labelled) becomes the test set in this policy.
        splits = {"train": tr, "val": va, "test": dev_seen}

    # Create data/splits/ if needed.
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    # Only these four columns are needed downstream: meme id, image path, caption text, label.
    cols = ["id", "img", "text", "label"]
    # Process and save each split.
    for name, df in splits.items():
        # Keep the needed columns, drop rows with a repeated id (the same meme listed twice),
        # and renumber rows 0..N-1 (drop=True discards the old index instead of adding it as a column).
        df = df[cols].drop_duplicates("id").reset_index(drop=True)
        # Make sure labels are plain integers 0/1 (they might be read as float).
        df["label"] = df["label"].astype(int)
        # Write as JSON Lines: orient="records" + lines=True -> one {"id":..,"img":..,...} per line.
        df.to_json(SPLIT_DIR / f"{name}.jsonl", orient="records", lines=True)
        # Print split size and fraction of hateful memes (mean of 0/1 labels = positive rate).
        print(f"{name:12s} n={len(df):5d}  hateful={df['label'].mean():.1%}")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()

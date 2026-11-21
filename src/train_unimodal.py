"""End-to-end fine-tuned baselines (paper models M1-M3):

  --modality text   M1  BERT text-only
  --modality image  M2  ResNet-50 image-only
  --modality both   M3  early fusion: [BERT CLS ; ResNet pooled] -> MLP, trained jointly

Usage:
  python src/train_unimodal.py --modality text  --seeds 0 1 2
  python src/train_unimodal.py --modality image --seeds 0 1 2
  python src/train_unimodal.py --modality both  --seeds 0 1 2
Outputs: results/runs/<run_name>/seed<k>/{preds_*.csv, metrics.json, config.json}
Needs a GPU (Colab / Kaggle T4 is enough: ~5-15 min per seed).

How it works (overview):
  1. Data: each meme is (image, overlaid text, label 0/1). A Dataset reads one meme; a Collator turns a
     list of memes into a batch (tokenizes text with dynamic padding, stacks images into one tensor).
  2. Encoders (pretrained = transfer learning):
       - Text:  BERT-base. We take the [CLS] token vector of the last layer -> [B, 768].
       - Image: ResNet-50 (ImageNet weights) with its final fc layer removed -> pooled vector [B, 2048].
  3. Head: a small MLP (Dropout -> Linear -> ReLU -> Dropout -> Linear) gives ONE logit per meme -> [B].
     For M3 (early fusion) we concatenate text and image features -> [B, 768 + 2048 = 2816] first.
  4. Training: everything (backbones + head) is fine-tuned end to end with AdamW, differential learning
     rates (tiny for pretrained backbones, larger for the new head), linear warmup + linear decay,
     mixed precision (autocast + GradScaler), gradient clipping and BCEWithLogitsLoss.
  5. Model selection: after every epoch we compute AUROC on the validation set and keep the best
     weights (a form of early stopping). The test set is never used for any decision.
  6. The best model predicts probabilities for every eval split; these are saved and later reused
     (for example by late_fusion.py, which averages the text and image probabilities).
  7. Repeated for several random seeds so we can report mean +/- std, not one lucky run.
"""
# argparse: read command-line options such as --modality and --seeds.
import argparse
# copy: used for deepcopy so the saved "best" weights are not changed by later training steps.
import copy
# Path: OS-independent file paths (works on Windows, Linux, Colab).
from pathlib import Path

# pandas: stores predictions as tables (id, prob, label) that are easy to save as CSV.
import pandas as pd
# torch: the deep learning framework (tensors, autograd, GPU).
import torch
# nn: neural-network building blocks (layers, loss functions).
import torch.nn as nn
# PIL Image: loads the meme image files from disk.
from PIL import Image
# Dataset = how to get ONE sample; DataLoader = batching, shuffling and parallel loading.
from torch.utils.data import DataLoader, Dataset
# torchvision: pretrained CNNs (ResNet-50) and image transforms (augmentation, normalization).
from torchvision import models, transforms
# tqdm: progress bar for the training loop.
from tqdm import tqdm
# Hugging Face: AutoModel loads pretrained BERT, AutoTokenizer converts text into token ids.
from transformers import AutoModel, AutoTokenizer

# Shared helpers from our own common.py: data folder, metric computation (AUROC/accuracy), list of
# evaluation splits, device selection (GPU/CPU), split loading, run folders, saving results, seeding.
from common import (RAW_DIR, compute_metrics, eval_splits, get_device, load_split, run_dir, save_run,
                    set_seed)

# ImageNet per-channel (R, G, B) mean and standard deviation. ResNet-50 was pretrained on images
# normalized with exactly these numbers, so we must normalize our memes the same way; otherwise the
# input distribution would differ from what the pretrained weights expect.
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Training transforms = data augmentation. Each epoch the model sees a slightly different version of
# each image, which acts as a regularizer and reduces overfitting (the dataset is small, ~8.5k memes).
TRAIN_TF = transforms.Compose([
    # Crop a random region covering 80-100% of the image and resize it to 224x224 (ResNet input size).
    # The scale is kept high (0.8) so the overlaid meme text / key objects are not cropped away.
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # Mirror the image left-right with probability 0.5; meaning of a meme image is usually unchanged.
    transforms.RandomHorizontalFlip(),
    # Randomly change brightness, contrast and saturation by up to 20% -> robust to colour/lighting.
    transforms.ColorJitter(0.2, 0.2, 0.2),
    # ToTensor: PIL image (H, W, 3) in 0-255 -> float tensor [3, 224, 224] in 0-1.
    # Normalize: (x - mean) / std per channel, using the ImageNet statistics above.
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
# Evaluation transforms: deterministic (no randomness), so val/test scores are reproducible and
# measure the model, not random augmentation. Augmentation is only useful while learning.
EVAL_TF = transforms.Compose([
    # Resize the shorter side to 256, then take the central 224x224 crop (standard ImageNet eval).
    transforms.Resize(256), transforms.CenterCrop(224),
    # Same tensor conversion and ImageNet normalization as in training (must match!).
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


# ----------------------------------------------------------------------------------------------------
# MemeDataset
# Purpose : PyTorch Dataset that returns ONE meme at a time as a dict.
# Inputs  : df (pandas table with columns id, img, text, label), modality ("text"/"image"/"both"),
#           img_root (folder containing the images), train (True -> use augmentation).
# Outputs : __getitem__ returns {"id", "text", "label"} and, if images are needed, "pixels" [3,224,224].
# Viva tip: Q: Why a Dataset class? A: It separates "how to read one sample" from batching; the
#           DataLoader then handles batching, shuffling and parallel workers for us.
# Viva tip: Q: Why is text not tokenized here? A: Tokenizing per batch in the collator lets us pad only
#           to the longest text in that batch (dynamic padding), which is faster than fixed padding.
# ----------------------------------------------------------------------------------------------------
class MemeDataset(Dataset):
    def __init__(self, df, modality, img_root, train):
        # Reset the index to 0..N-1 so that iloc[i] positions match the indices the DataLoader asks for.
        self.df = df.reset_index(drop=True)
        # Remember which inputs are needed; image loading is skipped for the text-only model (faster).
        self.modality = modality
        # Root folder of the raw images (img column holds a relative path like "img/12345.png").
        self.img_root = img_root
        # Augmentation only for training; deterministic transforms for val/test.
        self.tf = TRAIN_TF if train else EVAL_TF

    def __len__(self):
        # Number of samples; the DataLoader uses it to know how many batches an epoch has.
        return len(self.df)

    def __getitem__(self, i):
        # Get row i of the table (one meme).
        r = self.df.iloc[i]
        # Basic fields. Label is converted to float because BCEWithLogitsLoss expects float targets.
        item = {"id": r["id"], "text": r["text"], "label": float(r["label"])}
        # Only load the image when the model actually uses it (image-only or early fusion).
        if self.modality in ("image", "both"):
            # Open image, force 3 RGB channels (some files are RGBA/grayscale), then apply transforms
            # -> normalized tensor of shape [3, 224, 224].
            item["pixels"] = self.tf(Image.open(self.img_root / r["img"]).convert("RGB"))
        # Return a dict; the Collator combines many such dicts into one batch.
        return item


# ----------------------------------------------------------------------------------------------------
# Collator (used as DataLoader collate_fn)
# Purpose : turn a list of B sample-dicts into one batch-dict of tensors.
# Inputs  : tokenizer (or None for image-only), max_len (max number of tokens per text).
# Outputs : {"id": list[B], "label": [B], "text": {input_ids [B,L], attention_mask [B,L], ...},
#            "pixels": [B,3,224,224]}  (text / pixels only when the modality needs them).
# Viva tip: Q: What is the attention mask? A: 1 for real tokens, 0 for padding, so BERT ignores the
#           padded positions in self-attention.
# Viva tip: Q: What is dynamic padding? A: Pad each batch only to its own longest text (padding=True)
#           instead of always to max_len; less wasted computation.
# ----------------------------------------------------------------------------------------------------
class Collator:
    def __init__(self, tokenizer, max_len):
        # Store the tokenizer and the truncation length (default 64 tokens; meme texts are short).
        self.tok, self.max_len = tokenizer, max_len

    def __call__(self, batch):
        # ids stay a plain Python list (strings/ints, not needed on GPU); labels -> float tensor [B].
        out = {"id": [b["id"] for b in batch],
               "label": torch.tensor([b["label"] for b in batch])}
        # Text models: tokenize all B texts together.
        if self.tok is not None:
            # padding=True    -> pad to the longest text in THIS batch (dynamic padding).
            # truncation=True -> cut texts longer than max_length tokens.
            # return_tensors="pt" -> PyTorch tensors. Result holds input_ids, token_type_ids and
            # attention_mask, each of shape [B, L] where L <= max_len. [CLS] and [SEP] are added.
            out["text"] = self.tok([b["text"] for b in batch], padding=True, truncation=True,
                                   max_length=self.max_len, return_tensors="pt")
        # Image models: stack B tensors of [3,224,224] into one tensor [B,3,224,224].
        if "pixels" in batch[0]:
            out["pixels"] = torch.stack([b["pixels"] for b in batch])
        # One batch, ready to be moved to the GPU.
        return out


# ----------------------------------------------------------------------------------------------------
# BaselineModel
# Purpose : one class for all three baselines. M1 = BERT + head, M2 = ResNet-50 + head,
#           M3 = early fusion (concatenate BERT [CLS] and ResNet pooled features, then head).
# Inputs  : modality ("text"/"image"/"both"), text_model (HF name, e.g. "bert-base-uncased"),
#           dropout probability for the head.
# Outputs : forward(batch) returns raw logits of shape [B] (no sigmoid; the loss applies it).
# Viva tip: Q: Early vs late fusion? A: Early fusion joins the FEATURES of both modalities and trains
#           one classifier on them, so it can learn interactions; late fusion only combines the final
#           probabilities of two separately trained models.
# Viva tip: Q: Why replace ResNet's fc with Identity? A: The original fc maps 2048 features to the 1000
#           ImageNet classes; we remove it to get the 2048-d pooled feature vector and add our own head.
# ----------------------------------------------------------------------------------------------------
class BaselineModel(nn.Module):
    def __init__(self, modality, text_model, dropout=0.2):
        # Initialize nn.Module so sub-layers and parameters are registered correctly.
        super().__init__()
        # Save modality; forward() uses it to decide which encoders to run.
        self.modality = modality
        # dim will hold the total size of the feature vector given to the head.
        dim = 0
        # Text encoder for M1 and M3.
        if modality in ("text", "both"):
            # Load pretrained BERT (transfer learning: it already understands English).
            self.bert = AutoModel.from_pretrained(text_model)
            # hidden_size = 768 for bert-base -> the [CLS] vector has 768 numbers.
            dim += self.bert.config.hidden_size
        # Image encoder for M2 and M3.
        if modality in ("image", "both"):
            # ResNet-50 with ImageNet pretrained weights (IMAGENET1K_V2 = the improved torchvision recipe).
            self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Remove the 1000-class ImageNet classifier; the network now outputs the global-average
            # pooled feature vector [B, 2048].
            self.cnn.fc = nn.Identity()
            # ResNet-50 pooled features are 2048-dimensional.
            dim += 2048
        # Hidden layer width: 512 for the fused model (compress 2816 -> 512); for single modality keep
        # the same width as the input (768 or 2048).
        hidden = 512 if modality == "both" else dim
        # Classification head (new, randomly initialized). Dropout fights overfitting, ReLU adds
        # non-linearity, the last Linear outputs ONE logit (binary task: hateful vs not).
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, batch):
        # Collect feature vectors from each active modality.
        feats = []
        if self.modality in ("text", "both"):
            # Run BERT on input_ids/attention_mask/token_type_ids. last_hidden_state is [B, L, 768];
            # [:, 0] takes position 0 = the [CLS] token, used as a summary of the whole sentence -> [B,768].
            feats.append(self.bert(**batch["text"]).last_hidden_state[:, 0])  # [CLS]
        if self.modality in ("image", "both"):
            # ResNet on pixels [B,3,224,224] -> pooled features [B, 2048] (fc is Identity).
            feats.append(self.cnn(batch["pixels"]))
        # Concatenate along the feature dimension (early fusion when both present): [B, 768+2048].
        # Head gives [B, 1]; squeeze(1) removes the last axis -> logits [B] to match labels [B].
        return self.head(torch.cat(feats, dim=1)).squeeze(1)

    # ------------------------------------------------------------------------------------------------
    # param_groups
    # Purpose : build optimizer parameter groups with DIFFERENT learning rates per part of the model.
    # Inputs  : lr_text (BERT, default 2e-5), lr_image (ResNet, 1e-4), lr_head (new MLP, 1e-3).
    # Outputs : list of dicts {"params", "lr"} that AdamW accepts.
    # Viva tip: Q: Why differential learning rates? A: Pretrained backbones already have good weights,
    #           so small steps avoid destroying them (catastrophic forgetting); the head starts random
    #           and needs larger steps to learn quickly.
    # ------------------------------------------------------------------------------------------------
    def param_groups(self, lr_text, lr_image, lr_head):
        # The head always exists and gets the largest learning rate.
        groups = [{"params": self.head.parameters(), "lr": lr_head}]
        # Add BERT parameters with a small LR only if this model has a BERT encoder.
        if hasattr(self, "bert"):
            groups.append({"params": self.bert.parameters(), "lr": lr_text})
        # Add ResNet parameters with their own (medium) LR only if this model has a CNN.
        if hasattr(self, "cnn"):
            groups.append({"params": self.cnn.parameters(), "lr": lr_image})
        return groups


# ----------------------------------------------------------------------------------------------------
# to_device
# Purpose : move all tensors of a batch to the GPU/CPU device used by the model.
# Inputs  : batch dict from the Collator, device.
# Outputs : a new dict with label, text tensors and pixels on the device (ids stay as a Python list).
# Viva tip: Q: Why move data to the device? A: Model and inputs must be on the same device; otherwise
#           PyTorch raises an error.
# ----------------------------------------------------------------------------------------------------
def to_device(batch, device):
    # Shallow copy so we do not modify the original dict.
    out = dict(batch)
    # Labels [B] to device (needed by the loss).
    out["label"] = batch["label"].to(device)
    # Tokenizer output is a dict of tensors (input_ids, attention_mask, ...): move each one.
    if "text" in batch:
        out["text"] = {k: v.to(device) for k, v in batch["text"].items()}
    # Image tensor [B,3,224,224] to device.
    if "pixels" in batch:
        out["pixels"] = batch["pixels"].to(device)
    return out


# ----------------------------------------------------------------------------------------------------
# predict
# Purpose : run the model on a whole split and return predicted probabilities.
# Inputs  : model, DataLoader of a split (val/test), device.
# Outputs : DataFrame with columns id, prob (P(hateful) in 0-1), label (0/1).
# Viva tip: Q: What do model.eval() and torch.no_grad() do? A: eval() turns off dropout (and uses
#           running stats in BatchNorm) so predictions are deterministic; no_grad() stops storing the
#           computation graph, saving memory and time because we do not backpropagate here.
# ----------------------------------------------------------------------------------------------------
# Decorator: disable gradient tracking for everything inside predict().
@torch.no_grad()
def predict(model, loader, device):
    # Evaluation mode: dropout off, BatchNorm uses its stored running mean/variance.
    model.eval()
    # Accumulate results over all batches.
    ids, probs, labels = [], [], []
    for batch in loader:
        # Move inputs to GPU/CPU.
        batch = to_device(batch, device)
        # Mixed precision on GPU (faster inference, same as training); disabled on CPU.
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            # Raw logits [B].
            logits = model(batch)
        # Keep meme ids in the same order as the predictions.
        ids += batch["id"]
        # Sigmoid turns logits into probabilities in (0,1); .float() converts from fp16 for precision;
        # .cpu().tolist() moves them to normal Python numbers.
        probs += torch.sigmoid(logits.float()).cpu().tolist()
        # True labels as ints 0/1 (needed later for AUROC).
        labels += batch["label"].int().cpu().tolist()
    # One row per meme; saved as CSV and reused by late fusion and the analysis scripts.
    return pd.DataFrame({"id": ids, "prob": probs, "label": labels})


# ----------------------------------------------------------------------------------------------------
# train_one
# Purpose : full training run for ONE random seed: build data, model, optimizer, scheduler; train for
#           args.epochs; keep the epoch with the best validation AUROC; save predictions and metrics.
# Inputs  : args (command-line options), seed (int), device.
# Outputs : nothing returned; files written to results/runs/<run_name>/seed<k>/.
# Viva tip: Q: Why select the model on validation and not test? A: Test must stay unseen to give an
#           unbiased estimate of real performance; choosing on test would leak information (overfit to test).
# Viva tip: Q: Why AUROC? A: The classes are imbalanced and AUROC measures ranking quality over all
#           thresholds, so it does not depend on choosing a 0.5 cut-off.
# Viva tip: Q: Why mixed precision + GradScaler? A: fp16 is faster and uses less GPU memory; the scaler
#           multiplies the loss so small gradients do not underflow to zero in fp16.
# ----------------------------------------------------------------------------------------------------
def train_one(args, seed, device):
    # Fix random seeds (Python, NumPy, PyTorch) so weight init, shuffling and dropout are reproducible.
    set_seed(seed)
    # Load the BERT tokenizer only when text is used (WordPiece, lowercase for bert-base-uncased).
    tok = AutoTokenizer.from_pretrained(args.text_model) if args.modality in ("text", "both") else None
    # One collator shared by all loaders (tokenizer=None for image-only).
    collate = Collator(tok, args.max_len)

    # Small helper that builds a DataLoader for a given split name.
    def loader(split, train):
        # Dataset for this split; train=True switches on augmentation.
        ds = MemeDataset(load_split(split), args.modality, args.img_root, train)
        # shuffle only for training (random batch order each epoch helps SGD; eval order does not matter).
        # num_workers loads images in parallel processes; pin_memory speeds up CPU->GPU copies.
        return DataLoader(ds, batch_size=args.batch, shuffle=train, num_workers=args.workers,
                          collate_fn=collate, pin_memory=True)

    # Training loader (augmented + shuffled).
    train_dl = loader("train", True)
    # One loader per evaluation split (e.g. val, test), no augmentation, no shuffle.
    eval_dls = {s: loader(s, False) for s in eval_splits()}

    # Build the model and move its weights to the GPU.
    model = BaselineModel(args.modality, args.text_model).to(device)
    # AdamW = Adam with DECOUPLED weight decay (L2-style regularization applied directly to weights,
    # not mixed into the adaptive gradient). weight_decay=0.01 is the standard value for BERT fine-tuning.
    opt = torch.optim.AdamW(model.param_groups(args.lr_text, args.lr_image, args.lr_head), weight_decay=0.01)
    # Total number of optimizer steps = epochs * batches per epoch.
    total = args.epochs * len(train_dl)
    # First 10% of steps are warmup (at least 1 step to avoid division by zero).
    warmup = max(1, int(0.1 * total))  # linear warmup, then linear decay to 0
    # LambdaLR multiplies each group's base LR by a factor that depends on the step:
    #   during warmup: factor = (step+1)/warmup, rising from ~0 to 1 -> avoids large, unstable updates
    #   at the start when the head is random and Adam's statistics are not yet reliable;
    #   after warmup: factor = (total-step)/(total-warmup), falling linearly to 0 -> smaller steps
    #   near the end help the model settle into a good minimum. min() picks the active phase.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min((step + 1) / warmup, max(0.0, (total - step) / max(1, total - warmup))))
    # GradScaler for mixed precision: scales the loss up before backward so fp16 gradients do not
    # underflow; disabled on CPU (no autocast there).
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in ONE numerically stable op (log-sum-exp trick).
    # Separate sigmoid then BCE can produce log(0) = -inf for large logits.
    loss_fn = nn.BCEWithLogitsLoss()

    # Track the best validation AUROC and a copy of the weights at that epoch (early stopping style).
    best_auc, best_state = -1, None
    for epoch in range(args.epochs):
        # Training mode: dropout active, BatchNorm updates its statistics.
        model.train()
        # Progress bar over the training batches.
        pbar = tqdm(train_dl, desc=f"seed{seed} ep{epoch + 1}/{args.epochs}", leave=False)
        for batch in pbar:
            # Move the batch to the GPU.
            batch = to_device(batch, device)
            # Forward pass in mixed precision (fp16 where safe, fp32 where needed).
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                # logits [B] cast to float32 for a stable loss; labels [B] are floats 0/1.
                loss = loss_fn(model(batch).float(), batch["label"])
            # Clear old gradients (set_to_none is slightly faster and uses less memory than zeroing).
            opt.zero_grad(set_to_none=True)
            # Backpropagation on the scaled loss -> gradients for every trainable parameter.
            scaler.scale(loss).backward()
            # Divide gradients back by the scale factor so clipping sees the TRUE gradient values.
            scaler.unscale_(opt)
            # Gradient clipping: if the total gradient norm exceeds 1.0, rescale it to 1.0.
            # Prevents exploding gradients / sudden huge updates that can ruin pretrained weights.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Optimizer step (skipped automatically if gradients contain inf/NaN from fp16 overflow).
            scaler.step(opt)
            # Adjust the scale factor for the next iteration (lower it after overflow, raise it slowly).
            scaler.update()
            # Advance the learning-rate schedule by one step (scheduler is per batch, not per epoch).
            sched.step()
            # Show the current batch loss on the progress bar.
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        # End of epoch: predict on the validation set.
        val_df = predict(model, eval_dls["val"], device)
        # Compute AUROC and accuracy on validation.
        val = compute_metrics(val_df["label"], val_df["prob"])
        print(f"seed{seed} epoch {epoch + 1}: val AUROC={val['auroc']:.4f} acc={val['acc']:.4f}")
        # Keep this epoch if it is the best so far on VALIDATION AUROC (never on test).
        if val["auroc"] > best_auc:  # model selection on val only
            best_auc = val["auroc"]
            # Copy weights to CPU and deepcopy them, so later training steps do not overwrite the
            # snapshot and GPU memory is not doubled.
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    # Restore the best epoch's weights (the final epoch may have overfitted).
    model.load_state_dict(best_state)
    # Predict probabilities on every eval split with the best model.
    preds = {s: predict(model, dl, device) for s, dl in eval_dls.items()}
    # Save predictions CSVs, metrics and the config (all args + seed + best val AUROC) for this seed.
    save_run(args.run_name, seed, preds, vars(args) | {"seed": seed, "best_val_auroc": best_auc})
    # Optionally save the weights too (large file, so off by default).
    if args.save_model:
        torch.save(best_state, run_dir(args.run_name, seed) / "model.pt")


# ----------------------------------------------------------------------------------------------------
# main
# Purpose : parse command-line options, fill in per-modality defaults, and train one model per seed.
# Inputs  : command-line arguments (see Usage in the module docstring).
# Outputs : nothing returned; one results folder per seed.
# Viva tip: Q: Why several seeds? A: Results vary with random init, shuffling and dropout; averaging
#           over seeds (mean +/- std) gives a more reliable comparison between models.
# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Which baseline to train: text (M1), image (M2) or both (M3 early fusion).
    ap.add_argument("--modality", choices=["text", "image", "both"], required=True)
    # List of random seeds; one full training run per seed.
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # Pretrained text encoder name from Hugging Face.
    ap.add_argument("--text_model", default="bert-base-uncased")
    # Number of epochs; if not given, a modality-specific default is used below.
    ap.add_argument("--epochs", type=int, default=None, help="default: text 4, image 6, both 5")
    # Batch size (32 fits on a T4 GPU with mixed precision).
    ap.add_argument("--batch", type=int, default=32)
    # Max tokens per meme text; meme captions are short, so 64 keeps almost all text and is fast.
    ap.add_argument("--max_len", type=int, default=64)
    # Learning rate for BERT: very small (2e-5 is the standard BERT fine-tuning value).
    ap.add_argument("--lr_text", type=float, default=2e-5)
    # Learning rate for ResNet-50: small but larger than BERT's.
    ap.add_argument("--lr_image", type=float, default=1e-4)
    # Learning rate for the new randomly initialized head: largest.
    ap.add_argument("--lr_head", type=float, default=1e-3)
    # Number of background processes that load and augment images.
    ap.add_argument("--workers", type=int, default=2)
    # Folder containing the meme images.
    ap.add_argument("--img_root", default=str(RAW_DIR))
    # Name of the results folder; defaults to the paper model name below.
    ap.add_argument("--run_name", default=None)
    # Flag: also save model weights (not needed for the analysis, which only uses predictions).
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    # Convert the image root string into a Path object (so we can use "/" to join paths).
    args.img_root = Path(args.img_root)
    # Default epochs per modality (images need a few more epochs; BERT converges fast and overfits).
    args.epochs = args.epochs or {"text": 4, "image": 6, "both": 5}[args.modality]
    # Default run names that match the model names used in the report (M1, M2, M3).
    args.run_name = args.run_name or {"text": "M1_bert_text", "image": "M2_resnet_image",
                                      "both": "M3_early_fusion_bert_resnet"}[args.modality]
    # GPU if available, otherwise CPU.
    device = get_device()
    print(f"device={device}  run={args.run_name}")
    # Train one independent model per seed.
    for seed in args.seeds:
        train_one(args, seed, device)


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()

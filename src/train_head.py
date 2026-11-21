"""Train a fusion head (fusion_heads.py) on cached CLIP features.  Models M6/M7 + the fusion ablation.

Usage:
  python src/train_head.py --fusion concat                      # M6: CLIP + MLP head
  python src/train_head.py --fusion cross                       # M7: Hate-CLIPper style cross fusion
  python src/train_head.py --fusion image_only --features clip_l14_inpainted   # true image-only (angle D)
  python src/train_head.py --fusion concat --modality_dropout 0.3              # extra ablation
Run name defaults to C_<features>_<fusion>. Saves model.pt per seed (small) for modality_ablation.py.

How it works:
  1. Load the cached CLIP features (from extract_clip.py) for train and the eval splits as tensors.
  2. Build a FusionHead for the chosen fusion, optimizer AdamW (Adam with decoupled weight decay) and a
     cosine learning-rate schedule. Loss = BCEWithLogitsLoss (binary cross-entropy on 1 logit).
  3. Each epoch: shuffle the training set with randperm, train in mini-batches (optionally with modality
     dropout), then compute AUROC on the validation set.
  4. Early stopping: keep a copy of the weights with the best val AUROC; stop if val AUROC has not
     improved for `patience` epochs.
  5. Reload the best weights, predict probabilities on every eval split, save predictions + config
     + model.pt. Repeat for 3 seeds (0,1,2) so results can be reported as mean +- std.
"""
# argparse: command-line options (fusion type, hyper-parameters, seeds).
import argparse
# copy.deepcopy: take an independent snapshot of the best model weights.
import copy

# pandas: build prediction tables (id, prob, label) that save_run writes to disk.
import pandas as pd
# torch: tensors, optimizers, schedulers.
import torch
# nn: loss function BCEWithLogitsLoss.
import torch.nn as nn

# Project helpers: metrics (AUROC etc.), eval split names, device, cached feature loader, output folder,
# saving predictions/config, and seeding of all random generators for reproducibility.
from common import compute_metrics, eval_splits, get_device, load_features, run_dir, save_run, set_seed
# FUSIONS = allowed fusion names; FusionHead = the model.
from fusion_heads import FUSIONS, FusionHead


# ---------------------------------------------------------------------------------------------------
# to_tensors
# Purpose : Convert one split's numpy feature arrays into torch tensors on the chosen device.
# Inputs  : feats = dict-like npz with "img" [N,768], "txt" [N,768], "label" [N]; device.
# Outputs : (img [N,768], txt [N,768], label [N] float32).
# Viva tip: Q: Why put the whole dataset on the GPU at once? A: The features are tiny
#           (~8.5k x 768 floats ~ 26 MB per modality), so no DataLoader is needed and training is fast.
# Viva tip: Q: Why labels as float32? A: BCEWithLogitsLoss expects float targets (0.0 / 1.0).
# ---------------------------------------------------------------------------------------------------
def to_tensors(feats, device):
    # Build the three tensors directly on the device.
    return (torch.tensor(feats["img"], device=device), torch.tensor(feats["txt"], device=device),
            torch.tensor(feats["label"], dtype=torch.float32, device=device))


# ---------------------------------------------------------------------------------------------------
# predict_probs
# Purpose : Get P(hateful) for every sample, in chunks, without gradients.
# Inputs  : model, img [N,768], txt [N,768], batch = chunk size (1024).
# Outputs : numpy array of probabilities [N].
# Viva tip: Q: Why model.eval()? A: It switches off Dropout so predictions are deterministic and use
#           the full network.
# ---------------------------------------------------------------------------------------------------
# No gradient tracking needed for inference: saves memory and time.
@torch.no_grad()
def predict_probs(model, img, txt, batch=1024):
    # Evaluation mode: Dropout disabled (LayerNorm behaves the same in both modes).
    model.eval()
    # For each chunk start s, run the model on rows s..s+batch and turn logits into probabilities via sigmoid.
    out = [torch.sigmoid(model(img[s:s + batch], txt[s:s + batch])) for s in range(0, len(img), batch)]
    # Join chunks into one [N] tensor, move to CPU, convert to numpy for sklearn metrics.
    return torch.cat(out).cpu().numpy()


# ---------------------------------------------------------------------------------------------------
# train_one
# Purpose : Train one FusionHead with one random seed, with early stopping on val AUROC, and save it.
# Inputs  : args (parsed CLI options), seed (int), device.
# Outputs : nothing returned; writes predictions, config and model.pt into the run folder.
# Viva tip: Q: Why early stopping on AUROC rather than loss? A: AUROC is the official Hateful Memes
#           metric and is threshold-free; we select the model on exactly what we report.
# Viva tip: Q: Why 3 seeds? A: Small dataset -> results vary with initialization/shuffling; averaging
#           over seeds shows whether differences between fusions are real or just noise.
# Viva tip: Q: What is modality dropout? A: During training we randomly zero the image OR the text for
#           some samples, so the head cannot rely on one modality only and must use both.
# ---------------------------------------------------------------------------------------------------
def train_one(args, seed, device):
    # Fix Python/numpy/torch random seeds -> reproducible initialization and shuffling.
    set_seed(seed)
    # Training features as tensors: tr_img [N,768], tr_txt [N,768], tr_y [N].
    tr_img, tr_txt, tr_y = to_tensors(load_features(args.features, "train"), device)
    # Raw numpy features of every eval split (we keep "ids" from here for the prediction files).
    evals = {s: load_features(args.features, s) for s in eval_splits()}
    # Same eval splits converted to tensors: {split: (img, txt, y)}.
    ev_t = {s: to_tensors(f, device) for s, f in evals.items()}

    # Build the fusion head; in_dim = feature size taken from the data (768 for ViT-L/14, 512 for B/32).
    model = FusionHead(args.fusion, tr_img.shape[1], proj_dim=args.proj_dim, cross_dim=args.cross_dim,
                       hidden=args.hidden, dropout=args.dropout).to(device)
    # AdamW = Adam with DECOUPLED weight decay: weights are shrunk directly (w -= lr*wd*w) instead of
    # adding an L2 term to the gradient, which works better with Adam's adaptive step sizes.
    # Weight decay is a regularizer that keeps weights small -> less overfitting.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Cosine annealing: LR follows half a cosine from lr down to ~0 over T_max = epochs. Big steps early
    # (fast progress), small steps late (fine convergence).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # Binary cross-entropy with built-in sigmoid: loss = -[y log s(z) + (1-y) log(1 - s(z))], computed stably.
    loss_fn = nn.BCEWithLogitsLoss()

    # best_auc starts at -1 so the first epoch always counts as an improvement; best_state = best weights;
    # bad = number of consecutive epochs without improvement (for early stopping).
    best_auc, best_state, bad = -1, None, 0
    # Number of training samples.
    n = len(tr_y)
    # Epoch loop (max args.epochs, may stop earlier).
    for epoch in range(args.epochs):
        # Training mode: Dropout active.
        model.train()
        # randperm(n) = random permutation of 0..n-1 (shuffle every epoch); split(batch) cuts it into
        # mini-batches of indices, each of size args.batch (last one may be smaller).
        for idx in torch.randperm(n, device=device).split(args.batch):
            # Select this mini-batch: img [B,768], txt [B,768].
            img, txt = tr_img[idx], tr_txt[idx]
            if args.modality_dropout > 0:  # randomly blank one modality so the head can't over-rely on it
                # drop[b] = True with probability modality_dropout -> this sample gets one modality removed.
                drop = torch.rand(len(idx), device=device) < args.modality_dropout
                # which[b] = coin flip: True -> remove the image, False -> remove the text.
                which = torch.rand(len(idx), device=device) < 0.5
                # Mask ~(drop & which) is 0 where we drop the image, 1 elsewhere; unsqueeze(1) -> [B,1] so it
                # broadcasts over the 768 features. (F.normalize keeps a zero vector as zero.)
                img = img * ~(drop & which).unsqueeze(1)
                # Same for text: zeroed where drop is True and which is False. Never both at once.
                txt = txt * ~(drop & ~which).unsqueeze(1)
            # Forward pass -> logits [B]; compare with labels [B] -> scalar mean loss.
            loss = loss_fn(model(img, txt), tr_y[idx])
            # Clear old gradients (set_to_none is slightly faster/leaner than filling with zeros).
            opt.zero_grad(set_to_none=True)
            # Backpropagation: compute d(loss)/d(parameter) for all trainable parameters.
            loss.backward()
            # Update parameters with AdamW.
            opt.step()
        # Advance the cosine schedule once per epoch (T_max is measured in epochs).
        sched.step()

        # Validation tensors.
        v_img, v_txt, v_y = ev_t["val"]
        # Val AUROC = probability that a random hateful meme scores higher than a random non-hateful one.
        val_auc = compute_metrics(v_y.cpu().numpy(), predict_probs(model, v_img, v_txt))["auroc"]
        # Improvement: remember the score, deep-copy the weights (a plain reference would keep changing
        # as training continues), and reset the patience counter.
        if val_auc > best_auc:
            best_auc, best_state, bad = val_auc, copy.deepcopy(model.state_dict()), 0
        # No improvement: count it, and stop training once patience is used up (avoids overfitting).
        else:
            bad += 1
            if bad >= args.patience:
                break

    # Restore the best-on-validation weights (not the last epoch's weights).
    model.load_state_dict(best_state)
    # Will hold one prediction table per eval split.
    preds = {}
    # For every eval split, store id, predicted probability, and true label (int 0/1).
    for s, (img, txt, y) in ev_t.items():
        preds[s] = pd.DataFrame({"id": evals[s]["ids"].astype(str), "prob": predict_probs(model, img, txt),
                                 "label": y.cpu().numpy().astype(int)})
    # Config to save: all CLI args merged (dict | dict, Python 3.9+) with seed, input dim, best val AUROC,
    # the epoch where training stopped, and the number of trainable parameters (model size comparison).
    config = vars(args) | {"seed": seed, "in_dim": int(tr_img.shape[1]), "best_val_auroc": best_auc,
                           "stopped_epoch": epoch + 1, "n_params": sum(p.numel() for p in model.parameters())}
    # Write predictions + config into runs/<run_name>/seed<seed>/.
    save_run(args.run_name, seed, preds, config)
    # Save the best weights (state_dict = dict of parameter tensors) for later analysis
    # (modality_ablation.py reloads it).
    torch.save(best_state, run_dir(args.run_name, seed) / "model.pt")


# ---------------------------------------------------------------------------------------------------
# main
# Purpose : Parse the hyper-parameters, set the run name, and train one model per seed.
# Inputs  : command-line arguments.
# Outputs : one run folder per seed (via train_one).
# Viva tip: Q: Why lr 1e-4 and dropout 0.3? A: Small head on a small dataset overfits fast; a low LR,
#           dropout, weight decay and early stopping together keep it from memorizing the train set.
# ---------------------------------------------------------------------------------------------------
def main():
    # Command-line parser.
    ap = argparse.ArgumentParser()
    # Which fusion operator to train (must be one of FUSIONS); required.
    ap.add_argument("--fusion", choices=FUSIONS, required=True)
    # Which cached feature folder to use (e.g. clip_l14, clip_l14_inpainted, clip_b32).
    ap.add_argument("--features", default="clip_l14")
    # Random seeds to run; default three seeds 0, 1, 2.
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # Maximum number of epochs (also the cosine schedule length).
    ap.add_argument("--epochs", type=int, default=40)
    # Early-stopping patience: epochs without val AUROC improvement before stopping.
    ap.add_argument("--patience", type=int, default=8)
    # Mini-batch size.
    ap.add_argument("--batch", type=int, default=64)
    # Initial learning rate for AdamW.
    ap.add_argument("--lr", type=float, default=1e-4)
    # Weight decay strength (regularization).
    ap.add_argument("--wd", type=float, default=1e-2)
    # Projection size d for most fusions.
    ap.add_argument("--proj_dim", type=int, default=512)
    # Projection size for cross fusion (outer product -> cross_dim^2 features).
    ap.add_argument("--cross_dim", type=int, default=64)
    # Hidden size of the classifier MLP.
    ap.add_argument("--hidden", type=int, default=512)
    # Dropout probability used in projectors, classifier and Transformer.
    ap.add_argument("--dropout", type=float, default=0.3)
    # Probability that a training sample has one modality zeroed (0 = off).
    ap.add_argument("--modality_dropout", type=float, default=0.0)
    # Optional custom run name; otherwise built automatically below.
    ap.add_argument("--run_name", default=None)
    # Parse the actual command line.
    args = ap.parse_args()

    # Default run name like C_clip_l14_cross (C = CLIP-feature family of models).
    if args.run_name is None:
        args.run_name = f"C_{args.features}_{args.fusion}"
        # Add a suffix when modality dropout is on, so runs do not overwrite each other.
        if args.modality_dropout > 0:
            args.run_name += f"_mdrop{args.modality_dropout}"
    # GPU if available, else CPU.
    device = get_device()
    # Train an independent model for each seed.
    for seed in args.seeds:
        train_one(args, seed, device)


# Run main() only when executed as a script.
if __name__ == "__main__":
    main()

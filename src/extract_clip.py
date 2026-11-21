"""Extract frozen CLIP image + text embeddings once and cache them, so every fusion experiment
afterwards trains in seconds (even on CPU).

Usage:
  python src/extract_clip.py                                   # ViT-L/14 on original memes -> features/clip_l14/
  python src/extract_clip.py --img_root data/inpainted --tag clip_l14_inpainted   # text removed from image
  python src/extract_clip.py --model openai/clip-vit-base-patch32 --tag clip_b32  # smaller backbone ablation
Output: features/<tag>/<split>.npz with ids, img [N,D], txt [N,D], label [N]

How it works:
  1. CLIP (Contrastive Language-Image Pre-training, OpenAI 2021) is a pair of encoders - a Vision
     Transformer for images and a Transformer for text - trained on ~400M image-caption pairs with a
     contrastive loss: matching image/caption pairs are pulled together, non-matching pairs pushed
     apart. As a result both encoders map into ONE shared embedding space (768-d for ViT-L/14).
  2. We load CLIP once, keep it FROZEN (eval mode, no gradients), and run every meme of every split
     through it: image -> vision_model -> pooler_output -> visual_projection -> img vector,
     caption text -> tokenizer (max 77 tokens) -> text_model -> pooler_output (EOS token) ->
     text_projection -> txt vector.
  3. The vectors are saved to features/<tag>/<split>.npz. All later models (linear_probe.py,
     train_head.py) read these cached arrays instead of re-running CLIP.
  Why freeze + cache? (a) Speed: CLIP ViT-L/14 has ~430M parameters; running it once instead of every
  epoch saves hours. (b) The Hateful Memes train set is small (~8.5k memes), so fine-tuning such a big
  model would overfit easily. (c) The pretrained space is already very good at relating images and text.
"""
# argparse: read command-line options such as --model, --tag, --batch.
import argparse
# Path: OS-independent file paths (works on Windows and Linux).
from pathlib import Path

# numpy: used to save the final feature arrays into a compressed-friendly .npz file.
import numpy as np
# torch: tensors, GPU support, no_grad, etc.
import torch
# PIL.Image: opens the meme image files (.png).
from PIL import Image
# DataLoader batches the data and can load images in parallel worker processes; Dataset is the base class.
from torch.utils.data import DataLoader, Dataset
# tqdm: progress bar while extracting.
from tqdm import tqdm
# Hugging Face CLIP: CLIPModel = both encoders + projections; CLIPProcessor = image preprocessing + tokenizer.
from transformers import CLIPModel, CLIPProcessor

# Project helpers: feature folder, raw image folder, list of eval splits, CPU/GPU choice, split loader.
from common import FEAT_DIR, RAW_DIR, eval_splits, get_device, load_split


# ---------------------------------------------------------------------------------------------------
# ImageTextDataset
# Purpose : PyTorch Dataset that returns one meme at a time as (preprocessed image tensor, caption text).
# Inputs  : df = DataFrame of a split (columns: id, img, text, label), img_root = folder containing the
#           images, image_processor = CLIP's image preprocessor (resize to 224, center crop, normalize).
# Outputs : __getitem__ gives (pixels [3,224,224], text string).
# Viva tip: Q: Why not tokenize text inside the Dataset? A: Tokenizing a whole batch together lets us
#           pad to the longest caption in that batch (padding=True), which is simpler and efficient.
# Viva tip: Q: Why convert("RGB")? A: Some images are RGBA/grayscale; CLIP expects exactly 3 channels.
# ---------------------------------------------------------------------------------------------------
class ImageTextDataset(Dataset):
    # Constructor: store the split table, the image folder and the preprocessor.
    def __init__(self, df, img_root, image_processor):
        # reset_index so that row positions 0..N-1 match iloc indices used in __getitem__.
        self.df = df.reset_index(drop=True)
        # Root folder where the image paths in column "img" are relative to.
        self.img_root = img_root
        # CLIP image processor (resize, crop, scale to [0,1], normalize with CLIP mean/std).
        self.proc = image_processor

    # Number of memes in this split; the DataLoader uses this to know when an epoch ends.
    def __len__(self):
        return len(self.df)

    # Return the i-th meme.
    def __getitem__(self, i):
        # Take row i of the DataFrame (has "img" path and "text" caption).
        r = self.df.iloc[i]
        # Open the image file and force 3-channel RGB.
        img = Image.open(self.img_root / r["img"]).convert("RGB")
        # Preprocess to a tensor; processor returns a batch of 1 -> [1,3,224,224], so [0] gives [3,224,224].
        pixels = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        # Return the image tensor and the raw caption string (tokenized later, per batch).
        return pixels, r["text"]


# ---------------------------------------------------------------------------------------------------
# collate
# Purpose : Tell the DataLoader how to merge a list of (pixels, text) samples into one batch.
# Inputs  : batch = list of B tuples (pixels [3,224,224], text str).
# Outputs : (pixels [B,3,224,224] tensor, list of B text strings).
# Viva tip: Q: Why a custom collate? A: The default collate cannot stack strings into a tensor; we keep
#           texts as a plain Python list and only stack the image tensors.
# ---------------------------------------------------------------------------------------------------
def collate(batch):
    # torch.stack joins the B image tensors along a new batch dimension; texts stay a list.
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# ---------------------------------------------------------------------------------------------------
# main
# Purpose : Load frozen CLIP, encode every split (train + val/test) and save img/txt embeddings to .npz.
# Inputs  : command-line args (--model, --tag, --img_root, --batch, --workers).
# Outputs : files features/<tag>/<split>.npz with arrays ids [N], label [N], img [N,768], txt [N,768]
#           (768 for ViT-L/14; 512 for ViT-B/32).
# Viva tip: Q: Why @torch.no_grad()? A: CLIP is frozen, we never backprop, so turning off gradient
#           tracking saves a lot of memory and time.
# Viva tip: Q: Why use the projected embeddings and not the raw encoder output? A: Only after
#           visual_projection / text_projection are image and text in the SAME shared space that CLIP
#           was trained to align, so they are directly comparable and fusable.
# ---------------------------------------------------------------------------------------------------
# Decorator: disable autograd for everything inside main (pure inference).
@torch.no_grad()
def main():
    # Build the command-line parser.
    ap = argparse.ArgumentParser()
    # Which CLIP checkpoint to use; default is the large ViT-L/14 (patch size 14, 768-d embeddings).
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    # Name of the output folder under features/ (lets us keep several feature versions side by side).
    ap.add_argument("--tag", default="clip_l14")
    # Folder with the images; can point to inpainted images (text removed) for the "true image-only" study.
    ap.add_argument("--img_root", default=str(RAW_DIR))
    # Number of memes encoded together on the GPU.
    ap.add_argument("--batch", type=int, default=64)
    # Number of background processes that load/preprocess images in parallel.
    ap.add_argument("--workers", type=int, default=2)
    # Parse the actual command line into args.
    args = ap.parse_args()

    # Pick GPU (cuda) if available, else CPU.
    device = get_device()
    # Half precision (fp16) on GPU: half the memory and faster, accuracy loss is negligible for inference.
    # On CPU fp16 is slow/unsupported for many ops, so we use full fp32 there.
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Download/load pretrained CLIP weights, move to device, and set eval() (disables dropout -> deterministic).
    model = CLIPModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    # Load the matching preprocessor: image transforms + BPE tokenizer of this exact checkpoint.
    proc = CLIPProcessor.from_pretrained(args.model)
    # Output folder e.g. features/clip_l14/.
    out_dir = FEAT_DIR / args.tag
    # Create it (and parents) if missing; no error if it already exists.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loop over the training split plus every evaluation split (e.g. val, test).
    for split in ["train"] + eval_splits():
        # Load the split table (id, img path, text, label).
        df = load_split(split)
        # DataLoader over this split. No shuffle: we must keep the order so features line up with df ids.
        dl = DataLoader(ImageTextDataset(df, Path(args.img_root), proc.image_processor),
                        batch_size=args.batch, num_workers=args.workers, collate_fn=collate)
        # Lists that collect per-batch embeddings; concatenated at the end.
        img_feats, txt_feats = [], []
        # Iterate over batches: pixels [B,3,224,224], texts = list of B strings.
        for pixels, texts in tqdm(dl, desc=f"{args.tag}/{split}"):
            # Projected CLIP embeddings (same space CLIP uses for image-text matching).
            # Vision Transformer: cut the image into 14x14 patches, run attention layers. pixels are cast
            # to the model dtype (fp16 on GPU) and moved to the device.
            vis = model.vision_model(pixel_values=pixels.to(device, dtype))
            # pooler_output = the (layer-normed) CLS token = summary of the whole image, [B,1024] for ViT-L.
            # visual_projection is a linear map into the shared space -> [B,768]. Cast back to fp32 and
            # move to CPU so GPU memory does not fill up while we collect all batches.
            img_feats.append(model.visual_projection(vis.pooler_output).float().cpu())
            # Tokenize captions: pad to the longest caption in the batch, and truncate at 77 tokens because
            # CLIP's text encoder only has 77 position embeddings (its maximum context length).
            tok = proc.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            # Text Transformer; attention_mask tells it to ignore padding tokens.
            txt = model.text_model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            # For CLIP text, pooler_output is the hidden state at the EOS (end-of-text) token, which acts
            # as the sentence summary (causal mask means EOS has seen every word). Project -> [B,768].
            txt_feats.append(model.text_projection(txt.pooler_output).float().cpu())
        # Save one .npz per split: ids and labels from df (same order), and stacked features [N,768].
        # NOTE: features are saved UN-normalized; L2 normalization is done later inside the models.
        np.savez(out_dir / f"{split}.npz", ids=df["id"].values, label=df["label"].values,
                 img=torch.cat(img_feats).numpy(), txt=torch.cat(txt_feats).numpy())
        # Log where the file went and how many memes it holds.
        print(f"saved {out_dir / f'{split}.npz'}  ({len(df)} memes)")


# Run main() only when this file is executed directly (not when imported).
if __name__ == "__main__":
    main()

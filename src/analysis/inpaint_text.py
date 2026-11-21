"""Remove the overlay text from every meme image (research angle D).

CLIP can READ text inside images, so a CLIP "image-only" model secretly sees the caption too.
We detect text with EasyOCR, mask it, and fill it with OpenCV inpainting. Then re-extract features:
  python src/extract_clip.py --img_root data/inpainted --tag clip_l14_inpainted
  python src/train_head.py --fusion image_only --features clip_l14_inpainted
and compare with image_only on the original images.

Usage: python src/analysis/inpaint_text.py      (GPU: ~20-30 min for all splits; resumable)
Output: data/inpainted/img/*.png  + results/figures/inpaint_examples.png

How it works (overview):
- Why is this needed? CLIP was trained on 400M image-caption pairs from the internet and learned to read
  printed text in images (OCR-like ability). A meme's caption is printed on the image, so a CLIP
  "image-only" model is not really image-only: it also gets the text. To measure the true image-only
  performance we must remove the caption from the pixels.
- Step 1 (OCR): EasyOCR finds text regions and returns, for each one, a bounding box (4 corner points),
  the recognised string and a confidence score.
- Step 2 (mask): we draw every box filled with white (255) on a black (0) mask of the image size.
  White = "pixels to repaint". We dilate (grow) the mask a little so the edges and outlines of letters
  are covered too.
- Step 3 (inpaint): OpenCV's Telea algorithm fills the masked pixels using the surrounding pixels, starting
  at the border of the hole and moving inward. The caption disappears and the background is guessed.
- Results are written to data/inpainted with the same relative paths; already-done images are skipped,
  so the script can be stopped and restarted (resumable). Finally a before/after figure is saved.
"""
# argparse: reads command-line options (here --dilate).
import argparse
# sys: lets us change the import search path.
import sys
# Path: OS-independent file paths.
from pathlib import Path

# Add the "src" folder to the import path so we can import our "common" helper module.
# Must stay before the "from common import ..." line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# cv2 = OpenCV: reading/writing images, drawing the mask, dilation and inpainting.
import cv2
# easyocr: deep-learning OCR library that detects and reads text in images.
import easyocr
# matplotlib: plotting library for the example figure.
import matplotlib

# "Agg" backend = draw into files only, no window; needed on servers without a display.
matplotlib.use("Agg")
# pyplot plotting API (imported after the backend is set).
import matplotlib.pyplot as plt
# numpy: arrays for the mask and the box coordinates.
import numpy as np
# pandas: combining the split tables.
import pandas as pd
# torch: only used to check if a CUDA GPU is available for EasyOCR.
import torch
# tqdm: shows a progress bar in the terminal.
from tqdm import tqdm

# Project helpers: figure folder, output folder for inpainted images, original image folder,
# the list of evaluation splits, and the split loader.
from common import FIG_DIR, INPAINT_DIR, RAW_DIR, eval_splits, load_split


# ---------------------------------------------------------------------------------------------------------
# inpaint
# Purpose : remove the overlay text from ONE image and save the cleaned image.
# Inputs  : reader - a ready EasyOCR Reader object (created once, reused for speed)
#           src    - path of the original image
#           dst    - path where the inpainted image is written
#           dilate - size of the square kernel used to grow the text mask (0 = no dilation)
# Output  : nothing returned; the cleaned image is written to dst.
# Viva tip: Q: What is inpainting? A: Filling a missing/masked region of an image with plausible content
#           computed from the pixels around it. We use it to "erase" the caption.
# Viva tip: Q: Why dilate the mask? A: OCR boxes are tight; letters often have outlines or shadows just
#           outside the box. Growing the mask removes those leftovers so CLIP cannot read partial letters.
# Viva tip: Q: Why ignore boxes with confidence <= 0.1? A: Very low-confidence detections are often not text
#           (textures, edges); masking them would destroy image content for no reason.
# ---------------------------------------------------------------------------------------------------------
def inpaint(reader, src: Path, dst: Path, dilate: int):
    # Read the image as a numpy array of shape (H, W, 3). Note: OpenCV uses BGR channel order, not RGB.
    img = cv2.imread(str(src))
    # Start with an all-black mask of shape (H, W), type uint8 (values 0..255). 0 = keep this pixel.
    mask = np.zeros(img.shape[:2], np.uint8)
    # Run OCR. With detail=1 each result is (box, text, confidence); box = 4 corner points [x, y].
    # We do not need the recognised text itself, only where it is, so it is ignored with "_".
    for box, _, conf in reader.readtext(img, detail=1):
        # Keep only detections with confidence above 0.1 (remove obvious false detections).
        if conf > 0.1:
            # Fill the 4-corner polygon with white (255) on the mask. Boxes may be rotated, so we use a
            # polygon fill instead of a rectangle. OpenCV needs integer (int32) point coordinates.
            cv2.fillPoly(mask, [np.array(box, dtype=np.int32)], 255)
    # If dilation is enabled (dilate > 0):
    if dilate:
        # Dilation grows every white area: a pixel becomes white if any pixel under the dilate x dilate
        # kernel is white. iterations=2 applies it twice, so the mask grows even more.
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8), iterations=2)
    # Inpaint: fill all white mask pixels using neighbouring pixels. 5 = radius (in pixels) of the
    # neighbourhood considered. INPAINT_TELEA = the "fast marching" method by Telea (2004), which fills the
    # hole from its border inward using a weighted average of known nearby pixels.
    out = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    # Make sure the output folder exists.
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Save the cleaned image to disk.
    cv2.imwrite(str(dst), out)


# ---------------------------------------------------------------------------------------------------------
# main
# Purpose : inpaint every unique image of the train and evaluation splits, then save a before/after figure.
# Inputs  : command-line --dilate (mask dilation kernel size, default 7)
# Outputs : inpainted images in INPAINT_DIR and results/figures/inpaint_examples.png
# Viva tip: Q: Why does a CLIP "image-only" model do surprisingly well? A: CLIP can read the caption printed
#           in the image, so it secretly uses the text. Inpainting removes the text to test the real image
#           signal; the drop in AUROC shows how much came from reading text.
# Viva tip: Q: Is inpainting perfect? A: No. Some text may be missed by OCR, and the filled area can be
#           blurry. That is why we look at example images to sanity-check the result.
# ---------------------------------------------------------------------------------------------------------
def main():
    # Create the argument parser.
    ap = argparse.ArgumentParser()
    # Dilation kernel size in pixels (7 x 7 by default); 0 turns dilation off.
    ap.add_argument("--dilate", type=int, default=7)
    # Parse the command line.
    args = ap.parse_args()

    # Create the OCR reader for English once (loading the OCR neural networks is slow).
    # Use the GPU if CUDA is available, otherwise run on CPU.
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    # Load the train split and all evaluation splits, stack them into one table, and keep each image only
    # once (the same image file may appear in more than one split).
    df = pd.concat([load_split(s) for s in ["train"] + eval_splits()]).drop_duplicates("img")
    # Loop over all relative image paths with a progress bar.
    for rel in tqdm(df["img"], desc="inpainting"):
        # Output path keeps the same relative path, just under INPAINT_DIR instead of RAW_DIR.
        dst = INPAINT_DIR / rel
        # Skip images already done -> the script is resumable if it was interrupted.
        if not dst.exists():
            # Remove the text from this image and save it.
            inpaint(reader, RAW_DIR / rel, dst, args.dilate)

    # Make sure the figure folder exists.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Pick 4 random images; random_state=1 makes the choice the same every time (reproducible).
    sample = df.sample(4, random_state=1)["img"]
    # Figure grid: 2 rows (original / text removed) x 4 columns (one per sample image).
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    # k = column number, rel = relative path of the image.
    for k, rel in enumerate(sample):
        # Row 0 shows the original from RAW_DIR, row 1 the cleaned image from INPAINT_DIR.
        for row, root, name in [(0, RAW_DIR, "original"), (1, INPAINT_DIR, "text removed")]:
            # Read with OpenCV (BGR) and convert to RGB, because matplotlib expects RGB colour order.
            axes[row, k].imshow(cv2.cvtColor(cv2.imread(str(root / rel)), cv2.COLOR_BGR2RGB))
            # Small title saying which version it is.
            axes[row, k].set_title(name, fontsize=9)
            # Hide axis ticks for images.
            axes[row, k].axis("off")
    # Fix spacing between subplots.
    fig.tight_layout()
    # Save the before/after figure.
    fig.savefig(FIG_DIR / "inpaint_examples.png", dpi=110)
    # Report where the inpainted images are.
    print(f"done -> {INPAINT_DIR}")


# Run main() only when the file is executed directly.
if __name__ == "__main__":
    main()

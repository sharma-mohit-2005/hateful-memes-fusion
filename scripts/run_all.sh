#!/usr/bin/env bash
# Full pipeline, top to bottom. Run from the repo root on a GPU machine (Colab / Kaggle):
#   bash scripts/run_all.sh
# Each stage is idempotent-ish: comment out stages you have already finished.
set -euo pipefail
SEEDS="0 1 2"

echo "== 1. data =="
python src/download_data.py
python src/prepare_splits.py
python src/eda.py

echo "== 2. baselines M1-M4 (fine-tuned BERT / ResNet) =="
python src/train_unimodal.py --modality text  --seeds $SEEDS
python src/train_unimodal.py --modality image --seeds $SEEDS
python src/train_unimodal.py --modality both  --seeds $SEEDS
python src/late_fusion.py --text_run M1_bert_text --image_run M2_resnet_image

echo "== 3. CLIP features + linear probe (M5) =="
python src/extract_clip.py --tag clip_l14
python src/linear_probe.py --features clip_l14

echo "== 4. fusion ablation on CLIP (M6, M7 + all operators) =="
for F in image_only text_only concat product concat_product cross gated attention; do
  python src/train_head.py --fusion $F --features clip_l14 --seeds $SEEDS
done
python src/train_head.py --fusion concat --features clip_l14 --modality_dropout 0.3 --seeds $SEEDS
python src/late_fusion.py --text_run C_clip_l14_text_only --image_run C_clip_l14_image_only --prefix C_late

echo "== 5. angle D: remove text from images =="
python src/analysis/inpaint_text.py
python src/extract_clip.py --img_root data/inpainted --tag clip_l14_inpainted
for F in image_only concat cross; do
  python src/train_head.py --fusion $F --features clip_l14_inpainted --seeds $SEEDS
done

echo "== 6. analysis =="
python src/analysis/make_tables.py
python src/analysis/confounders.py --split test --sim_features clip_l14_inpainted --show_run C_clip_l14_cross
python src/analysis/modality_ablation.py --split test
python src/analysis/error_examples.py --run C_clip_l14_cross --split test
echo "All done. See results/summary.md, results/confounders/, results/modality_ablation_test.md, results/figures/"

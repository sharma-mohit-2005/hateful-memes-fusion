# Multimodal Hateful Meme Detection: Does Fusion Beat Unimodal?

Deep Learning course project (Sem 5, Sitare University).
We classify memes from the **Hateful Memes Challenge** (Kiela et al., NeurIPS 2020) as hateful or not. The dataset is
built with *benign confounders*: the same caption on a different image, or the same image with a different caption,
flips the label. Because of that, a model has to understand image and text **together** to get these memes right.

**Research questions**
1. Does fusing image and text beat the best single-modality model, and by how much?
2. Which **fusion strategy** works best on top of frozen CLIP embeddings? (concat, product, cross, gated, attention, late)
3. How do models perform on **benign-confounder pairs**, compared with the rest of the data?
4. Does a model really use both modalities? (test-time modality shuffling)
5. Is a CLIP "image-only" model really image-only? CLIP can read the caption printed in the image, so we test this by removing the text from the image first.

## Models

| ID | Model | Script |
|----|-------|--------|
| M1 | BERT (bert-base-uncased) fine-tuned, text only | `train_unimodal.py --modality text` |
| M2 | ResNet-50 (ImageNet) fine-tuned, image only | `train_unimodal.py --modality image` |
| M3 | Early fusion: [BERT CLS ; ResNet] → MLP, trained end-to-end | `train_unimodal.py --modality both` |
| M4 | Late fusion of M1 + M2 probabilities (average / val-tuned weight) | `late_fusion.py` |
| M5 | CLIP ViT-L/14 linear probe (logistic regression) | `linear_probe.py` |
| M6 | Frozen CLIP + MLP head (concat) | `train_head.py --fusion concat` |
| M7 | Frozen CLIP + cross fusion (Hate-CLIPper style) | `train_head.py --fusion cross` |
| Ablation | image_only, text_only, product, concat_product, gated, attention, modality dropout | `train_head.py --fusion ...` |

Every model is trained with **3 seeds**, selected on **val** AUROC, and reported as mean ± std on **test**.
Splits: `train` = train (8.5k), `val` = dev_seen (500), `test` = test_seen (1000), plus `test_unseen` (2000) as an
extra test set. Metrics are **AUROC** (primary), accuracy, and F1.

## Repository layout

```
├── src/
│   ├── common.py            paths, metrics, saving runs
│   ├── download_data.py     dataset from Hugging Face → data/hateful_memes/
│   ├── prepare_splits.py    → data/splits/{train,val,test,test_unseen}.jsonl
│   ├── eda.py               dataset stats + figures
│   ├── train_unimodal.py    M1, M2, M3 (fine-tuned BERT / ResNet)
│   ├── late_fusion.py       M4
│   ├── extract_clip.py      cache CLIP image/text embeddings → features/
│   ├── linear_probe.py      M5
│   ├── fusion_heads.py      all fusion operators
│   ├── train_head.py        M6, M7 + fusion ablation
│   └── analysis/
│       ├── make_tables.py        results/summary.md + bar chart
│       ├── confounders.py        benign-confounder pair analysis
│       ├── modality_ablation.py  shuffle / remove one modality at test time
│       ├── inpaint_text.py       erase the caption from the images (OCR + inpainting)
│       └── error_examples.py     most confident false positives / negatives
├── notebooks/run_pipeline.ipynb  run everything on Colab / Kaggle
├── scripts/run_all.sh            same pipeline as one bash script
├── report/REPORT_TEMPLATE.md     report skeleton
└── results/                      metrics, predictions, figures (committed; model weights are not)
```

## How to run

You need a GPU. A free Colab or Kaggle **T4** is enough; a 4 GB laptop GPU is too small for fine-tuning BERT or ResNet.

```bash
pip install -r requirements.txt
python src/download_data.py && python src/prepare_splits.py
bash scripts/run_all.sh          # or run notebooks/run_pipeline.ipynb cell by cell
```

Approximate time on a T4: baselines ~1.5 h, CLIP features ~10 min, whole fusion ablation ~15 min,
inpainting ~30 min.

## Results

> Paste `results/summary.md`, `results/confounders/results_test.md` and `results/modality_ablation_test.md` here
> after running, and add `results/figures/test_auroc.png`.

## References
1. Kiela et al., *The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes*, NeurIPS 2020. arXiv:2005.04790
2. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML 2021. arXiv:2103.00020
3. Kumar & Nandakumar, *Hate-CLIPper: Multimodal Hateful Meme Classification based on Cross-modal Interaction of CLIP Features*, 2022. arXiv:2210.05916

**Content warning:** the dataset contains offensive content. Dataset © Meta; it is used under its research license
and is not redistributed in this repo.

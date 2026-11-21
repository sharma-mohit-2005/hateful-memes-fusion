# Report skeleton (6–8 pages, IEEE / NeurIPS two-column style recommended, write in Overleaf)

For each section, the line in brackets says which result file to take the content from.

---

## Abstract (150–200 words)
Problem → why it's hard (benign confounders) → what you did (unimodal vs 8 fusion strategies on CLIP + analysis)
→ headline numbers (best unimodal AUROC vs best fusion AUROC) → the key insight.

## 1. Introduction
- Hateful memes: the meaning comes only from the image and text *together*.
- Research questions RQ1–RQ5 (copy them from the README).
- Contributions (3–4 bullets): controlled fusion ablation; confounder-pair metric; modality reliance test;
  the text-inpainting test of whether CLIP "image-only" is really image-only.

## 2. Related Work
Hateful Memes challenge [Kiela'20]; VisualBERT/ViLBERT (from the challenge paper); CLIP [Radford'21];
Hate-CLIPper [Kumar'22]; fusion taxonomy (early / late / intermediate).

## 3. Dataset
[results/eda_stats.csv, figures/eda_distribution.png]
- Split sizes and class balance; why we use AUROC.
- Benign confounders: explanation plus one example figure (blur or crop offensive parts).

## 4. Methods
- 4.1 Unimodal baselines: BERT, ResNet-50 (setup and hyperparameters, from config.json).
- 4.2 Early and late fusion with fine-tuned encoders.
- 4.3 Frozen CLIP + fusion heads: write the equation for each operator (taken from `src/fusion_heads.py`).
- 4.4 Training protocol: 3 seeds, val-based model selection, early stopping, metrics.
- 4.5 Analysis methods: confounder pair mining (text match + CLIP image similarity), pair_acc and pair_rank,
  modality shuffling, OCR + inpainting.

## 5. Results
[results/summary.md, figures/test_auroc.png]
**Table 1**: all models, test AUROC / accuracy (mean ± std), grouped as Unimodal | Fusion (fine-tuned) | CLIP.
Answer RQ1: by how much does the best fusion model beat the best unimodal model?

## 6. Fusion Ablation
[C_clip_l14_* rows of summary.md]
**Table 2**: fusion operator vs test AUROC and #params. Answer RQ2: do interaction terms (product, cross) help
more than plain concatenation? Does modality dropout help?

## 7. Analysis
- 7.1 Benign confounders [results/confounders/results_test.md, pairs_test.png]: pair_acc and pair_rank for
  unimodal vs fusion models. Expected finding: unimodal models are near chance on pairs.
- 7.2 Modality reliance [results/modality_ablation_test.md]: AUROC drop when text or image is shuffled.
- 7.3 Is image-only really image-only? [C_clip_l14_image_only vs C_clip_l14_inpainted_image_only,
  figures/inpaint_examples.png]
- 7.4 Qualitative errors [results/errors/*.png and csv]: hand-label ~40 errors into categories and report a table
  of category counts.

## 8. Limitations & Ethics
Dataset size and label noise; English only; OCR mistakes in inpainting; models can encode bias against
identity groups; risks of misuse and of over-censoring; offensive-content handling.

## 9. Conclusion
3–4 sentences answering the title question, plus future work (larger VLMs, LLM-generated captions, multilingual memes).

## References
At least the three references in the README, plus any other papers you cite.

## Appendix
Hyperparameters table, extra plots, test_unseen results.

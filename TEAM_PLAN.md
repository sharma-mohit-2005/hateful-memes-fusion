# Team Plan & Contributions

Deadline: **22 Nov 2026**. Work is split equally: every member owns ~250 lines of code, one set of Colab
experiments and 2-4 report sections. Every member opens their own pull requests from their own GitHub account.

## Who does what

| Member | Owns (code) | Runs on Colab | Report sections |
|---|---|---|---|
| **Mohit Sharma** (lead) | `src/fusion_heads.py`, `src/train_head.py`, `src/linear_probe.py`, `src/analysis/make_tables.py` + notebook, README, report template | CLIP linear probe (M5), fusion ablation (M6, M7, all fusion operators), final results table | Abstract, Introduction, Fusion Method & Ablation, Conclusion |
| **Member A** | `src/train_unimodal.py`, `src/late_fusion.py`, `src/analysis/error_examples.py` | M1 BERT text-only, M2 ResNet image-only, M4 late fusion, error examples | Related Work, Unimodal Baselines, Error Analysis |
| **Member B** | `src/common.py`, `src/download_data.py`, `src/prepare_splits.py`, `src/eda.py`, `src/extract_clip.py` | Data download + EDA, M3 early fusion, CLIP feature extraction | Dataset, Early Fusion |
| **Member C** | `src/analysis/confounders.py`, `src/analysis/modality_ablation.py`, `src/analysis/inpaint_text.py` | Text inpainting experiment, benign-confounder analysis, modality reliance test | Analysis, Limitations & Ethics |

Mohit (lead) also reviews and merges all pull requests and keeps README and the notebook up to date.

## Order of work (dependencies)

1. **Base** (Mohit): README, team plan, notebook, report template on `main`.
2. **Data PR** (B) is merged first, because every script imports `src/common.py`.
3. **Code PRs** of A, C and Mohit (any order).
4. **Training** on Colab, then one **results PR** per member.
5. **Final analysis** after all results are merged: C (confounders, modality test), A (error examples),
   Mohit (final tables) → README results section → report.

## Timeline

| When | Mohit | A | B | C |
|---|---|---|---|---|
| 21 Nov, now | Push base, add collaborators, protect `main`, set up Overleaf | Code PR | **Data PR first** | Code PR |
| 21 Nov, evening | Fusion ablation, results PR | M1, M2, M4, results PR | EDA, M3, results PR | Inpainting, results PR |
| 21 Nov, night | Final tables, README results | Error examples | Write Dataset section | Confounders + modality test |
| 22 Nov | Merge report, submit | Write sections | Write sections | Write sections |

## Git workflow (pull requests)

`main` is protected: nobody pushes to it directly. All work arrives through pull requests (PRs).

1. Start from the latest main: `git checkout main && git pull`
2. Create a branch for one piece of work: `git checkout -b <type>/<short-name>`
   (e.g. `feat/baselines`, `results/baselines`)
3. Commit only your own files and your own `results/...` outputs.
4. Push the branch: `git push -u origin <branch>` and open a PR on GitHub into `main`.
5. PR description: what you did, how you ran it, key numbers (for results PRs).
6. Mohit reviews and merges; Mohit's own PRs are reviewed by another member before merging.
7. Never commit `data/`, `features/`, `*.pt` or `results/figures/eda_samples.png` (offensive memes).

## Use of AI tools
Code scaffolding was assisted by an AI coding tool (permitted for this course). Experiments, analysis and the
report were carried out by the team members listed above.

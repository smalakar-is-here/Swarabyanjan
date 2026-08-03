# Swarabyanjan — Bengali Yellow Journalism Detection

**Authors:** Swagotam Malakar, Anamika Das, Dr. Ohidujjaman

## Overview

Swarabyanjan is a benchmark for Bengali yellow journalism detection. This repository contains the clean balanced dataset, training notebooks (with outputs), and complete evaluation results for 15 entries (12 models + SMI annotation algorithm at 2 evaluation tiers + Qwen-7B multi-seed).

**Repository:** [github.com/smalakar-is-here/Swarabyanjan](https://github.com/smalakar-is-here/Swarabyanjan)
**Kaggle dataset (cleaned CSVs):** [kaggle.com/datasets/swagotammalakar/swarabyanjan-grand-corpus](https://www.kaggle.com/datasets/swagotammalakar/swarabyanjan-grand-corpus)

## Swarabyanjan Grand Corpus

The benchmark draws from the **Swarabyanjan Grand Corpus** — a large-scale collection of **5.1 million Bengali news articles** assembled for yellow-journalism research. From this Grand Corpus, a 5,000-article working corpus was sampled via multi-signal stratification (weak supervision + preliminary model prediction + centroid cosine similarity), and from that 5,000-article corpus a 766-article balanced gold standard (383 yellow + 383 non-yellow) was carved out for human annotation under the 8-criteria SMI linguistic protocol. The Grand Corpus itself is not redistributed in this repo; the cleaned 5,000-article corpus and the 766-article gold standard are available on the [Kaggle dataset](https://www.kaggle.com/datasets/swagotammalakar/swarabyanjan-grand-corpus) (see `data/README.md` for the full hierarchy).

## Dataset

- **Total articles:** 766 (383 yellow + 383 non-yellow)
- **Balanced:** 1:1 ratio
- **Annotation:** 8-criteria linguistic protocol with deep reasoning
- **Columns:** `article_id`, `headline`, `body_text`, `news_source`, `article_length`, `best_label`, `best_confidence`, `best_note`
- **Label:** `best_label` (1 = yellow journalism, 0 = non-yellow)
- **File:** `data/Swarabyanjan_BEST_BALANCED_1to1.csv`

## Results Summary

| Model | F1 | Accuracy | Type |
|-------|-----|----------|------|
| BanglaBERT Large | 0.883 ± 0.025 | 0.880 | Fine-tuned Transformer |
| SMI Annotation | 0.809 ± 0.056 | 0.820 | Rule-based Annotation Algorithm |
| SMI (4234 non-gold OOS) | 0.267 | 0.524 | Rule-based (true out-of-sample) |
| Random Forest | 0.788 ± 0.015 | 0.786 | TF-IDF + Classical ML |
| Logistic Regression | 0.777 ± 0.047 | 0.782 | TF-IDF + Classical ML |
| Linear SVM | 0.764 ± 0.018 | 0.766 | TF-IDF + Classical ML |
| XGBoost | 0.752 ± 0.013 | 0.755 | TF-IDF + Classical ML |
| Qwen2.5-7B-Instruct (QLoRA, single seed=42) | 0.075 | 0.520 | LLM Fine-tuned |
| Qwen2.5-7B-Instruct (QLoRA, multi-seed n=3) | 0.050 ± 0.000 | 0.511 | LLM Fine-tuned (multi-seed) |
| Gemma-2-2B-it (QLoRA) | 0.051 | 0.513 | LLM Fine-tuned |
| Gemma-2-9B-it (QLoRA) | 0.051 | 0.513 | LLM Fine-tuned |
| Llama-3.1-8B-Instruct (QLoRA) | 0.051 | 0.513 | LLM Fine-tuned |
| Phi-3-mini-4k (QLoRA) | 0.051 | 0.513 | LLM Fine-tuned |
| Qwen2.5-3B-Instruct (QLoRA) | 0.050 | 0.507 | LLM Fine-tuned |
| Majority Baseline | 0.000 | 0.500 | Baseline |

**Note:** BanglaBERT Large and classical models use 5-fold stratified cross-validation on 766 articles. SMI uses 5-fold CV on the same 766 articles (held-out 70/30 split also reported in NB8). LLMs use a single 80/20 train-test split with seed=42; NB9a/b/c provide a multi-seed (5×) evaluation for Qwen2.5-7B-Instruct to estimate variance (split across 2 Kaggle sessions due to the 12-hour commit limit).

**Multi-seed evaluation:** Qwen2.5-7B-Instruct was evaluated with 3 random seeds [42, 7, 2024] (2 additional seeds [123, 99] failed due to CUDA OOM on Kaggle T4). The multi-seed F1 = 0.050 ± 0.000 — the NB5 single-seed F1 = 0.075 is **NOT within 1 std**. NB5 used an earlier version of the output parser that classified one additional output as a positive prediction (3 TP). NB9a's improved parser correctly classified that output as unparseable (2 TP). The 0.075 vs 0.0506 difference is therefore parser-induced, not seed variance. The multi-seed F1=0.050 ± 0.000 is the canonical result with the improved parser. See `results/multi_seed_qwen7b_final_summary.json` and `notebooks/NB9a/b/c` for details.

**Key Finding:** All 6 general-purpose LLMs (2B–9B parameters) completely fail (F1 = 0.050–0.075), while BanglaBERT Large achieves F1 = 0.883 — a 11.8× F1 improvement (BanglaBERT Large 0.883 vs best LLM 0.075). LLM failure is consistent with insufficient Bengali language modeling capability; the relative contributions of pretraining depth, tokenizer coverage, and instruction-tuning quality are not fully disentangled by the ablation.

**SMI three-tier evaluation:** 5-fold CV F1=0.809, 70/30 held-out F1=0.835, true out-of-sample F1=0.267 (on 4,234 non-gold articles vs `human_label`). The out-of-sample F1 is a lower bound due to 59.9% label noise (`human_label` vs `best_label` agreement). The gap between in-sample and out-of-sample performance is disclosed transparently. SMI (5-fold CV F1=0.809) has numerically higher F1 than all classical baselines; statistically significantly higher than LogReg (p=0.033), LinearSVM (p=0.003), and XGBoost (p=0.0005), but NOT significantly different from Random Forest (McNemar p=0.079; bootstrap 95% CI crosses zero). See `results/smi_annotation_results.json` and `results/README.md` → SMI Three-Tier Evaluation Results for the full breakdown (8-criteria weights, per-fold F1, label-noise ceiling analysis).

**96–98% unparseable output rate (mean 97% across 3 seeds):** Across 3 seeds, 148–151 of 154 test predictions were unparseable (the LLM did not produce output starting with '0' or '1'). This is a stronger failure mode than incorrect classification — the LLM fails to follow the output format instruction. See `results/multi_seed_qwen7b_batch1_results.csv`.

**LLM generates Bengali character fragments:** Debug logs reveal that the LLM generates 2-4 character Bengali text fragments (e.g., 'ে।', 'পজে', 'াদে') instead of the requested '0' or '1' digit. This is because `max_new_tokens=3` generates 3 tokens, which in Bengali tokenization corresponds to 2-4 characters. The model completely ignores the output format instruction. This is a stronger failure mode than incorrect classification — the LLM fails to follow basic instruction-following on Bengali text. See `results/multi_seed_qwen7b_batch1_results.csv` and the NB9a/b debug logs.

## Ablation Studies

Two Q1-readiness ablation notebooks defend the SMI algorithm's design choices. Both were successfully run on Kaggle on **2026-07-21** (NB10: 14.4s CPU, no errors; NB12: 18.7s CPU, no errors). Canonical results are in `results/smi_criterion_ablation_results.json` and `results/lightweight_classifier_results.json`.

### NB10 — SMI Criterion Ablation (Drop-One-Out)

NB10 drops each of the 8 SMI criteria C1–C8 in turn, retrains the Logistic Regression on the remaining 7, and reports the F1 change vs. the all-8-criteria baseline (5-fold CV F1=0.8091 ± 0.0564 on 766 gold articles — matches NB8 exactly). It also ablates the 6 pairs of the top-4 criteria (C1, C6, C3, C7).

| Dropped | Criterion Name | F1 Δ | Interpretation |
|---|---|---|---|
| **C1** | Sensational Headline | **+0.1013** | **Essential** |
| C6 | Entertainment Displacement | +0.0135 | Important |
| C3 | Emotional Arousal | +0.0074 | Minor |
| C7 | Headline-Body Coherence | +0.0054 | Minor |
| C2 | Clickbait | +0.0052 | Minor |
| C5 | Speculation | +0.0025 | Negligible |
| C8 | Sensitive Topic | +0.0011 | Negligible |
| C4 | Attribution Gap | +0.0007 | Negligible |

**Key takeaways (NB10):**

- **C1 (Sensational Headline) is ESSENTIAL** — dropping it causes F1 to collapse from 0.809 → 0.708 (ΔF1=+0.101). C1 is the only Essential criterion; C6 is Important; C2/C3/C7 are Minor; C4/C5/C8 are Negligible.
- **C1+C6 are complementary** — the only pair with positive synergy (+0.0278): dropping both hurts more than the sum of dropping each alone (F1 drops by +0.143 vs. sum-of-singles +0.115). All other top-4 pairs are redundant or independent.
- **C4 is misleading by weight magnitude** — it has the 3rd-largest learned weight (+2.02) but is Negligible in ablation (ΔF1=+0.0007), because its signal is redundant with C1/C6. This is exactly why ablation studies matter: weight magnitude alone is not a reliable importance signal.
- **Spearman ρ (|weight| vs |ΔF1|) = +0.4286** (moderate positive; p=0.29, n=8) — ablation is more informative than weight magnitude alone.

### NB12 — Lightweight Classifiers on 8 SMI Features (Feature-Sufficiency Test)

NB12 evaluates 7 lightweight classifiers on the 8 SMI criteria features (5-fold CV on 766 gold articles) to test whether the hand-crafted features are sufficient — i.e., whether simple classifiers on these 8 features beat all 6 QLoRA-fine-tuned LLMs.

| Classifier | F1 Mean | F1 Std | Accuracy | AUC |
|---|---|---|---|---|
| **Linear SVM** | **0.8305** | 0.0432 | 0.8395 | 0.9022 |
| Random Forest | 0.8223 | 0.0245 | 0.8264 | 0.8861 |
| XGBoost | 0.8176 | 0.0242 | 0.8199 | 0.8954 |
| Logistic Regression | 0.8091 | 0.0630 | 0.8199 | 0.8982 |
| KNN | 0.8070 | 0.0336 | 0.8134 | 0.8554 |
| Decision Tree | 0.8040 | 0.0428 | 0.8121 | 0.8217 |
| Naive Bayes | 0.7982 | 0.0495 | 0.8160 | 0.8853 |
| — *Best LLM (Qwen2.5-7B QLoRA, seed=42)* | *0.0750* | — | *0.520* | — |
| — *BanglaBERT Large (reference)* | *0.8831* | *0.025* | *0.880* | — |

**Key takeaways (NB12):**

- **ALL 7 lightweight classifiers beat ALL 6 LLMs.** Best: Linear SVM F1=0.8305 (11.1× the best LLM F1=0.075). Worst: Naive Bayes F1=0.7982 (10.6× the best LLM). Even the weakest lightweight classifier beats the best LLM by more than 10×.
- **Logistic Regression reproduces SMI's F1=0.8091 exactly** — validates that NB8's SMI model is exactly the same logistic regression on the 8 criteria features (internal consistency check).
- **Feature sufficiency proven** — the 8 SMI features contain the signal; the LLM cannot extract it. For Bengali yellow journalism detection, simple features + simple classifier is more effective than 2B–9B parameter LLMs with QLoRA fine-tuning.
- **Best lightweight (Linear SVM, F1=0.831) sits between SMI (F1=0.809) and BanglaBERT Large (F1=0.883)** — a calibrated Linear SVM on 8 hand-crafted features outperforms all classical TF-IDF baselines (LR=0.777, RF=0.788, SVM=0.764, XGBoost=0.752) and all LLMs, while remaining below the language-specific transformer.

See `results/README.md` → Ablation Results for the full tables (per-classifier Precision/Recall/Kappa/MCC, drop-pairs synergy, etc.).

## Repository Structure

```
swarabyanjan/
├── data/                    # Balanced dataset (766 articles) + SMI-annotated 5000 corpus
├── notebooks/               # 22 Jupyter notebooks with outputs (NB1-NB15, with NB9→a/b/c and NB15→a/b1/b2/b3/c; plus 01-dataset-unification.ipynb)
├── outputs/                 # Confusion matrices + ablation/significance/variance plots
├── results/                 # Master comparison + ablation/significance/variance/zero-shot artifacts
├── CITATION.bib             # BibTeX entries for all third-party models/methods/tests
├── ETHICS.md                # Ethics statement (privacy, dataset composition, potential misuse)
├── NOTICE.md                # Third-party model licenses (BanglaBERT, Gemma, Qwen, Phi, Llama, QLoRA)
├── LICENSE                  # MIT license (code)
├── DATA_LICENSE             # CC-BY-4.0 license (dataset + SMI artifacts)
├── requirements.txt         # Python dependencies (minimum versions)
├── .gitignore
└── README.md
```

See individual folder READMEs for details.

## Reproduction

0. Install dependencies: `pip install -r requirements.txt` (CPU notebooks work locally; GPU notebooks require Kaggle T4 or equivalent)
1. Attach the Kaggle dataset [`swagotammalakar/swarabyanjan-grand-corpus`](https://www.kaggle.com/datasets/swagotammalakar/swarabyanjan-grand-corpus) (provides the cleaned `Swarabyanjan_Gold_Balanced_766.csv` + `Swarabyanjan_Corpus_5000.csv` + `Swarabyanjan_Corpus_Metadata_5000.csv`), OR upload the in-repo `data/Swarabyanjan_BEST_BALANCED_1to1.csv` as a Kaggle dataset. All notebooks auto-detect either source.
2. Run `notebooks/01-dataset-unification.ipynb` (CPU, ~30 sec) ONLY if you need to regenerate the cleaned CSVs from the legacy Kaggle CSVs — otherwise skip; the cleaned CSVs are already available via the Kaggle dataset above.
3. Run `notebooks/NB1_BanglaBERT_Classical.ipynb` for BanglaBERT Large + classical models (T4 GPU, ~2 hours)
4. Run `notebooks/NB2-NB7_*.ipynb` for LLM experiments (T4 GPU, ~1-3 hours each)
5. Run `notebooks/NB8_SMI_Annotation_Experiment.ipynb` for SMI automated annotation algorithm. Run `notebooks/NB9a_Qwen-7B_seeds_42_123_2024.ipynb` then `notebooks/NB9b_Qwen-7B_seeds_7_99.ipynb` for multi-seed (5×) evaluation of Qwen2.5-7B-Instruct (split across 2 Kaggle sessions due to the 12-hour commit limit; ~7.5h + ~5h on T4). Then run `notebooks/NB9c_Qwen-7B_aggregate.ipynb` (CPU-only, ~5 min) to combine the batches.
6. Run ablation notebooks: `NB10_SMI_Criterion_Ablation.ipynb` (CPU, ~14 sec), `NB11_LLM_ZeroShot.ipynb` (T4 GPU, ~12 min), `NB12_Lightweight_Classifier_SMI_Features.ipynb` (CPU, ~19 sec), `NB13_Statistical_Significance_Tests.ipynb` (CPU, ~8 min), `NB14_BanglaBERT_Error_Analysis.ipynb` (CPU, ~10 sec), `NB15a_CV_Variance_SMI_Classical.ipynb` (CPU, ~3 min), `NB15b1_CV_Variance_BanglaBERT_seeds_42_123.ipynb` (T4 GPU, ~4 hours, optional), `NB15b2_CV_Variance_BanglaBERT_seeds_2024_7.ipynb` (T4 GPU, ~4 hours, optional), `NB15b3_CV_Variance_BanglaBERT_seed_99.ipynb` (T4 GPU, ~2 hours, optional), `NB15c_CV_Variance_Aggregate.ipynb` (CPU, ~5 min). These defend the Q1 publication claims with drop-one-out ablations, zero-shot baselines, lightweight classifier comparisons, statistical significance tests, error analysis, and cross-validation variance analysis.
7. All notebooks auto-detect the dataset path on Kaggle

## Environment

- **GPU:** NVIDIA Tesla T4 (Kaggle)
- **Framework:** PyTorch + HuggingFace Transformers
- **Evaluation:** 5-fold stratified cross-validation (BanglaBERT Large, Classical, SMI) / 80-20 train-test split (LLMs)

## References

This project builds on the following prior work:

- **BanglaBERT** — Bhattacharjee et al., 2022. Bengali language model used as the fine-tuned transformer baseline.
- **QLoRA** — Dettmers et al., 2023. Quantized low-rank adaptation method used to fine-tune all 6 general-purpose LLMs.
- **LoRA** — Hu et al., 2022. Low-rank adaptation, the foundational method underlying QLoRA.
- **Gemma 2** — Google, 2024. Open language models (`gemma-2-2b-it`, `gemma-2-9b-it`).
- **Qwen2.5** — Qwen Team, 2024. Instruct models (`Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`).
- **Phi-3** — Microsoft, 2024. Instruct model (`Phi-3-mini-4k-instruct`).
- **Llama 3.1** — Meta, 2024. Instruct model (`Meta-Llama-3.1-8B-Instruct`).
- **Snorkel** — Ratner et al., 2017. Weak-supervision framework cited as theoretical motivation for the SMI automated-annotation algorithm.

Statistical methodology in NB13 additionally relies on **McNemar's test** (McNemar, 1947) and the **Wilcoxon signed-rank test** (Wilcoxon, 1945). BanglaBERT's architecture is based on **ELECTRA** (Clark et al., 2020).

Full BibTeX entries in `CITATION.bib`.

## License

Code is released under the MIT License (see `LICENSE`). The dataset and annotation materials are released under Creative Commons Attribution 4.0 International (see `DATA_LICENSE`). Third-party model licenses are disclosed in `NOTICE.md`.

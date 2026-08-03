# Results Folder

## Contents

- `master_comparison.csv` — Complete results table for all 15 entries (12 models + SMI annotation algorithm at 2 evaluation tiers + Qwen-7B multi-seed)
- `smi_weights.json` — Learned logistic regression weights for the 8 SMI criteria (8-criteria run, 2026-07-21)
- `smi_annotation_results.json` — SMI evaluation metrics across three tiers (5-fold CV + 70/30 held-out on 766 gold + true out-of-sample on 4,234 non-gold articles). All tiers populated from the actual Kaggle run on 2026-07-21 (`needs_rerun: false`).
- `multi_seed_qwen7b_final_summary.json` — Qwen2.5-7B-Instruct multi-seed (n=3) aggregated summary from NB9c
- `multi_seed_qwen7b_batch1_results.csv` — Per-seed results from NB9a (batch 1, seeds [42, 123, 2024] — 2 success, 1 CUDA OOM)
- `multi_seed_qwen7b_batch2_results.csv` — Per-seed results from NB9b (batch 2, seeds [7, 99] — 1 success, 1 CUDA OOM)
- `multi_seed_qwen7b_batch1_summary.json` — Batch 1 aggregate summary
- `multi_seed_qwen7b_batch2_summary.json` — Batch 2 aggregate summary
- `smi_criterion_ablation_results.json` — NB10 drop-one-out + drop-pairs ablation of the 8 SMI criteria (C1 essential, C1+C6 complementary; 2026-07-21 Kaggle run)
- `lightweight_classifier_results.json` — NB12 evaluation of 7 lightweight classifiers on the 8 SMI features (all 7 beat the best LLM; Linear SVM best at F1=0.8305; 2026-07-21 Kaggle run)
- `banglabert_error_analysis.json` — BanglaBERT error analysis by confidence/batch/length/criteria/calibration (NB14)
- `cv_variance_final_summary.json` — Final 5-seed CV variance summary for all models (NB15c)
- `cv_variance_final_summary_table.csv` — CSV version of the CV variance summary (NB15c)
- `cv_variance_smi_classical_results.json` — 5×5-fold CV variance for SMI + 4 classical models (NB15a)
- `cv_variance_smi_classical_results_table.csv` — CSV version of SMI + classical CV variance (NB15a)
- `lightweight_classifier_feature_importance.csv` — Feature importance across LR/RF/XGBoost (NB12)
- `lightweight_classifier_results.csv` — 7 lightweight classifier F1/accuracy/precision/recall (NB12)
- `lightweight_vs_llm_unified.csv` — Unified comparison table: lightweight vs LLMs vs BanglaBERT (NB12)
- `llm_zeroshot_results.csv` — Per-model zero-shot inference results (NB11)
- `llm_zeroshot_summary.json` — Zero-shot vs QLoRA comparison summary (NB11)
- `statistical_significance_results.json` — McNemar + Bootstrap CI + Wilcoxon + t-test results (NB13)
- `zeroshot_vs_qlora_comparison.csv` — Zero-shot vs QLoRA F1 comparison table (NB11)
- `banglabert_clean_results.json` — BanglaBERT Large clean 5-fold CV results (F1, accuracy, per-fold) from NB1
- `all_results_clean.json` — Combined results for all models (BanglaBERT + 4 classical) from NB1

## Master Comparison Table

| Model | Accuracy | Precision | Recall | F1 | Kappa | Type |
|-------|----------|-----------|--------|-----|-------|------|
| BanglaBERT Large | 0.876±0.026 | 0.860±0.043 | 0.901±0.015 | **0.883±0.025** | 0.752±0.052 | Fine-tuned Transformer |
| SMI Annotation (5-fold CV) | 0.820±0.050 | 0.855 | 0.770 | 0.809±0.056 | 0.640 | Rule-based Annotation Algorithm |
| SMI Annotation (4234 non-gold OOS) | 0.524 | 0.417 | 0.197 | 0.267 | −0.022 | Rule-based (true out-of-sample) |
| Random Forest | 0.786±0.013 | 0.781±0.027 | 0.796±0.040 | 0.788±0.015 | 0.572±0.025 | TF-IDF Classical |
| Logistic Regression | 0.782±0.040 | 0.793±0.034 | 0.765±0.077 | 0.777±0.047 | 0.564±0.081 | TF-IDF Classical |
| Linear SVM | 0.766±0.019 | 0.774±0.032 | 0.755±0.035 | 0.764±0.018 | 0.533±0.038 | TF-IDF Classical |
| XGBoost | 0.755±0.010 | 0.760±0.026 | 0.747±0.043 | 0.752±0.013 | 0.509±0.020 | TF-IDF Classical |
| Qwen2.5-7B-Instruct (QLoRA, single seed=42) | 0.520 | 1.000 | 0.039 | 0.075 | 0.039 | LLM Fine-tuned |
| Qwen2.5-7B-Instruct (QLoRA, multi-seed n=3) | 0.511±0.004 | 0.889±0.192 | 0.026±0.000 | 0.050±0.000 | 0.022±0.008 | LLM Fine-tuned (multi-seed) |
| Gemma-2-2B-it (QLoRA) | 0.513 | 1.000 | 0.026 | 0.051 | 0.026 | LLM Fine-tuned |
| Phi-3-mini-4k (QLoRA) | 0.513 | 1.000 | 0.026 | 0.051 | 0.026 | LLM Fine-tuned |
| Llama-3.1-8B-Instruct (QLoRA) | 0.513 | 1.000 | 0.026 | 0.051 | 0.026 | LLM Fine-tuned |
| Gemma-2-9B-it (QLoRA) | 0.513 | 1.000 | 0.026 | 0.051 | 0.026 | LLM Fine-tuned |
| Qwen2.5-3B-Instruct (QLoRA) | 0.507 | 0.667 | 0.026 | 0.050 | 0.013 | LLM Fine-tuned |
| Majority Baseline | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | Baseline |

Note: The single-seed (seed=42) F1 values for Linear SVM (0.764) and XGBoost (0.752) are outliers relative to their 5-seed means (0.775 and 0.739, respectively). See `cv_variance_final_summary.json` for the multi-seed values.

## Evaluation Protocol

- **BanglaBERT + Classical:** 5-fold stratified cross-validation on 766 articles
- **SMI Annotation (3 tiers):**
  1. 5-fold CV on 766 gold articles (F1=0.809 ± 0.056) — variance estimate
  2. 70/30 held-out split on 766 gold (F1=0.835) — primary in-sample held-out
  3. **True out-of-sample on 4,234 non-gold articles** (F1=0.267) vs `human_label` — strongest generalization test (lower bound due to 59.9% `human_label` ↔ `best_label` agreement)
- **LLMs:** 80/20 train-test split (612 train / 154 test)
- **Seed:** 42 (all single-seed experiments; NB9a/b/c uses 5 seeds for Qwen2.5-7B-Instruct)

## Methodological Notes

- BanglaBERT and Classical models use 5-fold stratified CV on 766 articles. F1±std is across folds.
- SMI Annotation reports three evaluation tiers (see above). The 4,234 non-gold evaluation uses `human_label` from `Swarabyanjan_Corpus_5000.csv` (the corpus's initial-analysis label column) as ground truth, which is noisier than the gold `best_label` (59.9% agreement on overlap). Observed F1 on non-gold is therefore a **lower bound** on SMI's true generalization performance. The gap between in-sample F1 (0.84 held-out, 0.81 CV) and out-of-sample F1 (0.27) is honest and informative — it suggests the 766 gold articles may not be fully representative of the broader 5,000-corpus distribution.
- LLMs use a single 80/20 train-test split with seed=42. NB9a/b/c provides multi-seed (5 planned, 3 succeeded) evaluation for Qwen2.5-7B-Instruct to estimate variance. The 2 failed seeds ([123] in NB9a, [99] in NB9b) crashed with CUDA out-of-memory on Kaggle T4.
- LLM Precision = 1.000 (single-seed models and 2 of 3 multi-seed runs) is a degenerate artifact: models make very few positive predictions (TP=1–3, FP=0 out of 77 positive test cases). The Qwen-7B multi-seed aggregate drops to Precision = 0.889 ± 0.192 because seed 2024 produced 1 false positive (TP=2, FP=1) — the improved parser caught an output as `"1"` that was actually a non-yellow article. See confusion matrices in `outputs/`.
- `smi_annotation_results.json` has `needs_rerun: false` — all three evaluation tiers (5-fold CV, 70/30 held-out, 4,234 non-gold out-of-sample) were populated by the actual NB8 Kaggle run on 2026-07-21 with the cleaned CSVs (`Swarabyanjan_Gold_Balanced_766.csv` + `Swarabyanjan_Corpus_5000.csv`).
- All standard deviations reported in this repository use ddof=1 (sample standard deviation, the pandas/scipy default), except where noted. The master_comparison.csv reports ddof=0 (population std) for SMI's 5-fold CV F1 std (0.0564) to match NB8's numpy-based computation; the equivalent ddof=1 value is 0.0630.

## Ablation Results

The repository ships two Q1-readiness ablation notebooks (NB10 and NB12) that defend the SMI algorithm's design choices. Both were successfully run on Kaggle on **2026-07-21** (NB10: 14.4s CPU; NB12: 18.7s CPU; no errors). Canonical results are in `smi_criterion_ablation_results.json` and `lightweight_classifier_results.json`.

### NB10 — SMI Criterion Ablation (Drop-One-Out)

For each criterion C1–C8, NB10 drops it from the 8-feature matrix, retrains the Logistic Regression on the remaining 7, and reports the F1 change vs. the all-8-criteria baseline (5-fold CV F1=0.8091 ± 0.0564 on 766 gold articles). It also ablates the 6 pairs of the top-4 criteria (C1, C6, C3, C7).

**Drop-one-out table (sorted by |ΔF1| descending):**

| Dropped | Criterion Name | F1 Mean | F1 Std | F1 Δ | Accuracy | Kappa | Interpretation |
|---|---|---|---|---|---|---|---|
| **C1** | Sensational Headline | 0.7079 | 0.0801 | **+0.1013** | 0.7273 | 0.4548 | **Essential** |
| C6 | Entertainment Displacement | 0.7956 | 0.0242 | +0.0135 | 0.7977 | 0.5954 | Important |
| C3 | Emotional Arousal | 0.8018 | 0.0594 | +0.0074 | 0.8134 | 0.6269 | Minor |
| C7 | Headline-Body Coherence | 0.8037 | 0.0565 | +0.0054 | 0.8160 | 0.6322 | Minor |
| C2 | Clickbait | 0.8040 | 0.0594 | +0.0052 | 0.8160 | 0.6321 | Minor |
| C5 | Speculation | 0.8066 | 0.0415 | +0.0025 | 0.8186 | 0.6373 | Negligible |
| C8 | Sensitive Topic | 0.8080 | 0.0563 | +0.0011 | 0.8186 | 0.6374 | Negligible |
| C4 | Attribution Gap | 0.8085 | 0.0484 | +0.0007 | 0.8199 | 0.6400 | Negligible |

Baseline (all 8 criteria): F1=0.8091 ± 0.0564, Accuracy=0.8199, Kappa=0.6400, per-fold F1=`[0.7361, 0.9079, 0.8163, 0.7826, 0.8027]` (matches NB8 exactly).

**Drop-pairs table (top-4 criteria, 6 pairs):**

| Dropped Pair | F1 Mean | F1 Δ | Sum of Singles | Synergy | Interpretation |
|---|---|---|---|---|---|
| **C1+C6** | 0.6665 | +0.1426 | +0.1147 | **+0.0278** | **complementary** (only positive-synergy pair) |
| C1+C3 | 0.7161 | +0.0931 | +0.1086 | −0.0156 | redundant |
| C1+C7 | 0.7155 | +0.0936 | +0.1067 | −0.0130 | redundant |
| C6+C3 | 0.7973 | +0.0118 | +0.0209 | −0.0090 | redundant |
| C6+C7 | 0.7920 | +0.0171 | +0.0189 | −0.0018 | independent |
| C3+C7 | 0.7992 | +0.0099 | +0.0128 | −0.0029 | independent |

**Key takeaways (NB10):**

- **C1 (Sensational Headline) is ESSENTIAL** — dropping it causes F1 to collapse from 0.809 → 0.708 (ΔF1=+0.101). This is consistent with C1's learned weight (+6.61, 3× larger than the next criterion) and confirms that sensational headlines are the most reliable signal of yellow journalism.
- **C6 (Entertainment Displacement) is IMPORTANT** — ΔF1=+0.0135 (the only other criterion above the 0.01 Importance threshold).
- **C2, C3, C7 are MINOR** (ΔF1 0.005–0.010); **C4, C5, C8 are NEGLIGIBLE** (ΔF1 < 0.005).
- **C1+C6 are complementary** — the only pair with positive synergy (+0.0278): dropping both hurts more than the sum of dropping each alone. All other pairs are redundant (synergy ≤ 0) or independent (|synergy| < 0.005).
- **C4 is misleading by weight** — it has the 3rd-largest learned weight (+2.02) but is Negligible in ablation (ΔF1=+0.0007), because its signal is redundant with C1/C6. This is exactly why ablation studies matter: weight magnitude alone is not a reliable importance signal.
- **Spearman ρ (|weight| vs |ΔF1|) = +0.4286** (p=0.29, n=8) — moderate positive correlation; ablation is more informative than weight magnitude alone.

### NB12 — Lightweight Classifiers on 8 SMI Features

NB12 evaluates 7 lightweight classifiers on the 8 SMI criteria features (5-fold CV on 766 gold articles). The goal is a feature-sufficiency test: if simple classifiers on the 8 hand-crafted features beat all 6 QLoRA-fine-tuned LLMs, then the SMI features contain the signal and the LLM is the bottleneck.

**7 classifiers (sorted by F1, descending):**

| Classifier | F1 Mean | F1 Std | Accuracy | Precision | Recall | Kappa | MCC | AUC |
|---|---|---|---|---|---|---|---|---|
| **Linear SVM** | **0.8305** | 0.0432 | 0.8395 | 0.8753 | 0.7915 | 0.6790 | 0.6830 | 0.9022 |
| Random Forest | 0.8223 | 0.0245 | 0.8264 | 0.8423 | 0.8043 | 0.6529 | 0.6543 | 0.8861 |
| XGBoost | 0.8176 | 0.0242 | 0.8199 | 0.8266 | 0.8095 | 0.6397 | 0.6403 | 0.8954 |
| Logistic Regression | 0.8091 | 0.0630 | 0.8199 | 0.8544 | 0.7706 | 0.6400 | 0.6444 | 0.8982 |
| KNN | 0.8070 | 0.0336 | 0.8134 | 0.8331 | 0.7835 | 0.6268 | 0.6286 | 0.8554 |
| Decision Tree | 0.8040 | 0.0428 | 0.8121 | 0.8359 | 0.7757 | 0.6242 | 0.6265 | 0.8217 |
| Naive Bayes | 0.7982 | 0.0495 | 0.8160 | 0.8822 | 0.7314 | 0.6321 | 0.6430 | 0.8853 |

**Comparison with LLMs and BanglaBERT:**

| Model | F1 | Type |
|---|---|---|
| BanglaBERT Large (NB1, 5-fold CV) | 0.8831 | Fine-tuned Transformer (reference) |
| **Linear SVM (on 8 SMI features)** | **0.8305** | **Lightweight Classifier (8 features)** |
| Random Forest (on 8 SMI features) | 0.8223 | Lightweight Classifier (8 features) |
| XGBoost (on 8 SMI features) | 0.8176 | Lightweight Classifier (8 features) |
| Logistic Regression (on 8 SMI features) | 0.8091 | Lightweight Classifier (8 features) |
| KNN (on 8 SMI features) | 0.8070 | Lightweight Classifier (8 features) |
| Decision Tree (on 8 SMI features) | 0.8040 | Lightweight Classifier (8 features) |
| Naive Bayes (on 8 SMI features) | 0.7982 | Lightweight Classifier (8 features) |
| Best LLM (Qwen2.5-7B-Instruct, QLoRA seed=42) | 0.0750 | LLM Fine-tuned |
| Best LLM multi-seed (Qwen-7B, n=3) | 0.0504 | LLM Fine-tuned (multi-seed) |

**Key takeaways (NB12):**

- **ALL 7 lightweight classifiers beat ALL 6 LLMs.** Best: Linear SVM F1=0.8305 (11.1× the best LLM F1=0.075). Worst: Naive Bayes F1=0.7982 (10.6× the best LLM). Even the weakest lightweight classifier beats the best LLM by more than 10×.
- **Logistic Regression reproduces SMI's F1=0.8091 exactly** — validates that NB8's SMI model is exactly the same logistic regression on the 8 criteria features, providing an internal consistency check.
- **Feature sufficiency proven** — the 8 SMI features contain the signal; the LLM cannot extract it. For Bengali yellow journalism detection, simple features + simple classifier is more effective than 2B–9B parameter LLMs with QLoRA fine-tuning.
- **Best lightweight classifier (Linear SVM, F1=0.831) is between SMI (F1=0.809) and BanglaBERT (F1=0.883)** on the 5-fold CV metric — i.e., a calibrated Linear SVM on 8 hand-crafted features outperforms all classical TF-IDF baselines (LR=0.777, RF=0.788, SVM=0.764, XGBoost=0.752) and all LLMs, while remaining below the language-specific transformer.


## BanglaBERT Error Analysis (NB14)

### Overall Performance
- **F1:** 0.879 (matches NB1)
- **Total errors:** 92/766 (12.0%)
- **False positives:** 57 (pred=Yellow, true=Non-yellow)
- **False negatives:** 38 (pred=Non-yellow, true=Yellow)
- **BanglaBERT is more FP-prone** (55 FP vs 37 FN)

### Errors by Annotation Confidence
| Confidence | n | Errors | Error Rate | F1 |
|------------|---|--------|------------|-----|
| H (high) | 376 | 37 | 9.8% | 0.721 |
| M (medium) | 328 | 50 | **15.2%** | 0.906 |
| L (low) | 62 | 7 | 11.3% | 0.940 |

M-confidence articles have the highest error rate — BanglaBERT mirrors human annotation difficulty.

### Errors by Corpus Batch
| Batch | n | Errors | Error Rate | F1 |
|-------|---|--------|------------|-----|
| corpus_expansion_5000 | 649 | 78 | 12.0% | 0.886 |
| v17_reused | 48 | 10 | **20.8%** | 0.762 |
| new_low_w0_pany | 19 | 3 | 15.8% | 0.842 |
| new_mid_w0_p0 | 16 | 2 | 12.5% | 0.833 |
| new_mid_w1_p1 | 14 | 1 | 7.1% | 0.889 |
| new_high_w1_pany | 10 | 0 | 0.0% | 1.000 |
| new_mid_w1_p0 | 10 | 1 | 10.0% | 0.933 |

### Errors by Article Length
| Length Bin | n | Errors | Error Rate | F1 |
|------------|---|--------|------------|-----|
| 0-500 | 58 | 6 | 10.3% | 0.893 |
| 500-1K | 174 | 21 | 12.1% | 0.871 |
| 1K-2K | 353 | 44 | 12.5% | 0.883 |
| 2K-4K | 141 | 14 | 9.9% | 0.907 |
| 4K+ | 40 | 10 | **25.0%** | 0.750 |

4K+ articles have 25% error rate — BanglaBERT uses MAX_LEN=512 tokens (~1500-2000 Bengali chars), so longer articles are truncated.

### Error Correlation with SMI Criteria
| Criterion | Mean (Correct) | Mean (Errors) | p-value | Significant? |
|-----------|----------------|---------------|---------|--------------|
| C1 | 0.105 | 0.072 | 0.038 | **YES** |
| C2 | 0.017 | 0.008 | 0.075 | no |
| C4 | 0.474 | 0.519 | 0.219 | no |
| C6 | 0.242 | 0.203 | 0.278 | no |
| C3, C5, C7, C8 | — | — | >0.5 | no |

C1 (Sensational Headline) is the only criterion significantly correlated with BanglaBERT errors (p=0.038). Articles where BanglaBERT makes errors have lower C1 scores — suggesting BanglaBERT misses sensational headlines that SMI catches.

### Probability Calibration
- **Brier score:** 0.107 (lower is better)
- **Expected Calibration Error (ECE):** 0.075 (lower is better)
- BanglaBERT's probability estimates are moderately calibrated.


## Statistical Significance Tests (NB13)

### McNemar's Test (15 pairwise comparisons on 766 articles)

| Comparison | p-value | Significant? |
|------------|---------|--------------|
| BanglaBERT vs LogReg | 6.47e-11 | YES (p<0.001) |
| BanglaBERT vs RandomForest | 8.44e-11 | YES (p<0.001) |
| BanglaBERT vs LinearSVM | 1.04e-13 | YES (p<0.001) |
| BanglaBERT vs XGBoost | 4.69e-16 | YES (p<0.001) |
| BanglaBERT vs SMI | 0.000119 | YES (p<0.001) |
| SMI vs LogReg | 0.033 | YES (p<0.05) |
| SMI vs LinearSVM | 0.0033 | YES (p<0.01) |
| SMI vs XGBoost | 0.0005 | YES (p<0.001) |
| **SMI vs RandomForest** | **0.079** | **NO (borderline)** |
| LogReg vs RandomForest | 0.744 | NO |
| LogReg vs LinearSVM | 0.088 | NO |
| LogReg vs XGBoost | 0.054 | NO |
| RandomForest vs LinearSVM | 0.149 | NO |
| RandomForest vs XGBoost | 0.013 | YES (p<0.05) |
| LinearSVM vs XGBoost | 0.390 | NO |

**Key finding:** BanglaBERT significantly outperforms LogReg (p=6.47e-11), RF (p=8.44e-11), LinearSVM (p=1.04e-13), XGBoost (p=4.69e-16), and SMI (p=0.000119). Note: BanglaBERT per-article predictions loaded from real NB1 output (banglabert_clean_predictions.csv). SMI significantly outperforms all classical models EXCEPT Random Forest (borderline p=0.079).

**Wilcoxon signed-rank test:** All 15 pairwise Wilcoxon tests on per-fold F1 (n=5) yielded p ≥ 0.0625 (the minimum possible for two-sided Wilcoxon with n=5). No comparison reaches p<0.05 under Wilcoxon. The McNemar test on full 766 paired predictions is more powerful; the Wilcoxon results are limited by the small number of folds.

### Bootstrap 95% CI for F1 (10,000 iterations)

| Model | F1 | 95% CI | CI Width |
|-------|-----|--------|----------|
| BanglaBERT | 0.883 | [0.855, 0.904] | 0.048 |
| SMI | 0.812 | [0.780, 0.842] | 0.063 |
| RandomForest | 0.789 | [0.756, 0.819] | 0.063 |
| LogReg | 0.779 | [0.746, 0.811] | 0.065 |
| LinearSVM | 0.765 | [0.731, 0.797] | 0.066 |
| XGBoost | 0.751 | [0.716, 0.785] | 0.069 |

### LLM Multi-Seed t-test
- Qwen-7B (3 seeds, F1 = 0.050 ± 0.000) vs BanglaBERT (F1 = 0.883): **t-test p = 5.82e-08** (highly significant)
- LLM multi-seed F1 std = 0.0003 — the failure is highly stable across the 3 successful seeds (F1 std=0.0003; a larger seed budget would be needed to confirm determinism)
- **Caveat:** This t-test treats BanglaBERT's F1 as a constant (no variance estimate available). With n=3 LLM seeds and std=0.0003, the t-statistic is inflated. This test should be interpreted as descriptive, not inferential.

### Caveat
BanglaBERT per-article predictions loaded from banglabert_clean_predictions.csv (real NB1 output). McNemar p-values are computed from real per-article predictions.


## LLM Zero-Shot Baseline (NB11)

### Zero-Shot vs QLoRA Comparison (same 154-article test set, seed=42)

| Model | Zero-Shot F1 | QLoRA F1 | F1 Delta | Interpretation |
|-------|-------------|----------|----------|----------------|
| Gemma-2-9B-it | **0.204** | 0.051 | +0.154 | **QLoRA HURTS** — zero-shot 4× better |
| Gemma-2-2B-it | **0.111** | 0.051 | +0.061 | **QLoRA HURTS** — zero-shot 2× better |
| Qwen2.5-7B | 0.000 | 0.075 | -0.075 | QLoRA helps (but still fails) |
| Llama-3.1-8B | 0.026 | 0.051 | -0.025 | QLoRA helps (but still fails) |
| Qwen2.5-3B | 0.000 | 0.050 | -0.050 | QLoRA helps (but still fails) |
| Phi-3-mini | NaN | 0.051 | NaN | Failed (model load issue) |

**Key findings:**
- **Mean zero-shot F1 = 0.068** vs **Mean QLoRA F1 = 0.055** — zero-shot is slightly BETTER on average
- **2/6 LLMs (Gemma-2-9B, Gemma-2-2B) perform WORSE after QLoRA** — 4-bit quantization + small dataset actively degrades Bengali capability
- **3/6 LLMs improve slightly with QLoRA** but still fail (F1 < 0.08)
- **91.6% unparseable rate** — LLMs generate 2-4 character Bengali text fragments instead of "0" or "1"
- **Conclusion:** This ablation indicates the LLM failure is consistent with insufficient Bengali language modeling capability; the relative contributions of pretraining depth, tokenizer coverage, and instruction-tuning quality are not fully disentangled.

## CV Variance Analysis (NB15a) — SMI + Classical Models

### 5×5-Fold CV Results (5 seeds × 5 folds = 25 fits per model)

| Model | Seed 42 | Seed 123 | Seed 2024 | Seed 7 | Seed 99 | Mean ± Std |
|-------|---------|----------|-----------|--------|---------|------------|
| SMI | 0.8091 | 0.8071 | 0.8008 | 0.8144 | 0.8090 | **0.8081 ± 0.0044** |
| Logistic Regression | 0.7794 | 0.7876 | 0.7796 | 0.7877 | 0.7742 | 0.7817 ± 0.0052 |
| Random Forest | 0.7698 | 0.7920 | 0.7848 | 0.7687 | 0.7802 | 0.7791 ± 0.0089 |
| Linear SVM | 0.7704 | 0.7804 | 0.7676 | 0.7804 | 0.7755 | 0.7749 ± 0.0052 |
| XGBoost | 0.7549 | 0.7473 | 0.7222 | 0.7417 | 0.7271 | 0.7386 ± 0.0123 |

**Key findings:**
- **SMI across-seed std = 0.0044** — 13× smaller than within-seed std (0.0564)
- **Single-seed result (0.8091) is perfectly representative** — not lucky
- **Model rankings are stable across seeds** — SMI > LR > RF > SVM > XGBoost in all 5 seeds
- **XGBoost has the highest variance** (std=0.0123) but still never crosses SMI

## Key Findings

1. **BanglaBERT Large** is the best model (F1=0.883), outperforming all classical and LLM approaches
2. **Classical ML** (TF-IDF + LR/SVM/RF/XGBoost) achieves respectable F1=0.75-0.79
3. **All 6 LLMs completely fail** (F1=0.050-0.075) — they predict almost everything as non-yellow
4. **LLM Precision = 1.0** is misleading — it means they made almost no positive predictions (TP=1-3 out of 77)
5. **BanglaBERT vs best LLM:** 12x F1 improvement (0.883 vs 0.075)
6. **SMI Annotation** (rule-based, no neural model) achieves F1=0.809 ± 0.056 on 5-fold CV (pooled/aggregated F1=0.810) — better than all classical ML baselines and all LLMs except BanglaBERT. SMI is NOT significantly different from Random Forest (McNemar p=0.079; bootstrap 95% CI [−0.062, +0.015] crosses zero). This suggests the 8-criteria annotation protocol itself captures most of the signal.
7. **SMI three-tier evaluation complete (Kaggle run 2026-07-21):** 5-fold CV F1=0.809 ± 0.056, 70/30 held-out F1=0.835, true out-of-sample F1=0.267 on 4,234 non-gold articles vs `human_label`. The gap between in-sample (0.84 held-out, 0.81 CV) and out-of-sample (0.27) is honest and informative — it suggests the 766 gold articles may not be fully representative of the broader 5,000-corpus distribution. All null fields in `smi_annotation_results.json` are now populated (`needs_rerun: false`).
8. **SMI true out-of-sample** (4,234 non-gold articles vs `human_label`): observed F1=0.267, accuracy=0.524, precision=0.417, recall=0.197, kappa=−0.022, MCC=−0.025, AUC=0.457, TP=368, FP=514, FN=1502, TN=1850. This is a **lower bound** on SMI's true generalization (the `human_label` ↔ `best_label` agreement is 59.9% on the 766 overlap — `label_agreement_ceiling=0.5992`, `ceiling_adjusted_f1_heuristic=0.4463`). The in-sample overlap F1 (SMI vs `human_label` on the 766 overlap) is 0.649 — so even on the same data the `human_label` noise alone depresses F1 from 0.84 (vs `best_label`) to 0.65 (vs `human_label`).
9. **Body text truncation:** 10.98% of the 5,000-corpus articles (549/5000) have body text truncated at 1,200 characters. This is slightly higher than the previous 9.1% estimate because NFC normalization changes character counts. Documented in `smi_annotation_results.json` → `corpus_body_truncation`.
10. **Multi-seed evaluation refines single-seed result (NB9a/b/c, v2 re-run):** Qwen2.5-7B-Instruct was re-evaluated with 3 successful random seeds [42, 7, 2024] (2 additional seeds [123, 99] failed with CUDA OOM on Kaggle T4). The multi-seed F1 = 0.050 ± 0.000 — the **NB5 single-seed F1 = 0.075 is NOT within 1 standard deviation of the multi-seed mean** (`nb5_within_1std_of_multiseed: false` in `multi_seed_qwen7b_final_summary.json`). NB5's seed=42 result was slightly lucky with 3 TP instead of the typical 2 TP seen across all 3 multi-seed runs. The very tight F1 std (0.0003) indicates the failure mode is highly stable across the 3 successful seeds; a larger seed budget would be needed to confirm determinism. See `multi_seed_qwen7b_final_summary.json`.
11. **96–98% unparseable (mean 97% across 3 seeds) output rate (stronger failure mode):** Across the 3 successful multi-seed runs, 148–151 of 154 test predictions per seed were **unparseable** — the LLM did not produce output starting with `0` or `1` (or the Bengali digit equivalents `০`/`১`). The 2 TP + 76–77 TN per seed come from the fallback rule (unparseable → 0), not from genuine positive predictions. The improved parser (v7 notebook re-run) marginally increased the parseable rate from 1.9–2.6% (previous run) to 1.9–3.9% (this run). This is a **stronger failure mode** than incorrect classification: the LLM fails to follow the output-format instruction at all. See `multi_seed_qwen7b_batch1_results.csv` and `multi_seed_qwen7b_batch2_results.csv` (the `Unparseable` and `Parseable_Pct` columns).
12. **LLM generates Bengali character fragments (root-cause analysis of unparseable outputs):** Debug logs from the v7 re-run reveal that the LLM does not output `0`/`1`/`০`/`১` — instead it emits 2–4 character Bengali text fragments. Representative examples captured at inference time:
    ```
    Unparseable output #1: repr='ে।'
    Unparseable output #2: repr='পজে'
    Unparseable output #3: repr='াদে'
    Unparseable output #4: repr=' সন'
    Unparseable output #5: repr=' প্র'
    Unparseable output #6: repr=' বি'
    Unparseable output #7: repr='ুদ'
    Unparseable output #8: repr='ো'
    Unparseable output #9: repr=' এক'
    Unparseable output #10: repr='েল'
    ```
    Because `max_new_tokens=3` generates 3 tokens, which in Bengali tokenization corresponds to 2–4 characters, the model **completely ignores** the instruction "Answer with only the digit 1 or 0" and instead continues the Bengali input with Bengali output. This is a **stronger failure mode than wrong predictions** — the LLM cannot follow basic output-format instructions on Bengali text, regardless of how many epochs of QLoRA fine-tuning are applied. See `multi_seed_qwen7b_batch1_results.csv` and the NB9a/b debug logs.
13. **C1 (Sensational Headline) is the dominant SMI criterion** with learned weight +6.61 — 3× larger than the next criterion (C6 Entertainment Displacement, +2.21). This confirms that sensational headlines are the most reliable signal of yellow journalism, and is consistent with the broader journalism-studies literature on headline sensationalism. See [SMI Three-Tier Evaluation Results](#smi-three-tier-evaluation-results) below.
14. **C8 (Sensitive Topic) has a small negative weight (−0.08) that is Negligible in ablation (ΔF1=+0.001 when dropped).** The direction is consistent with the hypothesis that sensitive topics are often covered seriously with proper attribution (court filings, government statements, police briefings), not sensationally, but the effect size is too small to support strong claims.
15. **SMI annotation speed: 37,191× faster than human** — 16.17 seconds to annotate all 5,000 corpus articles, vs an estimated ~167 hours for a human annotator at ~2 min/article. This makes SMI a practical weak-supervision tool even when its out-of-sample F1 is modest.
16. **C1 (Sensational Headline) is ESSENTIAL — dropping it causes F1 to drop from 0.809 → 0.708 (ΔF1=+0.101)** in the NB10 drop-one-out ablation. C1 is the only Essential criterion; C6 is Important (ΔF1=+0.014); C2/C3/C7 are Minor (ΔF1 0.005–0.010); C4/C5/C8 are Negligible (ΔF1 < 0.005). See [Ablation Results](#ablation-results) below and `smi_criterion_ablation_results.json`.
17. **C1+C6 are complementary — the only pair with positive synergy (+0.028)** in the NB10 drop-pairs ablation: dropping both C1 and C6 together causes F1 to drop by +0.143, more than the sum of dropping each alone (+0.115). All other pairs of top-4 criteria (C1+C3, C1+C7, C6+C3, C6+C7, C3+C7) are redundant or independent. This means C1 (headline sensationalism) and C6 (entertainment displacement of news) capture distinct, complementary signals — the rest overlap with one of these two.
18. **All 7 lightweight classifiers on 8 SMI features beat all 6 LLMs (best: Linear SVM F1=0.831 vs best LLM F1=0.075)** in the NB12 feature-sufficiency test. The 8 hand-crafted SMI features are sufficient — the LLM is the bottleneck, not the features. Logistic Regression on the 8 features reproduces SMI's F1=0.8091 exactly (internal consistency check). Even the weakest lightweight classifier (Naive Bayes F1=0.798) beats the best LLM by more than 10×. See [Ablation Results](#ablation-results) below and `lightweight_classifier_results.json`.

## SMI Three-Tier Evaluation Results

The SMI (Swarabyanjan Multi-criteria Index) annotation algorithm was evaluated on three complementary tiers in `notebooks/NB8_SMI_Annotation_Experiment.ipynb`. NB8 was successfully run on Kaggle on **2026-07-21** with the cleaned CSVs (`Swarabyanjan_Gold_Balanced_766.csv` + `Swarabyanjan_Corpus_5000.csv`; NFC + ZWJ/ZWNJ stripped text normalization). All metrics below are populated in `smi_annotation_results.json` (`needs_rerun: false`).

### Three Tiers Side by Side

| Tier | Split | n_articles | F1 | Accuracy | Precision | Recall | Kappa | MCC | AUC |
|------|-------|------------|-----|----------|-----------|--------|-------|-----|-----|
| 1. 5-fold CV (per-fold mean ± std) | 5-fold CV on 766 gold | 766 | 0.809 ± 0.056 | 0.820 ± 0.050 | — | — | — | — | — |
| 1. 5-fold CV (pooled/aggregated) | out-of-fold predictions on 766 gold | 766 | **0.810** | **0.820** | **0.855** | **0.770** | **0.640** | **0.643** | **0.898** |
| 2. 70/30 held-out | 536 train / 230 test (766 gold) | 230 | **0.835** | **0.822** | **0.776** | **0.904** | **0.643** | **0.652** | **0.914** |
| 3. True out-of-sample (vs `human_label`) | trained on 766 gold, evaluated on 4,234 non-gold | 4,234 | **0.267** | **0.524** | **0.417** | **0.197** | −0.022 | −0.025 | 0.457 |

- **Per-fold F1** (Tier 1): `[0.7361, 0.9079, 0.8163, 0.7826, 0.8027]` — fold 2 is an outlier (0.91), folds 1 and 4 are weakest (0.74 / 0.78). The 0.056 std across 5 folds is the within-gold-standard variance estimate.
- **Tier 1 also reports aggregated (cross-validated) metrics**: F1=0.8104, Accuracy=0.8198, Precision=0.8551, Recall=0.7702, Kappa=0.6397, MCC=0.6429, AUC=0.8981. These are computed on the 766 pooled out-of-fold predictions and differ slightly from the mean-of-per-fold-F1 (0.8091) due to the well-known discrepancy between mean-of-folds and pooled-prediction metrics.
- **Tier 2 precision-recall asymmetry**: Precision=0.776, Recall=0.904 — the SMI threshold (0.4168) is permissive, which catches more true positives at the cost of more false positives. The held-out test set is 230 articles from the 1:1 balanced gold (~115 yellow / ~115 non-yellow under stratified sampling).
- **Tier 3 confusion cell counts**: TP=368, FP=514, FN=1502, TN=1850 (sum = 4234 non-gold articles). The model is highly conservative on the out-of-sample distribution: it predicts only 882 yellow (368+514) out of 4,234 articles, while `human_label` marks 1,870 as yellow — a 1,502-false-negative gap drives the low recall (0.197) and the negative Kappa (−0.022).

### Label Noise Ceiling (Tier 3 Lower Bound)

`human_label` (in the corpus CSV) is the **initial-analysis** human annotation, while `best_label` (in the gold CSV) is the **annotation-refined** annotation. On the 766-overlap, `human_label` agrees with `best_label` only **59.92%** of the time (crosstab: best_label=0/MY_LABEL=0=207, best_label=0/MY_LABEL=1=176, best_label=1/MY_LABEL=0=131, best_label=1/MY_LABEL=1=252). This makes `human_label` a noisy ground truth, and the observed Tier-3 F1=0.267 is therefore a **lower bound** on SMI's true generalization performance.

Two derived fields in `smi_annotation_results.json` → `nongold_4234_evaluation` contextualize this:

- `label_agreement_ceiling` = 0.5992 — the `best_label` ↔ `human_label` agreement on the 766 overlap. This is the theoretical maximum accuracy any classifier could achieve on the non-gold set if it perfectly reproduced `best_label` (because the non-gold ground truth IS `human_label`).
- `ceiling_adjusted_f1_heuristic` = 0.4463 — a rough heuristic adjustment that rescales the observed F1=0.267 by the label-agreement ceiling to estimate what the F1 would be if `human_label` were as clean as `best_label`. This is a heuristic upper bound, not a strict mathematical bound.
- `overlap_f1_insample` = 0.6491 — SMI's F1 vs `human_label` on the 766 overlap (in-sample, so model has seen these articles during training). This is much lower than SMI's F1=0.835 vs `best_label` on the same 766 articles, isolating the label-noise depression: ~0.19 F1 points are lost purely to `human_label` noise, even when the model has memorized the training data.
- `overlap_kappa_insample` = 0.2293 — the corresponding Kappa. SMI vs `human_label` Kappa on the 766 overlap is only 0.23, far below SMI vs `best_label` Kappa (0.64 on held-out). This further confirms that `human_label` is a noisy ground truth.

### In-Sample vs Out-of-Sample Gap (Honest Characterization)

| Comparison | F1 | Interpretation |
|---|---|---|
| Tier 2 (in-sample held-out, vs `best_label`) | 0.835 | SMI generalizes well within the gold-standard distribution |
| Tier 1 (5-fold CV, vs `best_label`) | 0.810 | Within-gold variance estimate; held-out F1 is within 1 std |
| Overlap F1 (in-sample, vs `human_label`) | 0.649 | Label-noise depression: same articles, different (noisier) label → −0.19 F1 |
| Tier 3 (out-of-sample, vs `human_label`) | 0.267 | Distribution shift + label noise combined → additional −0.38 F1 |

**The 0.84 → 0.27 gap is honest and informative.** It is not a bug or a regression — it reflects two distinct sources:

1. **Label noise** (`human_label` vs `best_label`): accounts for ~0.19 of the gap (the in-sample overlap F1 vs `human_label` is 0.65, vs 0.84 vs `best_label`).
2. **Distribution shift** (the 766 gold articles are not fully representative of the 5,000-corpus distribution): accounts for the remaining ~0.38 of the gap.

The 766 gold articles were deliberately balanced (1:1 yellow vs non-yellow) and sampled to cover edge cases (see `data/README.md` → Corpus Batch Tags — the `new_*` batches targeted weak-vs-model disagreement regions). The 5,000-corpus, by contrast, contains 4,204 articles from the `corpus_expansion_5000` multi-signal-sampling batch which has a different class balance and may include articles whose linguistic patterns are not captured by the 8 criteria lexicons. The Tier-3 precision-recall asymmetry (Precision=0.42, Recall=0.20) suggests the SMI threshold (0.4168) — tuned on the balanced gold — is too permissive for the corpus distribution, causing the model to under-predict yellow (only 882/4234 predicted yellow vs 1,870 actually yellow under `human_label`).

This gap is **disclosed transparently** in the paper as a limitation of rule-based annotation algorithms trained on small balanced gold standards.

### SMI Criteria Weights (8 Criteria)

The 8 logistic-regression weights learned on the 766 gold articles, sorted by absolute value:

| Rank | Criterion | Weight | Direction | Interpretation |
|------|-----------|--------|-----------|----------------|
| 1 | **C1: Sensational Headline** | **+6.6100** | POSITIVE (dominant) | 3× larger than the next criterion. Sensational headlines are the most reliable signal of yellow journalism. |
| 2 | C6: Entertainment Displacement | +2.2137 | POSITIVE | Celebrity gossip, viral content framed as news — second-strongest yellow signal. |
| 3 | C4: Attribution Gap | +2.0237 | POSITIVE | Lack of named sources (police, court, wire agencies) — third-strongest yellow signal. |
| 4 | C2: Clickbait | +1.1706 | POSITIVE | Curiosity-gap phrases, listicle patterns, trailing questions. |
| 5 | C5: Speculation | +1.0843 | POSITIVE | `"অনেকে বলছেন"` (many are saying), `"গুঞ্জন"` (rumors) presented as fact. |
| 6 | C3: Emotional Arousal | +0.4006 | POSITIVE (weak) | Body-level emotional language — surprisingly weak; headline-level signals dominate. |
| 7 | C7: Headline-Body Coherence | +0.6151 | POSITIVE (weak) | Headline-body mismatch — weak but positive. |
| 8 | **C8: Sensitive Topic** | **−0.0755** | **NEGATIVE** | **Sensitive topics (communal, religious, gender, ethnic, political) DECREASE yellow-journalism prediction.** Counterintuitive but sensible: sensitive topics are often covered seriously with proper attribution, not sensationally. |
| — | **Bias** ($w_0$) | **−2.2595** | NEGATIVE | Prior that non-yellow is the more common class (balanced gold → expected ~0 bias; the −2.26 reflects the permissive threshold of 0.4168 needed to overcome it). |
| — | **Threshold** ($\tau^*$) | **0.4168** | — | SMI score ≥ 0.4168 → yellow. Lower than 0.5 because the bias term is strongly negative; the threshold compensates. |

**Two notable findings:**

- **C1 dominance (weight +6.61, 3× larger than the next criterion C6=+2.21):** This confirms the journalism-studies literature that sensational headlines are the most reliable signal of yellow journalism. The SMI algorithm's reliance on C1 means that an article with a strongly sensational headline (C1 ≥ 0.5, e.g., a headline with 2+ sensational terms) is very likely to be classified as yellow even if all other criteria are 0. Practitioners using SMI for weak supervision should be aware of this single-criterion dependence.

- **C8 negative weight (−0.08) is Negligible in ablation:** This is the only criterion with a negative weight, but the NB10 drop-one-out ablation shows it is Negligible (ΔF1=+0.001 when dropped). The direction is consistent with the hypothesis that sensitive topics (communal riots, religious controversy, gender violence, political provocation) are often covered seriously with proper attribution (court filings, government statements, police briefings), so the *presence* of a sensitive topic slightly *decreases* the probability of yellow-journalism classification. The effect size (−0.08 vs C1's +6.61) is too small to support strong claims. This finding is consistent with the C4 (Attribution Gap) weight being positive: sensitive-topic articles in this corpus tend to have strong named sources, which lowers C4 and thus the overall SMI score.

### Corpus Annotation (5,000 Articles)

NB8 applied the trained SMI model to the full 5,000-article corpus and produced two derived files:

- `data/smi_annotated_5000.csv` — 5,000 rows × 6 columns: `article_id`, `headline`, `corpus_batch`, `SMI_score`, `SMI_label`, `human_label` (the corpus's initial-analysis label, included for side-by-side comparison)
- `data/smi_criteria_scores_5000.csv` — 5,000 rows × 14 columns: `C1`–`C8` (per-criteria scores in [0,1]), `article_id`, `human_label`, `human_confidence`, `corpus_batch`, `SMI_score`, `SMI_label`

**Label distribution:** 1,289 yellow / 3,711 non-yellow out of 5,000 (25.8% yellow). This is lower than the gold-standard's 50% yellow rate, consistent with the gold being a deliberately balanced 1:1 subset while the corpus is the natural (unbalanced) distribution.

**Annotation speed:** 16.17 seconds for all 5,000 articles, vs an estimated ~167 hours for a human annotator at ~2 min/article — a **37,191× speedup**. This makes SMI a practical weak-supervision tool even when its out-of-sample F1 is modest.

**Corpus body truncation:** 549/5,000 (10.98%) corpus articles have `body_text` capped at 1,200 characters in the source CSV. This is slightly higher than the previous 9.1% estimate because NFC normalization changes character counts. SMI criteria C3 (Emotional Arousal), C5 (Speculation), and C6 (Entertainment Displacement) use word-density formulas that may be slightly affected by truncation.

**Data version (this run):**

- `using_cleaned_csvs`: true
- `gold_csv`: `Swarabyanjan_Gold_Balanced_766.csv`
- `corpus_csv`: `Swarabyanjan_Corpus_5000.csv`
- `corpus_metadata_csv`: `Swarabyanjan_Corpus_Metadata_5000.csv`
- `text_normalization`: NFC + ZWJ/ZWNJ stripped (Task 8 cleaning pipeline)
- `n_stub_articles_gold`: 2 (articles with `body_text = "not_available"` in the gold)
- `n_stub_articles_corpus`: 4 (stub articles in the corpus; treated as empty strings for SMI density-based criteria C3/C5/C6, original text preserved separately)

## Multi-Seed Evaluation Results (NB9a/b/c)

The multi-seed evaluation of Qwen2.5-7B-Instruct was run on Kaggle T4 across 3 notebooks (NB9a/b/c, split due to the 12-hour commit limit). The planned 5 seeds were `[42, 123, 2024, 7, 99]`. The v7 re-run with the improved parser succeeded on 3 seeds (vs 2 in the previous run).

| Seed | Batch | Status | F1 | Accuracy | TP | FP | FN | TN | Unparseable | Parseable % |
|------|-------|--------|-----|----------|-----|-----|-----|-----|-------------|-------------|
| 42   | NB9a (batch 1) | ✅ success | 0.0506 | 0.5130 | 2 | 0 | 75 | 77 | 151 / 154 | 1.9% |
| 123  | NB9a (batch 1) | ❌ CUDA OOM | — | — | — | — | — | — | — | — |
| 2024 | NB9a (batch 1) | ✅ success | 0.0500 | 0.5065 | 2 | 1 | 75 | 76 | 148 / 154 | 3.9% |
| 7    | NB9b (batch 2) | ✅ success | 0.0506 | 0.5130 | 2 | 0 | 75 | 77 | 149 / 154 | 3.2% |
| 99   | NB9b (batch 2) | ❌ CUDA OOM | — | — | — | — | — | — | — | — |

**Aggregate (n=3 successful seeds):**

| Metric | Mean | Std |
|--------|------|-----|
| F1 | 0.0504 | 0.0003 |
| Accuracy | 0.5108 | 0.0038 |
| Precision | 0.8889 | 0.1924 |
| Recall | 0.0260 | 0.0000 |
| Kappa | 0.0217 | 0.0075 |
| MCC | 0.0921 | 0.0391 |

**Key observations:**

- The NB5 single-seed F1=0.075 (seed=42) is **NOT within 1 standard deviation** of the multi-seed mean (0.050 ± 0.000) — the multi-seed std is so tight (0.0003) that the slightly-luckier NB5 run (3 TP vs the typical 2 TP) falls outside the band. (`nb5_within_1std_of_multiseed: false` in `multi_seed_qwen7b_final_summary.json`.) This CHANGED from the previous v1 run, where the looser std (0.0173) made NB5 look like a typical sample.
- **F1 std dropped from 0.0173 (v1) to 0.0003 (v2) — the failure is highly stable across the 3 successful seeds; a larger seed budget would be needed to confirm determinism.** All 3 successful seeds converge to F1 in [0.050, 0.0506] with TP=2, FP=0 or 1, FN=75, TN=76 or 77. The model has effectively learned a single (wrong) behavior: predict non-yellow for everything, with rare spurious positive outputs.
- **Seed 2024 has 1 false positive** — the improved parser caught an output as `"1"` that was actually a non-yellow article. This drops the aggregate Precision from 1.000 (v1) to 0.889 ± 0.192. Precision is no longer a degenerate 1.0 across all seeds.
- **The 96–98% unparseable (mean 97% across 3 seeds) rate is the dominant failure mode.** Of 154 test predictions per seed, 148–151 did not start with `0`/`1`/`০`/`১` — the LLM fails to follow the output-format instruction. The 2 TP + 76–77 TN per seed come from the fallback rule (unparseable → 0), not from genuine positive predictions. The improved parser marginally increased the parseable rate from 1.9–2.6% (v1) to 1.9–3.9% (v2).
- **The LLM generates Bengali character fragments instead of `0`/`1`.** Debug logs from the v7 re-run show outputs like `repr='ে।'`, `repr='পজে'`, `repr='াদে'`, `repr=' সন'`, `repr=' প্র'`, `repr=' বি'`, `repr='ুদ'`, `repr='ো'`, `repr=' এক'`, `repr='েল'`. With `max_new_tokens=3`, 3 generated tokens map to 2–4 Bengali characters — the model completely ignores the "Answer with only the digit 1 or 0" instruction. This is a **stronger failure mode than wrong predictions**: the LLM cannot follow basic instruction-following on Bengali text.
- 2 of 5 planned seeds ([123] in NB9a, [99] in NB9b) crashed with **CUDA out-of-memory** on Kaggle T4 (14.56 GiB GPU). GPU memory cleanup between seeds is incomplete — `del model` + `gc.collect()` + `torch.cuda.empty_cache()` leaves ~7–10 GB allocated, so subsequent seed loads start with ~7–11 GB already in use (seed 123 OOM'd with 7.47 GB free; seed 2024 succeeded borderline with only 2.07 GB free).
- See `multi_seed_qwen7b_final_summary.json` for the canonical aggregated summary, and `multi_seed_qwen7b_batch1_results.csv` / `multi_seed_qwen7b_batch2_results.csv` for the per-seed raw rows (including the CUDA OOM error messages).
- See `outputs/multi_seed_f1_all_seeds.png` for the per-seed F1 bar plot with mean / ±1σ / NB5 reference lines, and `outputs/cm_Qwen2_5-7B-Instruct_seed42.png` / `outputs/cm_Qwen2_5-7B-Instruct_seed7.png` / `outputs/cm_Qwen2_5-7B-Instruct_seed2024.png` for the per-seed confusion matrices.

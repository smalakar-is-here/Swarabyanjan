# Outputs Folder

## Contents

Confusion matrix images for each LLM model (generated from notebook outputs), plus the multi-seed evaluation artifacts (per-seed confusion matrices and the F1 bar plot from NB9a/b/c).

| File | Model | TP | FP | FN | TN |
|------|-------|----|----|----|----|
| `cm_Gemma-2-2B-it.png` | Gemma-2-2B-it | 2 | 0 | 75 | 77 |
| `cm_Qwen2_5-3B-Instruct.png` | Qwen2.5-3B-Instruct | 2 | 1 | 75 | 76 |
| `cm_Phi-3-mini-4k.png` | Phi-3-mini-4k | 2 | 0 | 75 | 77 |
| `cm_Qwen2_5-7B-Instruct.png` | Qwen2.5-7B-Instruct (NB5, seed=42) | 3 | 0 | 74 | 77 |
| `cm_Llama-3_1-8B-Instruct.png` | Llama-3.1-8B-Instruct | 2 | 0 | 75 | 77 |
| `cm_Gemma-2-9B-it.png` | Gemma-2-9B-it | 2 | 0 | 75 | 77 |
| `cm_Qwen2_5-7B-Instruct_seed42.png` | Qwen2.5-7B-Instruct — multi-seed (NB9a, seed=42) | 2 | 0 | 75 | 77 |
| `cm_Qwen2_5-7B-Instruct_seed7.png` | Qwen2.5-7B-Instruct — multi-seed (NB9b, seed=7) | 2 | 0 | 75 | 77 |
| `cm_Qwen2_5-7B-Instruct_seed2024.png` | Qwen2.5-7B-Instruct — multi-seed (NB9a, seed=2024) | 2 | 1 | 75 | 76 |
| `multi_seed_f1_all_seeds.png` | Qwen2.5-7B-Instruct — multi-seed F1 per seed (NB9c, n=3) | — | — | — | — |
| `banglabert_error_analysis.png` | 4-panel error analysis (confidence, batch, length, calibration) (NB14) | — | — | — | — |
| `cm_zeroshot_Gemma-2-2B-it.png` | Zero-shot confusion matrix for Gemma-2-2B-it (NB11) | — | — | — | — |
| `cm_zeroshot_Gemma-2-9B-it.png` | Zero-shot confusion matrix for Gemma-2-9B-it (NB11) | — | — | — | — |
| `cm_zeroshot_Llama-3_1-8B-Instruct.png` | Zero-shot confusion matrix for Llama-3.1-8B-Instruct (NB11) | — | — | — | — |
| `cm_zeroshot_Qwen2_5-3B-Instruct.png` | Zero-shot confusion matrix for Qwen2.5-3B-Instruct (NB11) | — | — | — | — |
| `cm_zeroshot_Qwen2_5-7B-Instruct.png` | Zero-shot confusion matrix for Qwen2.5-7B-Instruct (NB11) | — | — | — | — |
| `cv_variance_final_boxplot.png` | Box plot of F1 across 5 seeds for all models (NB15c) | — | — | — | — |
| `cv_variance_smi_classical_boxplot.png` | Box plot of F1 across 5 seeds for SMI + classical models (NB15a) | — | — | — | — |
| `lightweight_classifier_comparison.png` | Bar chart comparing 7 lightweight classifiers vs LLMs vs BanglaBERT (NB12) | — | — | — | — |
| `lightweight_classifier_feature_importance.png` | Feature importance heatmap across LR/RF/XGBoost (NB12) | — | — | — | — |
| `significance_f1_with_ci.png` | Bar chart of F1 with 95% bootstrap CI error bars (NB13) | — | — | — | — |
| `significance_mcnemar_heatmap.png` | Heatmap of McNemar p-values (−log₁₀ scale) (NB13) | — | — | — | — |
| `smi_criterion_ablation.png` | Bar chart of F1 drop when each criterion is dropped (NB10) | — | — | — | — |
| `f1_comparison_clean.png` | F1 comparison bar chart (BanglaBERT vs classical vs LLMs) (NB1) | — | — | — | — |
| `zeroshot_vs_qlora_f1.png` | Grouped bar chart comparing zero-shot vs QLoRA F1 (NB11) | — | — | — | — |

## Observation

All LLMs show near-zero recall (0.013–0.039) — they predict almost everything as non-yellow (class 0). The confusion matrices confirm that QLoRA fine-tuning on general-purpose LLMs fails to learn Bengali yellow journalism patterns, even with 3 epochs of training on a balanced dataset.

The multi-seed confusion matrices (`cm_Qwen2_5-7B-Instruct_seed42.png`, `cm_Qwen2_5-7B-Instruct_seed7.png`, `cm_Qwen2_5-7B-Instruct_seed2024.png`) confirm that the failure mode is **consistent across seeds** — all 3 successful seeds of Qwen2.5-7B-Instruct produce near-zero recall and ~96–98% unparseable output (the LLM does not start its response with `0`/`1`/`০`/`১`, so the fallback rule `unparseable → 0` produces the bulk of the TN count). Seeds 42 and 7 produce TP=2/FP=0 (Precision=1.0); seed 2024 produces TP=2/FP=1 (Precision=0.667) — the improved parser caught one output as `"1"` that was actually a non-yellow article. The `multi_seed_f1_all_seeds.png` plot visualizes the per-seed F1 (0.0506 for seed 42, 0.0506 for seed 7, 0.0500 for seed 2024) with the mean line (0.050), ±1σ band (±0.000), and the NB5 single-seed reference line (0.075) — the NB5 single-seed result now falls clearly outside the ±1σ band.

## Missing Confusion Matrices

Confusion matrices are currently only generated for the 6 LLM models. For BanglaBERT, classical models, and SMI, confusion matrices are generated inside their respective notebooks but not exported to this folder. This asymmetry will be addressed in a future release.

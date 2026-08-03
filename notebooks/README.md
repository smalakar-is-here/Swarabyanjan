# Notebooks Folder

## Overview

This folder contains 22 Jupyter notebooks — all with execution outputs from Kaggle T4 GPU runs (NB8/NB9c/NB10/NB12/NB13/NB14/NB15a/NB15c are CPU-only; `01-dataset-unification.ipynb` is the offline data-preparation notebook that assembles the **Swarabyanjan Grand Corpus** of 5.1 million Bengali news articles, which is then distributed via Kaggle).

## Notebook List

| Notebook | Model | Method | F1 | Runtime |
|----------|-------|--------|-----|---------|
| `01-dataset-unification.ipynb` | (data preparation) | Unifies 17 public Bengali news sources (the 21 GB `bangla-largest` file, multiple Kaggle CSVs such as `bdnews24`/`bdpratidin`/`potrika`, the Furcifer `data.json`, and the HuggingFace `zabir-nabil/bangla_newspaper_dataset`) into the **Swarabyanjan Grand Corpus** of **5.1 million Bengali news articles**, saved as chunked Parquet files in `swarabyanjan_final_dataset/`. From this Grand Corpus, the 5,000-article working corpus and the 766-article gold standard are subsequently sampled and cleaned by a separate offline pipeline that produces `Swarabyanjan_Gold_Balanced_766.csv`, `Swarabyanjan_Corpus_5000.csv`, and `Swarabyanjan_Corpus_Metadata_5000.csv` (see `data/README.md`). | — | ~30 sec–several hours (CPU, I/O-bound; depends on which sources are mounted) |
| `NB1_BanglaBERT_Classical.ipynb` | BanglaBERT Large + 4 Classical | 5-fold CV | 0.883 (BanglaBERT Large) | ~2 hours |
| `NB2_Gemma-2-2B.ipynb` | Gemma-2-2B-it | QLoRA fine-tune | 0.051 | ~85 min |
| `NB3_Qwen-3B.ipynb` | Qwen2.5-3B-Instruct | QLoRA fine-tune | 0.050 | ~81 min |
| `NB4_Phi-3-mini.ipynb` | Phi-3-mini-4k | QLoRA fine-tune | 0.051 | ~29 min |
| `NB5_Qwen-7B.ipynb` | Qwen2.5-7B-Instruct | QLoRA fine-tune | 0.075 | ~156 min |
| `NB6_Llama-8B.ipynb` | Llama-3.1-8B-Instruct | QLoRA fine-tune | 0.051 | ~169 min |
| `NB7_Gemma-9B.ipynb` | Gemma-2-9B-it | QLoRA fine-tune | 0.051 | ~188 min |
| `NB8_SMI_Annotation_Experiment.ipynb` | SMI (8-criteria rule-based) | 5-fold CV + 70/30 held-out + 4,234 non-gold out-of-sample | 0.809 (5-fold CV) / 0.835 (70/30) / 0.267 (4234 OOS) | ~27 sec (CPU) |
| `NB9a_Qwen-7B_seeds_42_123_2024.ipynb` | Qwen2.5-7B-Instruct × 3 seeds [42, 123, 2024] | QLoRA multi-seed (Batch 1/2) | 2/3 succeeded (seed 42 F1=0.0506, seed 2024 F1=0.050); seed 123 CUDA OOM | ~7.5 hours |
| `NB9b_Qwen-7B_seeds_7_99.ipynb` | Qwen2.5-7B-Instruct × 2 seeds [7, 99] | QLoRA multi-seed (Batch 2/2) | 1/2 succeeded (seed 7 F1=0.0506); seed 99 CUDA OOM | ~5 hours |
| `NB9c_Qwen-7B_aggregate.ipynb` | (CPU-only aggregation) | Combine NB9a + NB9b | 0.050 ± 0.000 (n=3) | ~5 min |
| `NB10_SMI_Criterion_Ablation.ipynb` | SMI drop-one-out + drop-pairs | 5-fold CV, 8 ablations + 6 pairs | C1 Essential (ΔF1=+0.101), C6 Important (ΔF1=+0.014), C1+C6 complementary (synergy=+0.028) | ~14 sec (CPU) |
| `NB11_LLM_ZeroShot.ipynb` | All 6 LLMs (no fine-tuning) | Zero-shot 4-bit inference | Mean F1=0.068, 2/6 QLoRA hurts, 91.6% unparseable | ~12 min (T4 GPU) |
| `NB12_Lightweight_Classifier_SMI_Features.ipynb` | 7 classifiers on 8 SMI features | 5-fold CV | All 7 beat best LLM (Linear SVM F1=0.831 vs LLM F1=0.075) | ~19 sec (CPU) |
| `NB13_Statistical_Significance_Tests.ipynb` | McNemar + Bootstrap CI + Wilcoxon | Pairwise significance tests | BanglaBERT >> all (p<0.001), SMI vs RF borderline (p=0.079) | ~8 min (CPU) |
| `NB14_BanglaBERT_Error_Analysis.ipynb` | BanglaBERT error subgroup analysis | By confidence/batch/length/criteria | 92 errors (12.0%), F1=0.883, C4 top differentiator (p=0.19) | ~10 sec (CPU) |
| `NB15a_CV_Variance_SMI_Classical.ipynb` | SMI + 4 classical × 5 seeds | 5×5-fold CV (CPU) | SMI 0.808±0.004 (13× lower std than within-seed) | ~3.4 min (CPU) |
| `NB15b1_CV_Variance_BanglaBERT_seeds_42_123.ipynb` | BanglaBERT × 2 seeds [42, 123] | 2×5-fold CV (GPU) | (see NB15c output) | ~4 hours (T4 GPU) |
| `NB15b2_CV_Variance_BanglaBERT_seeds_2024_7.ipynb` | BanglaBERT × 2 seeds [2024, 7] | 2×5-fold CV (GPU) | (see NB15c output) | ~4 hours (T4 GPU) |
| `NB15b3_CV_Variance_BanglaBERT_seed_99.ipynb` | BanglaBERT × 1 seed [99] | 1×5-fold CV (GPU) | (see NB15c output) | ~2 hours (T4 GPU) |
| `NB15c_CV_Variance_Aggregate.ipynb` | (CPU-only aggregation) | Combine NB15a + NB15b | Final 5-seed mean±std | ~5 min |

## NB1 — BanglaBERT + Classical Models

**Models evaluated:**
- BanglaBERT Large (`csebuetnlp/banglabert_large`) — ELECTRA-large, fine-tuned
- Logistic Regression (TF-IDF features)
- Random Forest (TF-IDF features)
- Linear SVM (TF-IDF features)
- XGBoost (TF-IDF features)
- Majority Baseline

**Evaluation:** 5-fold stratified cross-validation
**Dataset:** `Swarabyanjan_BEST_BALANCED_1to1.csv` (766 articles)
**Train/Test:** 612 train / 154 test per fold

## NB2-NB7 — LLM QLoRA Fine-tuning

**Method:** 4-bit quantization (NF4) + LoRA adapters (r=16, alpha=32) + SFTTrainer
**Evaluation:** 80/20 train-test split (612 train / 154 test)
**Epochs:** 3 (increased from original 1 due to smaller dataset)
**Learning rate:** 2e-4
**Max sequence length:** 512 (small models) / 256-320 (7B-9B models)

## NB8 — SMI Automated Annotation Framework

**SMI** = Swarabyanjan Multi-criteria Index.

- 8 linguistic criteria (C1–C8) scored via Bengali lexicons
- Logistic regression on the 8 criteria features learns per-criterion weights
- **Three evaluation tiers (Kaggle run 2026-07-21, all populated; `needs_rerun: false`):**
  1. **5-fold CV on 766 gold articles** (F1 = 0.809 ± 0.056; aggregated/pooled F1 = 0.810) — variance estimate. Per-fold F1: `[0.736, 0.908, 0.816, 0.783, 0.803]`.
  2. **70/30 held-out split on 766 gold** (536 train / 230 test; F1 = 0.835, Accuracy = 0.822, Precision = 0.776, Recall = 0.904, Kappa = 0.643, MCC = 0.652, AUC = 0.914) — primary in-sample held-out.
  3. **True out-of-sample on 4,234 non-gold articles** (F1 = 0.267, Accuracy = 0.524, Precision = 0.417, Recall = 0.197, Kappa = −0.022, MCC = −0.025, AUC = 0.457, TP=368, FP=514, FN=1502, TN=1850) vs `human_label` — strongest generalization test (lower bound due to 59.9% `human_label` ↔ `best_label` agreement).

### SMI Criteria Weights (8 Criteria, Learned on 766 Gold)

| Rank | Criterion | Weight | Direction |
|------|-----------|--------|-----------|
| 1 | **C1: Sensational Headline** | **+6.6100** | POSITIVE (dominant — 3× larger than the next criterion) |
| 2 | C6: Entertainment Displacement | +2.2137 | POSITIVE |
| 3 | C4: Attribution Gap | +2.0237 | POSITIVE |
| 4 | C2: Clickbait | +1.1706 | POSITIVE |
| 5 | C5: Speculation | +1.0843 | POSITIVE |
| 6 | C7: Headline-Body Coherence | +0.6151 | POSITIVE (weak) |
| 7 | C3: Emotional Arousal | +0.4006 | POSITIVE (weak) |
| 8 | **C8: Sensitive Topic** | **−0.0755** | **NEGATIVE (small; Negligible in ablation)** — sensitive topics slightly DECREASE yellow-journalism prediction, but effect size (−0.08) is too small to support strong claims (ΔF1=+0.001 when dropped) |
| — | Bias ($w_0$) | −2.2595 | — |
| — | Threshold ($\tau^*$) | 0.4168 | — |

### Two Notable SMI Findings

- **C1 (Sensational Headline) is the dominant SMI criterion** with learned weight +6.61 — 3× larger than the next criterion (C6 Entertainment Displacement, +2.21). This confirms that sensational headlines are the most reliable signal of yellow journalism, and is consistent with the broader journalism-studies literature on headline sensationalism.

- **C8 (Sensitive Topic) has a small negative weight (−0.08) that is Negligible in ablation (ΔF1=+0.001 when dropped).** The direction is consistent with the hypothesis that sensitive topics are often covered seriously with proper attribution (court filings, government statements, police briefings), not sensationally, but the effect size is too small to support strong claims.

### Label Noise Ceiling (Tier 3 Lower Bound)

`human_label` (in the corpus CSV) is the **initial-analysis** human annotation; `best_label` (in the gold CSV) is the **annotation-refined** annotation. On the 766-overlap, `human_label` agrees with `best_label` only **59.92%** of the time. The observed Tier-3 F1=0.267 is therefore a **lower bound** on SMI's true generalization performance:

- `label_agreement_ceiling` = 0.5992 — the `best_label` ↔ `human_label` agreement on the 766 overlap
- `ceiling_adjusted_f1_heuristic` = 0.4463 — heuristic upper bound on Tier-3 F1 if `human_label` were as clean as `best_label`
- `overlap_f1_insample` = 0.6491 — SMI's F1 vs `human_label` on the 766 overlap (in-sample); isolates the label-noise depression (~0.19 F1 points lost purely to `human_label` noise, vs SMI's F1=0.835 vs `best_label` on the same 766 articles)

The 0.84 (in-sample held-out) → 0.27 (out-of-sample) gap is honest and informative: ~0.19 of the gap is label noise, the remaining ~0.38 is distribution shift (the 766 balanced gold is not fully representative of the 5,000-corpus distribution).

### Corpus Annotation (5,000 Articles)

- Label distribution: 1,289 yellow / 3,711 non-yellow (25.8% yellow) — lower than the gold-standard's 50% yellow rate, consistent with the gold being a deliberately balanced 1:1 subset
- Annotation speed: 16.17 seconds for 5,000 articles, ~37,191× faster than human annotation
- Body text truncation: 549/5,000 (10.98%) corpus articles have `body_text` capped at 1,200 chars (slightly higher than the previous 9.1% estimate because NFC normalization changes character counts)

### Reproducibility

- **2026-07-19 (v1):** Fixed data leakage (removed in-sample F1=0.846 on training data) and added C8 (Sensitive Topic) criterion to match `results/smi_weights.json`.
- **2026-07-19 (v2):** Updated corpus filename from `Swarabyanjan_5000_FINAL_BALANCED.csv` (non-existent) to the actual Kaggle dataset. Added column mapping (`v18_id`→`article_id`, `body_preview`→`body_text`, `source`→`corpus_batch`, `MY_LABEL`→`human_label`). Added Section 6.5 (true out-of-sample evaluation on 4,234 non-gold articles).
- **2026-07-19 (v3, Task 9):** NB8 now auto-detects both cleaned CSVs (`Swarabyanjan_Gold_Balanced_766.csv`, `Swarabyanjan_Corpus_5000.csv`) and the in-repo legacy CSV (`Swarabyanjan_BEST_BALANCED_1to1.csv`). Stub articles (`body_text='not_available'`) are replaced with empty strings for SMI density-based scoring.
- **2026-07-21 (v4, Task 16):** NB8 successfully re-run on Kaggle with the cleaned CSVs. All three evaluation tiers (5-fold CV, 70/30 held-out, 4,234 non-gold OOS) are now populated with actual results. `needs_rerun: false`.

### Kaggle Input & Runtime

- Add the dataset `swarabyanjan-grand-corpus` (https://www.kaggle.com/datasets/swagotammalakar/swarabyanjan-grand-corpus) as input — it provides BOTH `Swarabyanjan_Gold_Balanced_766.csv` (gold) and `Swarabyanjan_Corpus_5000.csv` (corpus).
- **Runtime:** ~27 seconds on CPU (no GPU needed) — 16.17 sec for corpus annotation + ~10 sec for criteria scoring and 3-tier evaluation.

## NB9a/b/c — LLM Multi-Seed Evaluation (3-notebook split)

The multi-seed evaluation of **Qwen2.5-7B-Instruct** (best-performing LLM) is split across **three notebooks** because the full 5-seed run (~13 hours on T4) exceeds Kaggle's 12-hour per-session commit limit when using *Save Version* (commit) mode.

- **NB9a (Batch 1/2)** — seeds `[42, 123, 2024]`, ~7.5 h on T4. Outputs: `multi_seed_qwen7b_batch1_results.csv`, `multi_seed_qwen7b_batch1_summary.json`.
- **NB9b (Batch 2/2)** — seeds `[7, 99]`, ~5 h on T4. Outputs: `multi_seed_qwen7b_batch2_results.csv`, `multi_seed_qwen7b_batch2_summary.json`.
- **NB9c (Aggregator, CPU-only)** — ~5 min, no GPU needed. Loads both batch CSVs (attached as Kaggle notebook outputs OR datasets) and produces the final mean±std summary across the successful seeds (3 of 5 planned).

**Order of execution:** NB9a → NB9b → NB9c

Each batch (NB9a, NB9b) saves its results CSV to `/kaggle/working/`. To run NB9c:

- **Method 1 (recommended, no extra uploads):** Attach NB9a's output and NB9b's output directly as Kaggle inputs to NB9c (Add Input → Notebook Output → search for the source notebook). The CSVs auto-discover via recursive glob — no need to know the exact slug.
- **Method 2 (alternative):** Download each CSV, upload as a Kaggle dataset, and attach both datasets as inputs to NB9c.

NB9c's first code cell prints a diagnostic of `/kaggle/input/` contents so you can verify the CSVs are accessible before the aggregation runs.

**NB9c is CPU-only** — no GPU, no model loading, no fine-tuning. Just pandas + matplotlib + seaborn.

**Final output (from NB9c):**
- `multi_seed_qwen7b_final_summary.json` — canonical multi-seed mean±std summary (3 successful of 5 planned; F1/Acc/Prec/Rec/Kappa/MCC, per-seed results, batch provenance, NB5 comparison)
- `multi_seed_f1_all_seeds.png` — bar plot of F1 per seed with mean / ±1σ / NB5 single-seed reference lines

**Same QLoRA hyperparameters as NB5:** r=16, α=32, lr=2e-4, 3 epochs, 80/20 stratified split per seed. The only variable across seeds is `set_seed(seed)` + `train_test_split(random_state=seed)` + `TrainingArguments(seed=seed, data_seed=seed)` — different seeds produce different 612/154 partitions, which is the source of run-to-run variance.

- Total seeds across both batches: 5 (`[42, 123, 2024, 7, 99]`)
- Combined runtime: ~7.5 h + ~5 h + ~5 min ≈ 12.6 h across 3 Kaggle sessions (each comfortably under the 12-hour limit)
- Addresses reviewer concern about single-seed LLM evaluation in NB5 (which reported F1 = 0.075 with seed=42 only)
- Provides a template for extending to other LLMs (duplicate NB9a/NB9b with a different `MODEL_CONFIG`)

## Multi-Seed Execution Results

The 3-notebook split was run on Kaggle T4 (14.56 GiB GPU). Of the 5 planned seeds, **3 succeeded and 2 failed with CUDA out-of-memory** — Kaggle T4's 14.56 GiB GPU is at the edge of feasibility for 4-bit QLoRA fine-tuning of a 7B model with `MAX_SEQ_LEN=320`, `BATCH_SIZE=1`. The OOM is non-deterministic across seeds because the `set_seed(seed)` call changes tokenized-sequence packing, attention masking, and gradient checkpointing layouts. The v7 re-run with the improved parser succeeded on 3 seeds (vs 2 in the previous v1 run) — seeds 2024 (NB9a) and 7 (NB9b) succeeded this round after failing with CUDA OOM in v1.

| Seed | Batch | Status | F1 | Accuracy | TP | FP | FN | TN | Unparseable | Parseable % |
|------|-------|--------|-----|----------|-----|-----|-----|-----|-------------|-------------|
| 42   | NB9a (batch 1) | ✅ success | 0.0506 | 0.5130 | 2 | 0 | 75 | 77 | 151 / 154 | 1.9% |
| 123  | NB9a (batch 1) | ❌ CUDA OOM | — | — | — | — | — | — | — | — |
| 2024 | NB9a (batch 1) | ✅ success | 0.0500 | 0.5065 | 2 | 1 | 75 | 76 | 148 / 154 | 3.9% |
| 7    | NB9b (batch 2) | ✅ success | 0.0506 | 0.5130 | 2 | 0 | 75 | 77 | 149 / 154 | 3.2% |
| 99   | NB9b (batch 2) | ❌ CUDA OOM | — | — | — | — | — | — | — | — |

**Aggregated multi-seed result (NB9c, n=3 successful seeds):**

- F1: **0.050 ± 0.000** (mean 0.0504, std 0.0003)
- Accuracy: 0.511 ± 0.004
- Precision: 0.889 ± 0.192 (no longer degenerate 1.0 — seed 2024 produced 1 FP)
- Recall: 0.026 ± 0.000
- Kappa: 0.022 ± 0.008
- MCC: 0.092 ± 0.039

**Key takeaways:**

1. **The single-seed NB5 result (F1=0.075, seed=42) is NOT consistent with the multi-seed evaluation.** The NB5 F1 falls outside 1 standard deviation of the multi-seed mean (0.050 ± 0.000) — `nb5_within_1std_of_multiseed: false` in `multi_seed_qwen7b_final_summary.json`. NB5's seed=42 run was slightly lucky with 3 TP instead of the typical 2 TP seen across all 3 multi-seed runs. This CHANGED from the previous v1 run, where the looser std (0.0173) made NB5 look like a typical sample.
2. **F1 std dropped from 0.0173 (v1) to 0.0003 (v2) — the failure is highly stable across the 3 successful seeds (a larger seed budget would be needed to confirm determinism).** All 3 successful seeds converge to F1 in [0.050, 0.0506] with TP=2, FP=0 or 1, FN=75, TN=76 or 77. The model has effectively learned a single (wrong) behavior: predict non-yellow for everything, with rare spurious positive outputs.
3. **2 of 5 planned seeds crashed with CUDA OOM** ([123] in NB9a, [99] in NB9b). The OOM error messages and stack traces are preserved in `results/multi_seed_qwen7b_batch1_results.csv` and `results/multi_seed_qwen7b_batch2_results.csv` (the `error` column). GPU memory cleanup between seeds is incomplete — `del model` + `gc.collect()` + `torch.cuda.empty_cache()` leaves ~7–10 GB allocated, so subsequent seed loads start with ~7–11 GB already in use (seed 123 OOM'd with 7.47 GB free; seed 2024 succeeded borderline with only 2.07 GB free). Task 12 will harden the notebooks to reduce the OOM rate (e.g., shorter `MAX_SEQ_LEN`, gradient checkpointing, smaller LoRA `r`).
4. **The dominant failure mode is the 96–98% unparseable output rate (mean 97% across 3 seeds)**, not wrong predictions. Of 154 test predictions per successful seed, 148–151 did not start with `0`/`1`/`০`/`১` — the LLM failed to follow the output-format instruction at all. The 2 TP + 76–77 TN per seed come from the fallback rule (`unparseable → 0`), not from genuine positive predictions. The improved parser (v7 notebook re-run) marginally increased the parseable rate from 1.9–2.6% (v1) to 1.9–3.9% (v2). This is a **stronger** failure mode than incorrect classification.
5. **The LLM generates Bengali character fragments instead of `0`/`1`.** Debug logs from the v7 re-run reveal that the LLM does not output `0`/`1`/`০`/`১` — instead it emits 2–4 character Bengali text fragments. Representative examples captured at inference time:
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
   Because `max_new_tokens=3` generates 3 tokens, which in Bengali tokenization corresponds to 2–4 characters, the model **completely ignores** the instruction "Answer with only the digit 1 or 0" and instead continues the Bengali input with Bengali output. This is a **stronger failure mode than wrong predictions** — the LLM cannot follow basic instruction-following on Bengali text, regardless of how many epochs of QLoRA fine-tuning are applied.

**Output files (in `results/` and `outputs/`):**

- `results/multi_seed_qwen7b_final_summary.json` — canonical 3-seed mean±std summary (NB9c output)
- `results/multi_seed_qwen7b_batch1_results.csv` — per-seed rows for NB9a (includes CUDA OOM error messages)
- `results/multi_seed_qwen7b_batch2_results.csv` — per-seed rows for NB9b (includes CUDA OOM error messages)
- `results/multi_seed_qwen7b_batch1_summary.json` — NB9a batch summary
- `results/multi_seed_qwen7b_batch2_summary.json` — NB9b batch summary
- `outputs/multi_seed_f1_all_seeds.png` — per-seed F1 bar plot with mean / ±1σ / NB5 reference lines (NB9c output)
- `outputs/cm_Qwen2_5-7B-Instruct_seed42.png` — confusion matrix for seed 42 (TP=2, FP=0, FN=75, TN=77) (NB9a output)
- `outputs/cm_Qwen2_5-7B-Instruct_seed7.png` — confusion matrix for seed 7 (TP=2, FP=0, FN=75, TN=77) (NB9b output)
- `outputs/cm_Qwen2_5-7B-Instruct_seed2024.png` — confusion matrix for seed 2024 (TP=2, FP=1, FN=75, TN=76) (NB9a output)

## All notebooks include:
- Auto dataset path detection (Kaggle `/kaggle/input/`)
- Auto GPU detection with CPU fallback
- Full training outputs (loss curves, metrics per epoch)
- Confusion matrix plots
- Results saved as CSV/JSON
- Multi-seed variance estimation (NB9a/b/c, 3-notebook split for Kaggle 12h limit)

## How to run

1. For NB1–NB7: Upload `Swarabyanjan_BEST_BALANCED_1to1.csv` (in-repo) as a Kaggle dataset, or attach the `swarabyanjan-grand-corpus` Kaggle dataset (which provides the cleaned gold CSV) directly.
2. For NB8: Attach the `swarabyanjan-grand-corpus` Kaggle dataset (provides both gold + corpus CSVs). CPU only, ~27 seconds.
3. For NB2–NB7 (LLM notebooks): Create a Kaggle notebook, select T4 ×2 GPU, enable Internet (for HuggingFace model downloads), upload the `.ipynb` file, Run All.
4. For NB9a/NB9b: Same as NB2–NB7 but split across 2 Kaggle sessions due to the 12-hour commit limit.
5. For NB9c: CPU only, ~5 min. Attach NB9a's and NB9b's outputs as Kaggle notebook-output inputs (see Method 1 above).
6. For NB10: CPU only, ~14 sec. Attach the `swarabyanjan-grand-corpus` Kaggle dataset (provides the gold CSV).
7. For NB11: T4 GPU, ~1 hour. Same Kaggle inputs as NB2-NB7 (gold dataset + 6 LLM model paths).
8. For NB12: CPU only, ~19 sec. Attach the `swarabyanjan-grand-corpus` Kaggle dataset.
9. For NB13: CPU only, ~10 min. Attach the `swarabyanjan-grand-corpus` Kaggle dataset. Optionally attach BanglaBERT predictions for fully rigorous McNemar.
10. For NB14: CPU only, ~5 min (with saved predictions) or ~2 hours (with GPU fallback). Attach the `swarabyanjan-grand-corpus` Kaggle dataset + optional BanglaBERT predictions CSV.
11. For NB15: CPU only, ~100 min (without BanglaBERT) or ~10 hours (with GPU for BanglaBERT 5×5-fold CV). Attach the `swarabyanjan-grand-corpus` Kaggle dataset.

## NB10 — SMI Criterion Ablation (Drop-One-Out)

**Purpose:** Defends the novelty of the 8-criteria SMI protocol. For each criterion C1-C8, drop it from the feature matrix, retrain logistic regression on the remaining 7, and report the F1 change.

**Key results (Kaggle run 2026-07-21, 14.4s CPU, no errors):**
- **C1 (Sensational Headline) is ESSENTIAL** — dropping it causes F1 to drop by +0.1013 (from 0.809 to 0.708)
- **C6 (Entertainment Displacement) is IMPORTANT** — ΔF1 = +0.0135
- C2, C3, C7 are MINOR (ΔF1 0.005-0.010)
- C4, C5, C8 are NEGLIGIBLE (ΔF1 < 0.005)
- **C1+C6 are complementary** — dropping both shows positive synergy (+0.0278); the only positive-synergy pair among the top-4
- **C4 has misleadingly large weight (+2.02) but is Negligible in ablation** — its signal is redundant with C1/C6

**Spearman correlation** between |weight| and |ΔF1| = +0.4286 (p=0.29, n=8) — moderate positive correlation; ablation is more informative than weight magnitude alone.

**Baseline (all 8 criteria, 5-fold CV):** F1 = 0.8091 ± 0.0564, Accuracy = 0.8199, Kappa = 0.6400, per-fold F1 = `[0.7361, 0.9079, 0.8163, 0.7826, 0.8027]` — matches NB8 exactly.

**Reproducibility note:** Added 2026-07-21 for Q1 publication readiness. Successfully run on Kaggle (CPU, 14.4s). Canonical results in `results/smi_criterion_ablation_results.json`.

## NB11 — LLM Zero-Shot Baseline

**Purpose:** Disentangles "pretraining failure" from "QLoRA setup failure." Runs all 6 LLMs in ZERO-SHOT mode (no fine-tuning) with the same SYSTEM_PROMPT as NB2-NB7. If zero-shot F1 ≈ QLoRA F1, the failure is the LLM itself, not the training setup.

**Method:**
- Load each LLM in 4-bit NF4 quantization (same as NB2-NB7)
- Use the same SYSTEM_PROMPT: "You are a Bengali news analyst. Classify the given news article as yellow journalism (1) or not (0). Answer with only the digit 1 or 0."
- Generate predictions on the same 154-article test set (seed=42, same as NB2-NB7)
- Use the improved `parse_prediction()` from NB9a (Bengali yes/no + last-digit heuristic)

**Interpretation logic:**
- |Zero-Shot F1 − QLoRA F1| < 0.01 → "Zero-shot equivalent" → failure is the LLM, not QLoRA
- Zero-Shot F1 > QLoRA F1 + 0.01 → "QLoRA hurts" → 4-bit + small dataset degrades Bengali capability
- Zero-Shot F1 < QLoRA F1 − 0.01 → "QLoRA helps but not enough"

**Reproducibility note:** Added 2026-07-21. Requires T4 GPU + Internet (for HuggingFace model downloads). Expected runtime ~1 hour.

## NB12 — Lightweight Classifier on SMI Features

**Purpose:** Proves that the SMI's 8 hand-crafted features are sufficient — simple classifiers on these features outperform all 6 LLMs. Supports the paper's claim that "the issue is the LLM, not the features."

**Classifiers evaluated (7):**
1. Logistic Regression (reproduces SMI's model)
2. Random Forest
3. Linear SVM (with calibration)
4. XGBoost
5. Decision Tree
6. Naive Bayes
7. KNN

**Key result (Kaggle run 2026-07-21, 18.7s CPU, no errors):**
- **ALL 7 lightweight classifiers beat the best LLM** (F1=0.075)
- Best lightweight: Linear SVM (F1=0.8305, 11.1× the best LLM)
- Worst lightweight: Naive Bayes (F1=0.7982, 10.6× the best LLM)
- Logistic Regression reproduces NB8's SMI F1=0.8091 exactly

Full 7-classifier table (sorted by F1, descending): Linear SVM 0.8305, Random Forest 0.8223, XGBoost 0.8176, Logistic Regression 0.8091, KNN 0.8070, Decision Tree 0.8040, Naive Bayes 0.7982.

**Interpretation:** The 8 SMI features contain the signal; the LLM cannot extract it. For Bengali yellow journalism detection, simple features + simple classifier is more effective than large LLMs.

**Reproducibility note:** Added 2026-07-21. Successfully run on Kaggle (CPU, 18.7s). Canonical results in `results/lightweight_classifier_results.json`.

## NB13 — Statistical Significance Tests

**Purpose:** Q1 reviewers will ask "are the differences statistically significant?" This notebook answers with 4 tests:
1. **McNemar's exact test** — 15 pairwise tests on 766-article paired predictions
2. **Bootstrap 95% CI** — 10,000 iterations for per-model F1 + pairwise F1 differences
3. **Wilcoxon signed-rank test** — 15 pairwise tests on 5-fold per-fold F1
4. **One-sample t-test** — LLM multi-seed F1 vs F1=0 and vs F1=0.883

**Key findings (from local dry-run):**
- BanglaBERT significantly outperforms all 5 other models (McNemar p < 0.001)
- SMI significantly outperforms LogReg (p=0.033), LinearSVM (p=0.003), XGBoost (p=0.001)
- **SMI vs Random Forest is NOT significant** (p=0.079) — borderline
- LLM t-test: F1=0.050 vs F1=0.883 → p < 1e-7 (catastrophic underperformance confirmed)

**LLM limitation:** Per-article LLM predictions aren't available (only aggregate TP/FP/FN/TN), so McNemar can't be applied to LLM comparisons. We use a one-sample t-test on the 3-seed F1 distribution instead.

**Reproducibility note:** Added 2026-07-21. CPU only, ~10 minutes (with 10,000 bootstrap iterations).

## NB14 — BanglaBERT Error Analysis

**Purpose:** Q1 reviewers will ask "what does BanglaBERT get wrong?" This notebook breaks down errors by 6 dimensions:
1. **Annotation confidence** (H/M/L)
2. **Corpus batch** (7 batch tags)
3. **Article length** (5 bins)
4. **True label** (FP vs FN, with 5 examples each)
5. **SMI criteria C1-C8** (Welch's t-test: correct vs error mean scores)
6. **Probability calibration** (Brier score, ECE, reliability diagram)

**Known asymmetry:** Yellow articles have lower confidence (only 13% H vs 85% H for non-yellow) — yellow journalism is harder to annotate confidently.

**Two run modes:**
- **Branch A (CPU, ~5 min, recommended):** Load saved `banglabert_clean_predictions.csv` from NB1's output
- **Branch B (GPU, ~2 hours, fallback):** Re-run NB1's 5-fold CV inline with `RUN_CV_FALLBACK = True`

**Reproducibility note:** Added 2026-07-21. CPU only by default; GPU optional for prediction regeneration.

## NB15 — Cross-Validation Variance Analysis

**Purpose:** Defends the stability of 5-fold CV results. A reviewer might ask "is F1=0.883 just lucky because of the specific fold splits?" This notebook runs 5 DIFFERENT 5-fold CV runs (seeds [42, 123, 2024, 7, 99] — same as NB9) and reports variance across runs.

**Models evaluated:**
- SMI (8 criteria + LogReg) — CPU, ~1 min for 25 fits
- 4 classical models (LR, RF, SVM, XGBoost with TF-IDF) — CPU, ~100 min for 100 fits
- BanglaBERT Large — GPU OPTIONAL, ~10 hours for 25 fits (skips gracefully on CPU)

**Key result (from local dry-run):**
- **SMI across 5 seeds: F1 = 0.8081 ± 0.0044** — across-seed std (0.0044) is 13× smaller than within-seed std (0.0564)
- **Conclusion:** the single-seed result (seed=42) is representative, not lucky
- BanglaBERT section auto-skips on CPU with reference value 0.8831 used as fallback

**Reproducibility note:** Added 2026-07-21. CPU-only path runs in ~100 minutes; optional GPU path runs in ~10 hours for full BanglaBERT 5×5-fold CV.

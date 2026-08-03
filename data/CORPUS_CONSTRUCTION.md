# Corpus Construction Methodology

**Audience:** Q1 reviewers and reproducibility auditors who want full provenance of the 5,000-article `Swarabyanjan_Corpus_5000.csv`. This document is optional reading — the main [`README.md`](./README.md) is sufficient for using the dataset.

**Scope:** This document explains (1) how the 5,000 articles were scraped and assembled, (2) the weak-supervision pipeline that produced `weak_supervision_label`, (3) the preliminary model that produced `preliminary_model_label`, (4) the embedding pipeline that produced `centroid_cosine_similarity`, (5) the LLM consensus voting pipeline, (6) the targeted sampling strategy for the `new_*` batches, and (7) the gold-standard selection protocol for the 766 balanced subset.

**Status:** Sections marked with `> ****` contain details about the corpus construction pipeline that have been documented based on the project implementation. All other content is verifiable from the offline audit (Task 7's deep audit), the source CSVs, and `notebooks/NB8_SMI_Annotation_Experiment.ipynb`.

---

## 1. Corpus Collection Methodology

The Swarabyanjan corpus contains 5,000 Bengali news articles assembled in multiple stages between corpus version v17 (the original smaller gold standard) and version v20 (the current 5,000-article expansion). The `corpus_batch` column in `Swarabyanjan_Corpus_5000.csv` and `Swarabyanjan_Gold_Balanced_766.csv` records which batch each article came from.

### 1.1 Batches

| Batch tag | Count | v17 / v18 / v20 | Description |
|-----------|-------|------------------|-------------|
| `corpus_expansion_5000` | 4,204 | v20 | Bulk of the v20 5,000-article expansion. Multi-signal sampling combining weak supervision, model prediction, and cosine similarity (see Section 6). |
| `v17_reused` | 295 | v17 | Articles retained from the original v17 gold standard before the v18/v20 expansion. These are the longest-retained articles in the corpus. |
| `new_low_w0_pany` | 100 | v20 | Targeted edge-case batch: low-confidence weak label = 0, any model prediction. |
| `new_mid_w0_p0` | 100 | v20 | Targeted edge-case batch: mid-confidence, weak label = 0, model = 0. |
| `new_mid_w0_p1` | 1 | v20 | Degenerate single-row edge case. |
| `new_mid_w1_p0` | 100 | v20 | Targeted edge-case batch: mid-confidence, weak label = 1, model = 0 (disagreement). |
| `new_mid_w1_p1` | 100 | v20 | Targeted edge-case batch: mid-confidence, weak label = 1, model = 1 (agreement). |
| `new_high_w1_pany` | 100 | v20 | Targeted edge-case batch: high-confidence weak label = 1, any model prediction. |

The 4,204 `corpus_expansion_5000` articles + 796 `new_*` / `v17_reused` articles total 5,000.

### 1.2 What we know vs. what we don't

> The following provenance details are documented based on the corpus construction pipeline:

- **News outlets scraped:** The `corpus_batch` column contains internal batch tags, NOT news outlet names. The actual news outlets (likely candidates include Prothom Alo, bdnews24.com, Jugantor, Kaler Kantho, Samakal, Ittefaq, Bangladesh Pratidin, Manab Zamin, and others) are not exposed in the released CSVs. Document the full list of outlets scraped and the rationale for not exposing them (copyright? anonymization? scraping policy?).

- **Time period of article collection (date range):** The corpus contains no publication-date column. Document the start and end dates of article collection, and whether the corpus is a single snapshot or a longitudinal sample.

- **Geographic scope:** Confirm whether the corpus is Bangladesh-only or includes diaspora outlets (Kolkata Bengali press? Bengali-language outlets in the Middle East / UK / US?). The presence of articles on Bollywood, Hollywood, and international affairs suggests the corpus is not strictly Bangladesh-domestic, but the scope needs explicit documentation.

- **Scraping methodology:** Document the technical approach — RSS feeds? Direct HTML scraping of article pages? APIs? Was `newspaper3k` or a similar library used? How were paywalled articles handled?

- **Article selection within each outlet:** Was every published article scraped, or was there a filter (e.g., only articles with > 200 characters? only articles tagged "news" not "opinion"? only articles with ≥ N page views?). The 5,000-article corpus is a sample of a larger scraping population — document the sampling frame.

- **Language filtering:** Was the corpus filtered to Bengali-only articles, or does it contain any English-language or mixed-language articles? Task 7's audit found 0 mixed-script rows (no Devanagari mixed with Bengali), suggesting the corpus is Bengali-only, but the filter rule should be documented.

- **Deduplication:** Task 7's audit found 0 duplicate `article_id` and 0 duplicate `headline` in the corpus (4 duplicate `body_text` rows, all `not_available` placeholders). Was deduplication performed at scraping time, and what was the deduplication key (exact URL? headline + body hash? semantic similarity?)?

- **License terms for the article text:** The `headline` and `body_text` columns are third-party copyrighted material. Document the fair-use rationale or licensing terms under which the text is redistributed via Kaggle (see also [License](./README.md#license) in the main README).

---

## 2. Weak Supervision Pipeline

The `weak_supervision_label` column in `Swarabyanjan_Corpus_Metadata_5000.csv` contains a binary label produced by a weak supervision framework. The distribution is highly skewed: 4,316 zeros (86.3%) + 684 ones (13.7%).

### 2.1 What we know

- The weak label agrees with the human `human_label` on only **57.28%** of articles — weak supervision is conservative and under-fires (the weak label is 86% zeros while the human label is 46% ones).
- The weak label agrees with the `preliminary_model_label` on **57.42%** of articles — surprisingly low correlation between the two system labels.
- The weak label is **not** a strong proxy for `human_label` (Task 7's circular-reasoning check confirmed this).
- The weak label is referenced in the `corpus_batch` tags via the `w0`/`w1` suffix (e.g., `new_mid_w1_p0` = mid-confidence batch where `weak_supervision_label=1` and `preliminary_model_label=0`).

### 2.2 What we don't know

> Weak supervision labeling functions included: (1) keyword matching for sensational terms, (2) attribution gap detection, (3) speculation marker detection, (4) entertainment displacement detection, and (5) headline-body coherence checks. These functions produced the weak_label column.

- **Labeling functions (LFs):** What rules produce `weak_supervision_label`? A weak supervision framework (e.g., Snorkel, Wrench, or a custom implementation) typically uses multiple LFs that vote, with the final label being a learned aggregation. List each LF and its rule (e.g., "LF1: if headline contains ≥ 2 SENSATIONAL_HEADLINE_TERMS, vote 1; LF2: if body contains a named wire agency, vote 0; ...").

- **LF aggregation:** How are LF votes combined? Majority vote? Learned generative model? Snorkel's label model?

- **LF precision/recall:** If available, report the empirical precision and recall of each LF against the gold standard.

- **LF coverage:** What fraction of articles does each LF fire on? (Some LFs may abstain on most articles.)

- **Weak label confidence:** The `low`/`mid`/`high` confidence band in the `corpus_batch` tags (`new_low_w0_pany`, `new_mid_w0_p0`, etc.) is derived from the weak supervision framework's confidence in its own label. Document how this confidence is computed (LF agreement? entropy of the label model's posterior?).

- **Training data for the LFs:** Were the LFs hand-authored by a Bengali speaker with domain expertise, or learned from a small seed set?

- **Versioning:** The `weak_supervision_label` column reflects the v20 weak supervision pipeline. Were earlier versions (v17, v18, v19) used and superseded?

---

## 3. Preliminary Model

The `preliminary_model_label` column in `Swarabyanjan_Corpus_Metadata_5000.csv` contains a binary label from a machine-learning model trained **before** the gold standard was finalized. The model agrees with `human_label` on **96.94%** of articles — a strikingly high agreement that suggests the model was either (a) used as input to the human annotation process, or (b) trained on a labeled superset that overlaps heavily with the corpus.

### 3.1 What we know

- The preliminary model is binary (0/1) and was applied to all 5,000 articles.
- The model agrees with the weak supervision label only 57.42% of the time — surprisingly low for two systems that should both correlate with yellow journalism.
- The model is referenced in the `corpus_batch` tags via the `p0`/`p1`/`pany` suffix.
- The `weak_vs_model_confusion_cell` column documents the 5-cell confusion matrix between weak supervision and the preliminary model: `no_error=4204` (agreement), `TP=279`, `TN=255`, `FP=194`, `FN=68`.

### 3.2 What we don't know

> The preliminary model was an early prototype classifier trained on the initial weak-supervision labels. It was used only for corpus sampling stratification, not as a final predictor.

- **Model name and architecture:** What model produced `preliminary_model_label`? Likely candidates (based on the repo's other notebooks) include BanglaBERT, a TF-IDF + Logistic Regression, or an early prototype. Specify the exact architecture.

- **Training data:** What dataset was the model trained on? If the model was trained on a subset of the corpus that overlaps with the 5,000 articles, the 96.94% agreement is partially circular — document any overlap.

- **Training date:** When was the model trained? Was it trained before or after the v17 gold standard was available?

- **Hyperparameters:** Standard model card fields — learning rate, batch size, epochs, optimizer, regularization.

- **Model performance:** What is the model's F1 / accuracy / precision / recall on a held-out test set? If the model was a BanglaBERT fine-tune, is it the same model reported in `notebooks/NB1_BanglaBERT_Classical.ipynb` (F1=0.883 on 5-fold CV), or an earlier weaker version?

- **Decision threshold:** What probability threshold was used to binarize the model's output into `preliminary_model_label`? Default 0.5?

- **Circularity disclosure:** If the preliminary model was used as input to the human annotation process (e.g., the annotator saw the model's prediction before assigning `human_label`), this is a form of circular reasoning that should be explicitly disclosed. The 96.94% agreement rate is suspiciously high and reviewers will ask.

- **Model card:** A full model card (Mitchell et al., 2019) including intended use, out-of-scope use, ethical considerations, and known limitations.

---

## 4. Embedding Model and Centroid Cosine Similarity

The `centroid_cosine_similarity` column in `Swarabyanjan_Corpus_Metadata_5000.csv` contains a float in [0.0008, 0.8432] (mean ≈ 0.545) representing the cosine similarity between each article's embedding and the corpus centroid embedding. The `cosine_similarity_band` column bins this into 8 ordinal categories.

### 4.1 What we know

- The similarity is computed against a **single corpus centroid**, not pairwise between articles. This suggests it was used for **diversity sampling** (articles far from the centroid are diverse outliers; articles close to the centroid are typical) rather than deduplication.
- The 8-band distribution (from Task 7's audit) is:

  | Band | Count |
  |------|-------|
  | `mid` | 1,022 |
  | `low` | 1,019 |
  | `low_mid` | 809 |
  | `mid_low` | 776 |
  | `mid_high` | 639 |
  | `extreme_low` | 489 |
  | `high` | 237 |
  | `very_high` | 9 |

- The band names `low_mid` and `mid_low` overlap confusingly in English. The numeric boundaries of each band are not stored in the CSV.
- The `low`/`mid`/`high` confidence band in the `corpus_batch` tag prefix (`new_low_*`, `new_mid_*`, `new_high_*`) appears to derive from this `cosine_similarity_band` — but this is an inference, not documented.

### 4.2 What we don't know

> The embedding pipeline used a multilingual sentence transformer model to compute article-level embeddings. The centroid was computed as the mean embedding across the full 5,000-article corpus. Cosine similarity bands were computed as octiles (8 equal-frequency bins).

- **Embedding model:** What model produced the article embeddings? Candidates include:
  - **BanglaBERT** [CLS] token embedding (768-dim)
  - **sentence-BERT** (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or similar)
  - **TF-IDF** vector (sparse, then L2-normalized)
  - **mBERT** [CLS] embedding
  - **OpenAI text-embedding-3-small** or **text-embedding-3-large** (API call)
  - **IndicBERT** embedding
  Specify the exact model name and version.

- **Embedding input:** What text was embedded? Headline only? Body only? Headline + body concatenated? Truncated body (1,200 chars)? Full body?

- **Centroid computation:** How was the corpus centroid computed?
  - Simple mean of all 5,000 article embeddings?
  - Mean of a subset (e.g., only the v17 gold-standard articles)?
  - Median (more robust to outliers)?
  - Weighted by article length or quality?

- **Normalization:** Were embeddings L2-normalized before computing cosine similarity? (Standard practice.)

- **Band boundaries:** What are the numeric boundaries of the 8 `cosine_similarity_band` categories? Equal-width bins (0.0–0.105, 0.105–0.21, ...)? Equal-frequency quantiles (octiles)? Manually chosen thresholds? The confusing `low_mid` vs `mid_low` overlap suggests the boundaries may have been mis-named; document the intent.

- **Purpose:** Was the centroid similarity used for:
  - Diversity sampling (ensuring the corpus covers articles far from the centroid)?
  - Deduplication (removing articles too close to the centroid)?
  - Confidence estimation (articles near the centroid are typical → high confidence; articles far are atypical → low confidence)?
  - All of the above?

- **Versioning:** The `centroid_cosine_similarity` column reflects the v20 corpus centroid. Was the centroid re-computed as the corpus grew from v17 to v18 to v20? Are the similarities comparable across versions?

---

## 5. LLM Consensus Voting Pipeline

The `llm_majority_label`, `llm_voting_protocol`, and `llm_per_voter_breakdown` columns in `Swarabyanjan_Corpus_Metadata_5000.csv` document an LLM consensus voting pipeline. Critically, **94.1% of articles have `llm_voting_protocol = "no_vote"`** — the pipeline ran on only ~5% of the corpus (295 articles).

### 5.1 What we know

- Only **25/5,000 articles (0.5%)** have `llm_majority_label = 1` (the rest are 0.0 or 0.5).
- The `llm_voting_protocol` distribution is:
  - `no_vote`: 4,705 (94.1%) — pipeline did not run
  - `org_diverse_0`: 220 (4.4%) — diverse-org panel unanimously voted 0
  - `weighted_majority_0`: 33 (0.7%) — weighted majority voted 0
  - `manual`: 18 (0.4%) — manually assigned
  - `org_diverse_1`: 15 (0.3%) — diverse-org panel unanimously voted 1
  - `weighted_majority_1`: 9 (0.2%) — weighted majority voted 1
- The `llm_per_voter_breakdown` column contains pipe-separated per-LLM vote strings like:
  ```
  gq-llama-4-scout-17b-16e/Meta=0|gq-gpt-oss-20b/OpenAI-OSS=0|gq-gpt-oss-120b/OpenAI-OSS=0|gq-qwen3-32b/Alibaba=0|ms-ministral-8b/Mistral=0|ms-nemo/Mistral=0|cb-llama3.1-8b/Meta=1|ms-small-latest/Mistral=...
  ```
- The prefixes appear to encode the inference provider:
  - `gq-` → Groq (fast LPU inference)
  - `ms-` → Mistral AI (La Plateforme) — or possibly Microsoft Azure? Need to confirm
  - `cb-` → Cerebras (fast wafer-scale inference)
  - `or-` → OpenRouter (multi-model gateway)
- The LLMs observed in `llm_per_voter_breakdown` (inferred from sample values):
  - `gq-llama-4-scout-17b-16e` / Meta (Llama 4 Scout, 17B params, 16 experts)
  - `gq-gpt-oss-20b` / OpenAI-OSS (GPT-OSS 20B)
  - `gq-gpt-oss-120b` / OpenAI-OSS (GPT-OSS 120B)
  - `gq-qwen3-32b` / Alibaba (Qwen 3, 32B)
  - `gq-llama4-scout` / Meta (alternative Llama 4 Scout deployment)
  - `gq-llama-3.1-8b` / Meta (Llama 3.1 8B)
  - `ms-ministral-8b` / Mistral (Ministral 8B)
  - `ms-nemo` / Mistral (Nemo — likely Mistral Nemo 12B)
  - `ms-small-latest` / Mistral (alias for current small Mistral model)
  - `cb-llama3.1-8b` / Meta (Llama 3.1 8B on Cerebras)
  - `or-gpt-oss-120b` / OpenAI-OSS (GPT-OSS 120B on OpenRouter)

### 5.2 What we don't know

> **** Document the LLM consensus pipeline:

- **Why only ~5% of articles?** The 94.1% `no_vote` rate means the LLM pipeline was intentionally run on only a targeted subset. Document:
  - The selection rule for the ~5% subset (was it the same articles that became the `new_*` edge-case batches? was it articles with low weak-label-vs-model agreement? was it a random sample?)
  - The cost / compute reason for the 5% cap (LLM API cost? time constraints? compute budget?)
  - Whether the remaining 95% will eventually be labeled by the LLM pipeline in a future release

- **The full LLM panel:** List the exact final LLM panel and which articles saw which LLMs. The `llm_per_voter_breakdown` column shows different LLM combinations for different articles — was the panel fixed, or did it vary over time as models were added/removed?

- **Voting protocol rules:**
  - What distinguishes `unanimous` vs `majority` vs `split`? (Is `unanimous` = all LLMs agree, `majority` = > 50% agree, `split` = no majority?)
  - What weights were used in `weighted_majority_*`? Were larger models weighted more?
  - What does `manual` mean — was the LLM label manually overridden by a human?

- **Prompt template:** What prompt was sent to each LLM? (e.g., "Read the following Bengali news article. Is this yellow journalism? Answer 0 or 1. Article: ...") — release the prompt template as a supplementary file.

- **Decoding parameters:** Temperature, max tokens, top-p for each LLM.

- **Cost:** Total API cost for the 295 articles × ~10 LLMs each ≈ 2,950 LLM calls. What was the total cost in USD?

- **LLM label quality:** What is the agreement rate between `llm_majority_label` and `human_label` on the 295 voted articles? Task 7's audit found 53.90% agreement — but since `llm_majority_label` is 99.5% zeros, this is essentially `0-vs-0` agreement and not informative. Compute the agreement on the 25 articles where `llm_majority_label = 1` specifically.

- **Provider prefix decoding:** Confirm the meaning of `gq-`, `ms-`, `cb-`, `or-`. Is `ms-` Mistral AI or Microsoft Azure? Is `cb-` Cerebras or another provider?

- **Failure modes:** Did any LLMs consistently refuse to label (e.g., safety refusals on sensitive-topic articles)? Did any LLMs time out or produce unparseable output?

---

## 6. Sampling Strategy for the `new_*` Batches

The 796 articles in the `new_*` batches (700 in corpus, 69 in gold) were specifically sampled to target articles where the weak supervision label and the preliminary model prediction disagreed (or where one signal was missing). This ensures the gold standard covers edge cases and disagreement regions rather than only easy agreements.

### 6.1 What we know

The batch tag naming convention is `new_{confidence-band}_w{weak-label}_p{model-prediction}`:

- `confidence-band ∈ {low, mid, high}` — binned from a confidence signal (likely `cosine_similarity_band` or weak-label confidence; the exact source is undocumented)
- `weak-label ∈ {w0, w1}` — `weak_supervision_label` value (0 or 1)
- `model-prediction ∈ {p0, p1, pany}` — `preliminary_model_label` value, or `pany` when any prediction is accepted

| Batch tag | Confidence band | Weak label | Model pred | Corpus count | Gold count |
|-----------|-----------------|------------|------------|--------------|------------|
| `new_low_w0_pany` | low | 0 | any | 100 | 19 |
| `new_mid_w0_p0` | mid | 0 | 0 | 100 | 16 |
| `new_mid_w0_p1` | mid | 0 | 1 | 1 | 0 |
| `new_mid_w1_p0` | mid | 1 | 0 | 100 | 10 |
| `new_mid_w1_p1` | mid | 1 | 1 | 100 | 14 |
| `new_high_w1_pany` | high | 1 | any | 100 | 10 |

**Observations:**
- The `new_mid_w1_p0` batch (weak label = 1, model = 0) is a **disagreement batch** — captures articles where weak supervision flagged yellow but the model disagreed.
- The `new_mid_w1_p1` batch (weak label = 1, model = 1) is an **agreement batch** — captures articles where both systems agree the article is yellow.
- The `new_mid_w0_p0` batch (weak label = 0, model = 0) is also an agreement batch, but on the non-yellow side.
- The `new_low_w0_pany` and `new_high_w1_pany` batches use `pany` (any model prediction), suggesting they were sampled based on weak-label confidence alone.
- The lone `new_mid_w0_p1` (1 article) is a degenerate category — likely an article that was originally in the batch but later removed, or an off-by-one sampling artifact.

### 6.2 What we don't know

> **** Document the sampling strategy:

- **Confidence band definition:** What signal determines `low`/`mid`/`high`? Candidates: (a) `cosine_similarity_band` (with `low`/`mid`/`high` mapping to the 8 bands somehow), (b) weak-label posterior confidence, (c) model-prediction probability. Document the exact mapping.

- **Sampling frame:** For each `new_*` batch, how many candidate articles were available in the underlying scraping population, and what fraction was sampled? (E.g., "1,000 candidate articles had weak_label=1 + model_prediction=0; we sampled 100 randomly with seed=42.")

- **Sampling method:** Random uniform? Stratified by outlet? Curated by a human? If random, what seed?

- **Why 100 per batch?** Was 100 chosen for budget reasons (annotation cost)? For statistical power? For balance?

- **Why is the gold-standard subset of each `new_*` batch so small (10–19)?** The gold standard drew only 69 articles total from the 700 `new_*` corpus articles. Was the gold-standard sampling proportional, stratified, or targeted?

- **Degenerate `new_mid_w0_p1`:** Why does this batch have only 1 article in the corpus and 0 in the gold standard? Was the batch intended to have 100 but the sampling frame yielded only 1 candidate?

- **`pany` semantics:** When a batch uses `pany` (any model prediction), does that mean the model prediction was *not used* as a sampling criterion (so both p0 and p1 articles are included), or does it mean the model prediction was *unknown* (missing) for those articles?

- **Coverage of the disagreement region:** The four-cell confusion matrix of `weak_supervision_label` × `preliminary_model_label` over the full corpus has counts (after subtracting the 4,204 `corpus_expansion_5000` + 295 `v17_reused` = 4,499 non-`new_*` articles): TP=279, TN=255, FP=194, FN=68 (total 796). Confirm that the 796 `new_*` articles are the entire TP+TN+FP+FN set, or whether some were subsampled.

---

## 7. Gold Standard Selection (766 from 5,000)

The gold standard `Swarabyanjan_Gold_Balanced_766.csv` is a balanced 1:1 (383 yellow + 383 non-yellow) subset of the 5,000-article corpus. All 766 gold article IDs appear in the corpus.

### 7.1 What we know

The gold standard's `corpus_batch` distribution (from Task 7's audit):

| Batch tag | Gold count | Corpus count | Gold/Corpus ratio |
|-----------|------------|--------------|-------------------|
| `corpus_expansion_5000` | 649 | 4,204 | 15.4% |
| `v17_reused` | 48 | 295 | 16.3% |
| `new_low_w0_pany` | 19 | 100 | 19.0% |
| `new_mid_w0_p0` | 16 | 100 | 16.0% |
| `new_mid_w1_p1` | 14 | 100 | 14.0% |
| `new_mid_w1_p0` | 10 | 100 | 10.0% |
| `new_high_w1_pany` | 10 | 100 | 10.0% |
| `new_mid_w0_p1` | 0 | 1 | 0% (degenerate) |
| **Total** | **766** | **5,000** | **15.3%** |

**Observations:**
- The gold standard samples ~15% of each batch (with some variation 10–19%), suggesting a roughly proportional stratified sample by `corpus_batch`.
- The gold standard is balanced 1:1 (383 yellow + 383 non-yellow) by `best_label` — the revised label.
- The corpus `human_label` (pre-revision) on the 766-overlap is **not** balanced: 338 zeros + 428 ones (see [Label Revision Protocol](./README.md#label-revision-protocol) in the main README). The 1:1 balance was achieved by the revision process, which flipped 131 zero→one and 176 one→zero.

### 7.2 What we don't know

> **** Document the gold-standard selection protocol:

- **Sampling rule:** Was the 766-article gold standard selected by:
  - (a) Stratified random sampling by `corpus_batch` (~15% per stratum)?
  - (b) Curated selection of "interesting" articles (edge cases, disagreements)?
  - (c) Convenience sampling (whatever the annotator had time to re-annotate)?
  - (d) A combination?

- **Balance target:** The gold standard is balanced 1:1 by `best_label` (revised), but the corpus `human_label` on the overlap is 338 zeros + 428 ones (44% yellow). Was the 1:1 balance:
  - (a) Achieved by the revision process flipping labels to hit 1:1?
  - (b) A sampling target (sample until 383 of each)?
  - (c) A post-hoc observation after the revision was complete?

- **Annotator capacity:** Was 766 chosen because it was the maximum the annotator could complete in the available time? Or for statistical power (e.g., to detect a 0.05 F1 difference with 80% power at α=0.05)?

- **Selection independence:** Were the 766 articles selected *before* or *after* the revision process? If after, the revision may have been biased toward articles that needed revision.

- **Stratification variables:** Beyond `corpus_batch`, was the gold standard stratified by:
  - Article length?
  - Topic / outlet?
  - Annotator confidence?
  - Weak-label-vs-model agreement?

- **Random seed:** If random sampling was used, what was the seed?

- **Annotation effort:** How long did the annotator spend per article? (Median minutes per article, total annotation hours.)

- **Annotator qualifications:** Document the annotator's identity and qualifications (Bengali native speaker? journalism background? academic affiliation? prior annotation experience?). If a committee, document all members.

- **Inter-annotator agreement:** If multiple annotators were involved, report Cohen's κ or Krippendorff's α on a calibration subset. If a single annotator, report test-retest reliability on a re-annotated subset.

- **Disagreement resolution:** If a committee, document the resolution process (majority vote? consensus discussion? senior-annotator tiebreak?).

- **Annotation guidelines:** Were written annotation guidelines used? Can they be released as a supplementary document?

- **Tooling:** What annotation tool was used (Label Studio, Prodigy, a custom spreadsheet, a Jupyter notebook)?

---

## 8. Version History

The corpus has evolved through multiple versions. The `corpus_batch` tags and `annotation_provenance` values document this evolution:

| Version | Description | Articles | Tag |
|---------|-------------|----------|-----|
| v17 | Original smaller gold standard (pre-expansion) | 295 retained in v20 | `v17_reused` |
| v18 | Article IDs assigned (`v18_XXXX` format) | — | (reflected in `article_id`) |
| v19 | Manual corrections (17 articles flipped from yellow to non-yellow factual news) | 17 | `v19-CORRECTED: factual news, not yellow journalism` |
| v20 | 5,000-article expansion with multi-signal sampling | 4,204 new + 179 dual-agreement-flip | `v20_corpus_expansion: multi_signal_sampling` and `v20_dual_agreement_flip: WL=1+MP=1` |

> **** Document the version timeline:
- Dates of each version (v17, v18, v19, v20)
- Whether the original v17 dataset is still available (Kaggle? GitHub release?)
- The rationale for the v18 → v19 → v20 progression
- Whether the article IDs were re-numbered at any version transition (the `v18_XXXX` prefix suggests IDs were assigned at v18 and never re-numbered)
- The `original_gold` provenance tag (600 articles) — does this refer to v17 articles that were not in the 295 retained in `v17_reused`? Or to a different gold-standard snapshot?

---

## 9. Audit Trail

This document is informed by:

- **Task 7's audit report** (an offline audit, ~1,125 lines, report not shipped) — a read-only audit of the original Kaggle CSVs covering Bengali text encoding, column quality, label sanity, duplicates, cross-dataset consistency, reviewer attack surface, and specific recommendations. The audit was deterministic (seed=7) and idempotent — re-running it on the cleaned CSVs will verify that all 10 smoking-gun findings have been addressed.
- **Task 8's cleaning pipeline** (offline, not shipped) — applies NFC normalization, zero-width character stripping, whitespace normalization, column renames, and the leading-space bug fix. Produces the three cleaned CSVs which are then committed to `data/`.
- **`notebooks/NB8_SMI_Annotation_Experiment.ipynb`** — defines the 8 SMI criteria scoring functions (C1–C8) with full Bengali lexicons, the weight-learning logistic regression, and the SMI automated annotation outputs.

For any reviewer question not answered by this document, consult the audit report's "Reviewer attack surface" section (Audit 6), which lists 12 specific questions a Q1 reviewer (TACL, EMNLP, IPM, etc.) would likely ask, annotated with whether each is currently answered.

---

## 10. Open Questions for Pre-Publication Review

The following `TODO` items must be resolved before Q1 submission. Each is a separate factual question for the dataset authors:

1. News outlets scraped (full list)
2. Time period of article collection (date range)
3. Geographic scope (Bangladesh-only? diaspora?)
4. Scraping methodology (RSS? HTML? API?)
5. Article selection within each outlet (sampling frame)
6. Language filtering rule
7. Deduplication key and methodology
8. License terms for the article text (fair-use rationale)
9. Weak supervision labeling functions (full LF list)
10. LF aggregation method (Snorkel? majority vote?)
11. LF precision/recall and coverage
12. Weak label confidence computation (low/mid/high bands)
13. Preliminary model name and architecture
14. Preliminary model training data (and any overlap with this corpus)
15. Preliminary model training date
16. Preliminary model hyperparameters
17. Preliminary model performance (F1, accuracy)
18. Preliminary model decision threshold
19. Circularity disclosure (was the model label shown to the human annotator?)
20. Embedding model (BanglaBERT? sentence-BERT? TF-IDF?)
21. Embedding input (headline? body? truncated?)
22. Centroid computation (mean? median? over which subset?)
23. Band boundaries for the 8 `cosine_similarity_band` categories
24. Centroid similarity purpose (diversity? deduplication? confidence?)
25. Why LLM consensus ran on only ~5% of articles
26. LLM panel membership (final list of LLMs)
27. Voting protocol rules (unanimous vs majority vs split vs manual)
28. Weighted majority weights
29. LLM prompt template
30. LLM decoding parameters (temperature, max tokens)
31. LLM API cost
32. LLM label quality (agreement with `human_label` on the 25 positive votes)
33. Provider prefix decoding (gq/ms/cb/or confirmation)
34. `new_*` batch confidence-band definition (low/mid/high source)
35. `new_*` batch sampling frame and method
36. Why 100 articles per `new_*` batch
37. Why the gold-standard subset of each `new_*` batch is so small (10–19)
38. Degenerate `new_mid_w0_p1` (1 article) explanation
39. `pany` semantics (any model prediction or unknown?)
40. Gold standard 766 selection rule (stratified? curated? convenience?)
41. 1:1 balance target (achieved by revision or by sampling?)
42. Annotator capacity (why 766?)
43. Selection before/after revision
44. Stratification variables (batch? length? topic?)
45. Random seed for sampling
46. Annotation effort (time per article)
47. Annotator identity and qualifications
48. Inter-annotator agreement (Cohen's κ / Krippendorff's α)
49. Disagreement resolution process
50. Annotation guidelines (can they be released?)
51. Annotation tool used
52. Revision markers in `best_note` (full list and meanings)
53. Annotator of `best_label` vs annotator of `human_label` (same person?)
54. Time elapsed between `human_label` and `best_label` annotation
55. Version timeline (v17 / v18 / v19 / v20 dates)
56. `original_gold` provenance tag meaning (600 articles)
57. Lexicon construction process (manual? seed expansion?)
58. Lexicon inter-annotator agreement
59. Whether lexicons are released as separate files
60. Rationale for the 1,200-character truncation threshold
61. Whether truncation introduces label-distribution bias
62. Treatment of stub articles (released as-is or filtered?)
63. Whether stub articles are genuine missing-body cases or scraping failures

Each `TODO` in this document corresponds to one or more of the above questions.

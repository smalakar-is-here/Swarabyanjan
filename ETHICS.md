# Ethics Statement

## Dataset Composition
The Swarabyanjan dataset contains Bengali news articles collected from publicly available online news sources. The articles were sampled to ensure representation of both yellow journalism and non-yellow journalism cases.

## Annotation
The dataset was annotated by the first author (Swagotam Malakar), a Bengali native speaker with background in computational linguistics. The annotation process followed an 8-criteria linguistic protocol (detailed in `data/README.md` and implemented in `notebooks/NB8_SMI_Annotation_Experiment.ipynb`).

**AI-assisted validation:** During the revision phase, AI language models were used to cross-check annotation reasoning and identify potential inconsistencies. The AI did not make labeling decisions; all final labels were determined by the human annotator. This approach is consistent with recent practices in NLP data annotation.

**IRB approval:** This study uses publicly available news articles published by Bengali news outlets. It does not involve human subjects, personal data collection from individuals, or interventions. Institutional Review Board (IRB) approval was not required as the research involves only publicly available text content and computational analysis.

**Inter-annotator agreement:** A single annotator performed the initial annotation. The revision from `human_label` to `best_label` was performed by the same annotator after developing the 8-criteria protocol. Inter-annotator agreement was not computed as only one annotator was involved. The 59.9% disagreement rate between the pre-revision and post-revision labels reflects the impact of the structured protocol, not inter-annotator variance.

**Annotation guidelines:** The full annotation protocol is documented in `data/README.md` under "8-Criteria Annotation Protocol" and in `data/CORPUS_CONSTRUCTION.md`.

## Privacy
The `best_note` column in the gold standard CSV contains annotation rationales that may reference named individuals (e.g., politicians, crime victims, accused persons). These references are drawn from the original news articles and are part of the public record. Researchers using this dataset should be aware of potential privacy implications.

## Potential Misuse
This dataset is designed for academic research in computational linguistics and journalism studies. It should not be used to:
- Target or harass individuals named in the articles
- Automatically censor or suppress news content
- Build systems that could be used for propaganda or misinformation

## Content Warning
Some articles contain descriptions of violence, sexual assault, and other sensitive topics. The SMI criterion C8 (Sensitive Topic) flags articles covering communal, religious, gender, ethnic, or politically provocative themes.

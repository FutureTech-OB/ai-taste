# Human Ratings

This folder contains the de-identified human evaluation data used in the release.

Primary public files:

- `expert_ratings_deidentified.jsonl`
- `student_ratings_deidentified.jsonl`

Canonical reproducibility files:

- `reproducibility/expert_reproducibility.jsonl`
- `reproducibility/expert_reproducibility_filtered.jsonl`
- `reproducibility/student_reproducibility.jsonl`
- `reproducibility/student_reproducibility_filtered.jsonl`

The student filtered reproducibility file retains only the background fields used for the released descriptive table: `gender`, `phd_year`, `publications`, `review_experience`, and `ai_familiarity`.

All files in this release are de-identified. Direct participant identifiers such as names, response IDs, IP addresses, and raw timestamps are not included.

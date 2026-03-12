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

The reproducibility files retain stable anonymous `rater_id` values plus rating,
confidence, and familiarity fields needed for person-level aggregation in the
released analyses. Participant profile fields and cohort labels are not included
in the public package.

All files in this release are de-identified. Direct participant identifiers such as names, response IDs, IP addresses, and raw timestamps are not included.

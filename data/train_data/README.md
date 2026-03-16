# Training Data (Placeholder)

The supervised fine-tuning training data described in the manuscript will be released upon publication.

The training corpus comprises two institutional-trace slices drawn from a 19-journal source universe, fully disjoint from the 120 benchmark pitches:

- **Recent/new slice**: 4,479 processed research-pitch/journal-outcome pairs (main SFT models)
- **Older slice**: 3,368 pairs (temporal-persistence comparison)

Each record contains `title`, `published_year`, `journal`, `type`, `rank`, and a research-question-with-context pitch text. See the Methods section of the manuscript for full details on training corpus construction and leakage controls.

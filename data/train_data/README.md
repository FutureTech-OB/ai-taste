# Train Data

This directory contains the released article-level training-data pools used to prepare the supervised fine-tuning inputs described in the manuscript.

- `RIOB.Article.json`: released recent-trace article pool used for the main SFT preparation path
- `RIOB_old.Article.json`: released older-trace article pool used for the temporal-persistence SFT preparation path

Each record exposes only `title`, `published_year`, `journal`, `type`, `rank`, and an `entries` object. When a released pitch text is available, `entries` contains `rq_with_context`; otherwise `entries` is empty or carries a null `rq_with_context`.

This repository does not include model checkpoints, raw scraping artifacts, or unreleased preprocessing intermediates beyond these shipped `train_data` files.

# De-Identification

This note summarizes only the privacy-relevant transformations visible in the
released package.

The released human-rating files use stable pseudonyms for rater joins. The package does not include names, email addresses, institutions, IP addresses, survey-system identifiers, recruitment messages, compensation records, or raw survey exports.

Article titles and journal names are retained because they define the benchmark items and are necessary for reproducibility.

Provider-local fine-tune identifiers are replaced with stable release aliases in prediction files.

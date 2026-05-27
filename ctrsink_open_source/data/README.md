# Data Preparation

Place processed train/validation/test files here or pass absolute paths to the
training script.

Minimum columns:

```text
positive_movie_titles
negative_movie_titles
title
genres
click_label
```

Supported formats:

- `.csv`
- `.jsonl`
- `.json`
- `.parquet`

For MovieLens, construct positive and negative history sequences per user, split
by time into train/valid/test, and use `click_label = 1` for ratings above the
chosen threshold and `0` otherwise.

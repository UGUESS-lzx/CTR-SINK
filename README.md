# CTR-Sink [KDD 2026]

[中文 README](README_zh.md)

This repository is a self-contained open-source implementation of CTR-Sink for
language-model-based click-through rate prediction. It removes all internal
ALPS, ODPS, Aistudio, Pangu, NAS from the original
research code.


## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Recommended Transformers versions by backbone:

| Backbone | Transformers version |
| --- | --- |
| RoBERTa / BERT | `transformers==4.29.2` |
| Qwen2 | `transformers==4.44.2` |
| Qwen3 | latest stable Transformers |

```bash
# RoBERTa / BERT
pip install "transformers==4.29.2"

# Qwen2
pip install "transformers==4.44.2"

# Qwen3
pip install -U transformers
```

## Data Format

The default schema matches the MovieLens experiments:

```text
positive_movie_titles,negative_movie_titles,title,genres,click_label
```

Behavior fields may be comma-separated strings:

```text
Toy Story (1995),Jumanji (1995),Heat (1995)
```

or Python/JSON-style lists:

```text
["Toy Story (1995)", "Jumanji (1995)", "Heat (1995)"]
```

See `examples/sample_data/` for a tiny smoke-test dataset.

## Reproduce Main Variants

Validation is run at the end of every epoch during training. The main metric is
reported as `valid_auc`, matching the original research code.

Original LM-CTR baseline:

```bash
python scripts/train.py \
  --config configs/qwen2_movielens_original.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen2_original
```

CTR-Sink with two-stage training:

```bash
python scripts/train.py \
  --config configs/qwen2_movielens_ctrsink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen2_ctrsink
```

CTR-Sink with inter-sink attention enhancement:

```bash
python scripts/train.py \
  --config configs/qwen2_movielens_ctrsink_inter_sink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen2_ctrsink_inter_sink
```

Qwen2-7B CTR-Sink:

```bash
python scripts/train.py \
  --config configs/qwen2_7b_movielens_ctrsink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen2_7b_ctrsink
```

Qwen2-7B CTR-Sink with inter-sink attention enhancement:

```bash
python scripts/train.py \
  --config configs/qwen2_7b_movielens_ctrsink_inter_sink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen2_7b_ctrsink_inter_sink
```

Qwen3-8B CTR-Sink:

```bash
python scripts/train.py \
  --config configs/qwen3_8b_movielens_ctrsink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen3_8b_ctrsink
```

Qwen3-8B CTR-Sink with inter-sink attention enhancement:

```bash
python scripts/train.py \
  --config configs/qwen3_8b_movielens_ctrsink_inter_sink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen3_8b_ctrsink_inter_sink
```

RoBERTa CTR-Sink with inter-sink attention enhancement:

```bash
python scripts/train.py \
  --config configs/roberta_movielens_ctrsink_inter_sink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/roberta_ctrsink_inter_sink
```

The sample data is only for checking that the pipeline runs. For paper-level
results, prepare the full MovieLens or KuaiRec splits and pass their paths to
`--train_file` and `--valid_file`.

## Important Defaults

- `use_split_layer = true`
- MovieLens prompts follow the original code by default: `用户打高分的电影序列为:`, `用户打低分的电影序列为:`, `电影名称为:`, and `电影类型为:`.
- `use_split_layer` follows the Singleton CTR (SCTR) and BAHE setting to reduce training cost. In our tests, it does not have a large impact on final performance, so it is enabled by default.
- Qwen2-7B / Qwen3-8B: use LoRA, `model_dtype = bf16`, `precision = bf16`, gradient checkpointing, and a small per-device batch size.
- Transformers version: use `transformers==4.29.2` for RoBERTa / BERT, `transformers==4.44.2` for Qwen2, and the latest stable Transformers for Qwen3.
- `use_inter_sink_attention` requires `use_sink = true` and `use_split_layer = true`.
- Non-two-stage training: 3 epochs.
- Two-stage training: 3 sink-only epochs + 3 all-token epochs.



## Citation

If you use this code, please cite the CTR-Sink paper.

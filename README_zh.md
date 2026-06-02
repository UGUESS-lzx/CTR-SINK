# CTR-Sink

本仓库是 CTR-Sink 的独立开源实现，用于基于语言模型的点击率预测
（LM-based CTR prediction）。相比原始研究代码，本版本移除了内部训练平台、
私有数据读取器、私有模型保存逻辑，保留论文方法中可复现、
可公开发布的核心训练流程。

- 支持 MovieLens / KuaiRec 风格的文本 CTR 数据。
- 支持基于公开 `facebook/contriever` checkpoint 的行为检索。
- 支持行为级 `[SINK]` token 和外部信号注入。
- 支持 CTR-Sink 的 two-stage 训练流程。
- 支持 Qwen2、Qwen3 和 BERT/RoBERTa 的 inter-sink attention 增强。
- 默认启用 split-layer 行为编码，即 `use_split_layer = true`。
- 支持 CSV、JSONL、JSON 和 Parquet 数据格式。
- 提供 Qwen2、RoBERTa、CTR-Sink 和 ablation 配置示例。

## 代码结构

```text
ctrsink_open_source/
├── configs/                 # 实验配置
├── ctrsink/                 # 核心 Python package
│   ├── config.py            # 训练参数定义
│   ├── data.py              # 数据读取和 batch 构造
│   ├── retrieval.py         # Contriever 行为检索
│   ├── modeling/            # 模型结构
│   └── training.py          # 训练与评估循环
├── data/                    # 数据准备说明
├── examples/sample_data/    # 最小样例数据
├── scripts/train.py         # 训练入口
├── scripts/evaluate.py      # 可选 checkpoint 验证入口
├── requirements.txt
└── pyproject.toml
```

## 安装

建议使用独立 Python 环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

不同 backbone 依赖的 Transformers 版本不同，建议按实验模型选择对应版本：

| Backbone | 推荐 Transformers 版本 |
| --- | --- |
| RoBERTa / BERT | `transformers==4.29.2` |
| Qwen2 | `transformers==4.44.2` |
| Qwen3 | 最新稳定版 Transformers |

示例：

```bash
# RoBERTa / BERT
pip install "transformers==4.29.2"

# Qwen2
pip install "transformers==4.44.2"

# Qwen3
pip install -U transformers
```

如果使用 Qwen2 / Qwen3 + LoRA，建议准备支持 `bfloat16` 的 GPU。CPU 可以用于检查数据
和小规模 smoke test，但不适合完整训练。

## 数据格式

默认数据 schema 对齐 MovieLens 实验：

```text
positive_movie_titles,negative_movie_titles,title,genres,click_label
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `positive_movie_titles` | 用户历史正反馈物品序列 |
| `negative_movie_titles` | 用户历史负反馈物品序列 |
| `title` | 当前候选物品标题 |
| `genres` | 当前候选物品类别（代码中没有用到） |
| `click_label` | CTR 标签，通常为 `0` 或 `1` |

行为序列可以写成逗号分隔字符串：

```text
Toy Story (1995),Jumanji (1995),Heat (1995)
```

也可以写成 Python / JSON 风格列表：

```text
["Toy Story (1995)", "Jumanji (1995)", "Heat (1995)"]
```

仓库中提供了最小样例数据：

```text
examples/sample_data/train.csv
examples/sample_data/valid.csv
```

样例数据只用于检查训练流程能否跑通，不用于复现论文指标。若要复现论文结果，
需要按照论文设置准备完整的 MovieLens 或 KuaiRec train / valid / test 划分。

## 快速开始

训练原始 LM-CTR baseline：

```bash
python scripts/train.py \
  --config configs/qwen2_movielens_original.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen2_original
```

训练 CTR-Sink：

```bash
python scripts/train.py \
  --config configs/qwen2_movielens_ctrsink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen2_ctrsink
```

训练带 inter-sink attention 的 CTR-Sink：

```bash
python scripts/train.py \
  --config configs/qwen2_movielens_ctrsink_inter_sink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/qwen2_ctrsink_inter_sink
```

训练 RoBERTa 版本：

```bash
python scripts/train.py \
  --config configs/roberta_movielens_ctrsink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/roberta_ctrsink
```

训练带 inter-sink attention 的 RoBERTa 版本：

```bash
python scripts/train.py \
  --config configs/roberta_movielens_ctrsink_inter_sink.json \
  --train_file examples/sample_data/train.csv \
  --valid_file examples/sample_data/valid.csv \
  --output_dir outputs/roberta_ctrsink_inter_sink
```

## 训练过程中的验证

原始代码没有单独的 test 脚本，验证是在训练过程中完成的：每个 epoch 结束后，
模型会在 validation set 上计算并输出 `valid_auc`。本开源版本也沿用这个习惯，
训练日志会显示类似下面的内容：

```text
epoch=1 stage=sink_only train_loss=0.693214 valid_loss=0.691882 valid_auc=0.7315
```

训练会在 `output_dir` 下保存 `best.pt`、`last.pt` 和 `metrics.json`，其中
`best.pt` 按 `valid_auc` 选择。

仓库额外保留了一个轻量的 checkpoint 验证脚本，方便开源用户复查已保存模型。
它不是原始训练流程的必要部分：

```bash
python scripts/evaluate.py \
  --checkpoint outputs/qwen2_ctrsink/best.pt \
  --valid_file examples/sample_data/valid.csv
```

## Two-Stage 训练

论文中的 two-stage 训练一共训练 6 个 epoch：

| 阶段 | epoch 数 | 训练目标 |
| --- | --- | --- |
| Stage 1 | 3 | 只训练 `[SINK]` token 相关表示 |
| Stage 2 | 3 | 训练完整行为 token 表示 |

对应配置为：

```json
{
  "two_stage": true,
  "stage1_epochs": 3,
  "stage2_epochs": 3
}
```

非 two-stage baseline 默认训练 3 个 epoch：

```json
{
  "two_stage": false,
  "epochs": 3
}
```

## 重要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `backbone` | `Qwen/Qwen2-0.5B` | 主干语言模型 |
| `tuning_method` | `lora` | 微调方式，可选 `lora` 或 `full` |
| `model_dtype` | `auto` | 加载 backbone 时使用的参数精度，可选 `auto`、`bf16`、`fp16`、`fp32` |
| `use_split_layer` | `true` | 是否使用 split-layer 行为编码，用于节省训练开销 |
| `use_retrieval` | `false` | 是否对用户历史行为做检索筛选 |
| `retriever_model` | `facebook/contriever` | 检索模型 |
| `use_sink` | `false` | 是否插入行为级 `[SINK]` token |
| `sink_signal` | `temporal` | `[SINK]` 外部信号，可选 `temporal`、`similarity`、`random`、`none` |
| `use_inter_sink_attention` | `false` | 是否启用 inter-sink attention，需要同时开启 `use_sink` 和 `use_split_layer` |
| `inter_sink_layers` | `0` | inter-sink attention 作用层数，`0` 表示按 backbone 自动选择 |
| `num_behaviors` | `50` | 每个用户最多保留的历史行为数 |
| `max_user_tokens` | `512` | 用户历史文本最大 token 长度 |
| `max_item_tokens` | `32` | 候选物品文本最大 token 长度 |
| `batch_size` | `16` | 训练 batch size |
| `lr` | `1e-5` | 学习率 |
| `warmup_ratio` | `0.05` | warmup 比例 |

默认 MovieLens prompt 与原始代码保持一致：

| 字段 | 默认 prompt |
| --- | --- |
| `positive_movie_titles` | `用户打高分的电影序列为:` |
| `negative_movie_titles` | `用户打低分的电影序列为:` |
| `title` | `电影名称为:` |
| `genres` | `电影类型为:` |

这些 prompt 对应配置项为 `positive_prompt`、`negative_prompt`、`title_prompt`
和 `genres_prompt`，可以在 JSON config 或命令行参数中覆盖。

`use_split_layer` 默认开启。如果需要关闭，可以使用：

```bash
python scripts/train.py \
  --config configs/qwen2_movielens_ctrsink.json \
  --no_split_layer
```

`use_split_layer` 是我们参考 Singleton CTR（SCTR）和 BAHE 的设置引入的，
主要目的是降低长行为序列带来的训练开销。根据我们的测试，开启该设置对最终
性能没有很大影响，因此开源版本默认保留 `use_split_layer=true`。

## 与原始代码的关系

原始代码是内部 ALPS 训练任务提交脚本。其中包含大量内部环境参数，例如队列、镜像、ODPS 表、
NAS 路径和私有保存逻辑。这些内容不适合进入开源仓库。


## 推荐实验配置

Qwen2 CTR-Sink：

```json
{
  "backbone": "Qwen/Qwen2-0.5B",
  "tuning_method": "lora",
  "use_split_layer": true,
  "use_retrieval": true,
  "use_sink": true,
  "sink_signal": "temporal",
  "two_stage": true,
  "stage1_epochs": 3,
  "stage2_epochs": 3,
  "batch_size": 16,
  "lr": 1e-5,
  "warmup_ratio": 0.05
}
```

Qwen2 CTR-Sink + inter-sink attention：

```json
{
  "backbone": "Qwen/Qwen2-0.5B",
  "tuning_method": "lora",
  "use_split_layer": true,
  "use_retrieval": true,
  "use_sink": true,
  "use_inter_sink_attention": true,
  "sink_signal": "temporal",
  "two_stage": true,
  "stage1_epochs": 3,
  "stage2_epochs": 3,
  "batch_size": 16,
  "lr": 1e-5,
  "warmup_ratio": 0.05
}
```

Qwen2-7B CTR-Sink：

```json
{
  "backbone": "Qwen/Qwen2-7B",
  "tuning_method": "lora",
  "model_dtype": "bf16",
  "use_split_layer": true,
  "use_retrieval": true,
  "use_sink": true,
  "sink_signal": "temporal",
  "two_stage": true,
  "stage1_epochs": 3,
  "stage2_epochs": 3,
  "batch_size": 1,
  "eval_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "lr": 1e-5,
  "precision": "bf16",
  "gradient_checkpointing": true
}
```

Qwen3-8B CTR-Sink：

```json
{
  "backbone": "Qwen/Qwen3-8B",
  "tuning_method": "lora",
  "model_dtype": "bf16",
  "use_split_layer": true,
  "use_retrieval": true,
  "use_sink": true,
  "sink_signal": "temporal",
  "two_stage": true,
  "stage1_epochs": 3,
  "stage2_epochs": 3,
  "batch_size": 1,
  "eval_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "lr": 1e-5,
  "precision": "bf16",
  "gradient_checkpointing": true
}
```

RoBERTa CTR-Sink：

```json
{
  "backbone": "uer/chinese_roberta_L-4_H-512",
  "tuning_method": "full",
  "use_split_layer": true,
  "use_retrieval": true,
  "use_sink": true,
  "sink_signal": "temporal",
  "two_stage": true,
  "stage1_epochs": 3,
  "stage2_epochs": 3,
  "batch_size": 64,
  "lr": 1e-4,
  "warmup_ratio": 0.05
}
```

RoBERTa CTR-Sink + inter-sink attention：

```json
{
  "backbone": "uer/chinese_roberta_L-4_H-512",
  "tuning_method": "full",
  "use_split_layer": true,
  "use_retrieval": true,
  "use_sink": true,
  "use_inter_sink_attention": true,
  "sink_signal": "temporal",
  "two_stage": true,
  "stage1_epochs": 3,
  "stage2_epochs": 3,
  "batch_size": 64,
  "lr": 1e-4,
  "warmup_ratio": 0.05
}
```

## 输出文件

训练完成后，`output_dir` 中会包含：

| 文件 | 说明 |
| --- | --- |
| `config.json` | 实际使用的训练配置 |
| `best.pt` | 验证集 `valid_auc` 最优 checkpoint |
| `last.pt` | 最后一个 epoch 的 checkpoint |
| `metrics.json` | 最优 epoch 和最后 epoch 的训练 / 验证指标 |


## 引用

如果你使用本仓库，请引用 CTR-Sink 论文。

```bash
@article{li2025ctr,
  title={CTR-Sink: Attention Sink for Language Models in Click-Through Rate Prediction},
  author={Li, Zixuan and Geng, Binzong and Xiong, Jing and He, Yong and Hu, Yuxuan and Chen, Jian and Chen, Dingwei and Chang, Xiyu and Zhang, Liang and Mo, Linjian and others},
  journal={arXiv preprint arXiv:2508.03668},
  year={2025}
}
```

#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ctrsink.config import TrainConfig
from ctrsink.training import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTR-Sink.")
    parser.add_argument("--config", default=None, help="Path to a JSON config.")
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--valid_file", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--retriever_model", default=None)
    parser.add_argument("--tuning_method", choices=["full", "lora"], default=None)
    parser.add_argument("--model_dtype", choices=["auto", "bf16", "fp16", "16", "fp32"], default=None)
    parser.add_argument("--use_split_layer", action="store_true", default=None)
    parser.add_argument("--no_split_layer", action="store_true")
    parser.add_argument("--use_retrieval", action="store_true", default=None)
    parser.add_argument("--use_sink", action="store_true", default=None)
    parser.add_argument("--use_cls_token", action="store_true", default=None)
    parser.add_argument("--use_inter_sink_attention", action="store_true", default=None)
    parser.add_argument("--inter_sink_layers", type=int, default=None)
    parser.add_argument("--sink_signal", choices=["temporal", "similarity", "random", "none"], default=None)
    parser.add_argument("--sink_token", default=None)
    parser.add_argument("--sink_num_embeddings", type=int, default=None)
    parser.add_argument("--sink_embedding_dim", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--two_stage", action="store_true", default=None)
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--stage2_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--max_user_tokens", type=int, default=None)
    parser.add_argument("--max_item_tokens", type=int, default=None)
    parser.add_argument("--num_behaviors", type=int, default=None)
    parser.add_argument("--emb_output_size", type=int, default=None)
    parser.add_argument("--ctr_hidden_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16", "16"], default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--user_positive_col", default=None)
    parser.add_argument("--user_negative_col", default=None)
    parser.add_argument("--item_title_col", default=None)
    parser.add_argument("--item_genres_col", default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--positive_prompt", default=None)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--title_prompt", default=None)
    parser.add_argument("--genres_prompt", default=None)
    parser.add_argument("--save_every_epoch", action="store_true", default=None)
    return parser.parse_args()


def overrides_from_args(args) -> Dict:
    values = vars(args).copy()
    values.pop("config", None)
    no_split = values.pop("no_split_layer", False)
    if no_split:
        values["use_split_layer"] = False
    return {k: v for k, v in values.items() if v is not None}


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    cfg.update(overrides_from_args(args))
    metrics = train(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

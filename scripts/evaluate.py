#!/usr/bin/env python
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ctrsink.config import TrainConfig
from ctrsink.data import TextCtrCollator, TextCtrDataset, read_table
from ctrsink.modeling.model import CtrSinkModel
from ctrsink.training import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CTR-Sink checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = TrainConfig()
    cfg.update(checkpoint["config"])
    cfg.valid_file = args.valid_file
    if args.batch_size is not None:
        cfg.eval_batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CtrSinkModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)

    dataset = TextCtrDataset(read_table(args.valid_file), cfg)
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=TextCtrCollator())
    metrics = evaluate(model, loader, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

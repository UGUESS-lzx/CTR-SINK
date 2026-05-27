import ast
import math
import os
from typing import Dict, List, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import TrainConfig


def read_table(path: str) -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported data suffix: {suffix}")


def parse_behaviors(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    sep = "|SEP|" if "|SEP|" in text else ","
    return [x.strip().strip("'\"") for x in text.split(sep) if x.strip()]


class TextCtrDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg: TrainConfig):
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict:
        row = self.frame.iloc[idx]
        return {
            "positive": row.get(self.cfg.user_positive_col, ""),
            "negative": row.get(self.cfg.user_negative_col, ""),
            "title": str(row.get(self.cfg.item_title_col, "")),
            "genres": str(row.get(self.cfg.item_genres_col, "")),
            "label": float(row.get(self.cfg.label_col, 0)),
        }


class TextCtrCollator:
    def __call__(self, samples: Sequence[Dict]) -> Dict:
        return {
            "positive": [x["positive"] for x in samples],
            "negative": [x["negative"] for x in samples],
            "title": [x["title"] for x in samples],
            "genres": [x["genres"] for x in samples],
            "labels": torch.tensor([x["label"] for x in samples], dtype=torch.float32),
        }

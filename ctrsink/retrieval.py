from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ContrieverRetriever:
    def __init__(self, model_name: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(out.dtype)
        emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(emb, p=2, dim=-1)

    @torch.no_grad()
    def rank(self, behaviors: List[str], query: str, top_k: int) -> Tuple[List[str], List[int], List[float]]:
        if not behaviors:
            return [], [], []
        texts = [query] + behaviors
        emb = self.encode(texts)
        sims = torch.matmul(emb[0], emb.T).float().cpu().numpy()
        order = [int(i) for i in np.argsort(sims)[::-1] if int(i) != 0]
        order = order[:top_k]
        return [behaviors[i - 1] for i in order], [i - 1 for i in order], [float(sims[i]) for i in order]

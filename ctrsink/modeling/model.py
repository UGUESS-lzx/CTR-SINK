import random
import inspect
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

from ..config import TrainConfig
from ..data import parse_behaviors
from ..retrieval import ContrieverRetriever
from .heads import CtrHead, ProjectionHead, masked_mean
from .inter_sink import enable_bert_inter_sink_attention, enable_qwen2_inter_sink_attention, enable_qwen3_inter_sink_attention


FEATURES = ["positive", "negative", "title", "genres"]


def resolve_torch_dtype(dtype: str):
    if dtype in {"auto", "", None}:
        return "auto"
    if dtype == "bf16":
        return torch.bfloat16
    if dtype in {"fp16", "16"}:
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported model_dtype: {dtype}")


class CtrSinkModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.is_qwen2 = "qwen2" in cfg.backbone.lower()
        self.is_qwen3 = "qwen3" in cfg.backbone.lower()
        self.is_qwen = self.is_qwen2 or self.is_qwen3

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"
        self.sink_token = self.tokenizer.cls_token if cfg.use_cls_token and self.tokenizer.cls_token else cfg.sink_token

        added = 0
        if cfg.use_sink and self.sink_token not in self.tokenizer.get_vocab():
            added = self.tokenizer.add_special_tokens({"additional_special_tokens": [self.sink_token]})
        self.sink_token_id = self.tokenizer.convert_tokens_to_ids(self.sink_token)

        self.encoder = AutoModel.from_pretrained(
            cfg.backbone,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=resolve_torch_dtype(cfg.model_dtype),
        )
        if added:
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False
        if cfg.gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers = getattr(self.encoder.config, "num_hidden_layers", None)
        if cfg.use_split_layer and self.num_layers is None:
            raise ValueError("use_split_layer requires config.num_hidden_layers.")

        if cfg.use_inter_sink_attention:
            if self.is_qwen3:
                layer_count = cfg.inter_sink_layers or max(1, self.num_layers // 3)
                enable_qwen3_inter_sink_attention(self.encoder, layer_count)
            elif self.is_qwen2:
                layer_count = cfg.inter_sink_layers or self.num_layers // 2
                enable_qwen2_inter_sink_attention(self.encoder, layer_count)
            else:
                layer_count = cfg.inter_sink_layers or min(2, self.num_layers)
                enable_bert_inter_sink_attention(self.encoder, layer_count)

        if cfg.tuning_method == "lora":
            if get_peft_model is None:
                raise ImportError("peft is required for LoRA tuning.")
            self.encoder = get_peft_model(
                self.encoder,
                LoraConfig(
                    r=32,
                    lora_alpha=64 if self.is_qwen2 else 32,
                    lora_dropout=0.1,
                    bias="none",
                    target_modules=self._lora_targets(),
                ),
            )

        self.retriever = None
        if cfg.use_retrieval:
            self.retriever = ContrieverRetriever(cfg.retriever_model, self.device)

        self.sink_embedding = nn.Sequential(
            nn.Embedding(cfg.sink_num_embeddings, cfg.sink_embedding_dim),
            nn.Linear(cfg.sink_embedding_dim, self.hidden_size),
        )
        self.projectors = nn.ModuleDict(
            {name: ProjectionHead(self.hidden_size, cfg.emb_output_size) for name in FEATURES}
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(len(FEATURES) * cfg.emb_output_size, cfg.emb_output_size),
            nn.LayerNorm(cfg.emb_output_size),
            nn.LeakyReLU(),
        )
        self.ctr_head = CtrHead(cfg.emb_output_size, cfg.ctr_hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _lora_targets(self) -> List[str]:
        if self.is_qwen:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
        return ["query", "key", "value"]

    def _base_encoder(self):
        return self.encoder.get_base_model() if hasattr(self.encoder, "get_base_model") else self.encoder

    def _embeddings(self):
        return self._base_encoder().get_input_embeddings()

    def _rank_behaviors(self, behaviors: List[str], query: str) -> Tuple[List[str], List[int], List[float]]:
        if self.retriever is None:
            behaviors = behaviors[: self.cfg.num_behaviors]
            return behaviors, list(range(len(behaviors))), [0.0 for _ in behaviors]
        if self.retriever.device != self.device:
            self.retriever.model.to(self.device)
        return self.retriever.rank(behaviors, query, self.cfg.num_behaviors)

    def _signal_ids(self, positions: List[int], similarities: List[float]) -> List[int]:
        n = self.cfg.sink_num_embeddings
        if self.cfg.sink_signal == "none":
            return [0 for _ in positions]
        if self.cfg.sink_signal == "random":
            return [random.randrange(n) for _ in positions]
        if self.cfg.sink_signal == "similarity":
            if not similarities:
                return []
            lo, hi = min(similarities), max(similarities)
            if abs(hi - lo) < 1e-12:
                return [n - 1 for _ in similarities]
            return [max(0, min(n - 1, int((x - lo) / (hi - lo) * (n - 1)))) for x in similarities]
        return [max(0, min(n - 1, int(p))) for p in positions]

    def _prepare_behaviors(self, values: Sequence, titles: Sequence[str]) -> Tuple[List[List[str]], List[List[int]]]:
        all_behaviors, all_signals = [], []
        for value, title in zip(values, titles):
            behaviors, positions, similarities = self._rank_behaviors(parse_behaviors(value), str(title))
            all_behaviors.append(behaviors)
            all_signals.append(self._signal_ids(positions, similarities))
        return all_behaviors, all_signals

    def _format_user_texts(self, prompt: str, behaviors: List[List[str]]) -> List[str]:
        texts = []
        for row in behaviors:
            sep = self.sink_token + "," if self.cfg.use_sink else ","
            texts.append(prompt + sep.join(row))
        return texts

    def _tokenize(self, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _replace_sink_embeddings(
        self, input_ids: torch.Tensor, signal_ids: Optional[List[List[int]]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        embeds = self._embeddings()(input_ids)
        sink_positions = []
        if not self.cfg.use_sink:
            return embeds, sink_positions
        for row_idx in range(input_ids.size(0)):
            positions = torch.where(input_ids[row_idx] == self.sink_token_id)[0]
            sink_positions.append(positions)
            if positions.numel() == 0:
                continue
            ids = signal_ids[row_idx] if signal_ids is not None else []
            ids = ids[: positions.numel()]
            ids = ids + [0] * (positions.numel() - len(ids))
            signal_tensor = torch.tensor(ids, dtype=torch.long, device=input_ids.device)
            embeds[row_idx, positions] = self.sink_embedding(signal_tensor).to(embeds.dtype)
        return embeds, sink_positions

    def _manual_causal_mask(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        dtype = inputs_embeds.dtype if inputs_embeds.dtype.is_floating_point else torch.float32
        min_value = torch.finfo(dtype).min
        causal = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=inputs_embeds.device)
        causal = torch.triu(causal, diagonal=1)
        causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()
        padding_mask = attention_mask[:, None, None, :].eq(0)
        return causal.masked_fill(padding_mask, min_value)

    def _causal_inputs(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        base = self._base_encoder()
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        if hasattr(base, "_update_causal_mask"):
            try:
                causal_mask = base._update_causal_mask(attention_mask, inputs_embeds, cache_position, None, False)
                return causal_mask, position_ids, cache_position
            except TypeError:
                pass
        causal_mask = self._manual_causal_mask(inputs_embeds, attention_mask)
        return causal_mask, position_ids, cache_position

    def _run_qwen_layers(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        start_layer: int,
        end_layer: int,
        sink_positions: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        base = self._base_encoder()
        hidden = inputs_embeds
        causal_mask, position_ids, cache_position = self._causal_inputs(inputs_embeds, attention_mask)
        position_embeddings = base.rotary_emb(hidden, position_ids) if hasattr(base, "rotary_emb") else None
        for layer in base.layers[start_layer:end_layer]:
            residual = hidden
            normed = layer.input_layernorm(hidden)
            use_position_embeddings = "position_embeddings" in inspect.signature(layer.self_attn.forward).parameters
            if use_position_embeddings:
                attn_kwargs = {
                    "attention_mask": causal_mask,
                    "position_embeddings": position_embeddings,
                    "past_key_value": None,
                    "cache_position": cache_position,
                }
            else:
                attn_kwargs = {
                    "attention_mask": causal_mask,
                    "position_ids": position_ids,
                    "past_key_value": None,
                    "output_attentions": False,
                    "use_cache": False,
                }
            if hasattr(layer.self_attn, "sink_interaction"):
                attn_kwargs["special_item_pos"] = sink_positions
            attn_out = layer.self_attn(normed, **attn_kwargs)[0]
            hidden = residual + attn_out
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        return base.norm(hidden)

    def _run_bert_layers(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        start_layer: int,
        end_layer: int,
        add_embeddings: bool,
        sink_positions: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        base = self._base_encoder()
        hidden = base.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds) if add_embeddings else inputs_embeds
        extended = base.get_extended_attention_mask(attention_mask, attention_mask.shape, attention_mask.device)
        for layer in base.encoder.layer[start_layer:end_layer]:
            if sink_positions is not None and hasattr(layer.attention.self, "sink_interaction"):
                self_outputs = layer.attention.self(
                    hidden,
                    attention_mask=extended,
                    head_mask=None,
                    output_attentions=False,
                    special_item_pos=sink_positions,
                )
                attention_output = layer.attention.output(self_outputs[0], hidden)
                intermediate_output = layer.intermediate(attention_output)
                hidden = layer.output(intermediate_output, attention_output)
            else:
                hidden = layer(hidden, extended)[0]
        return hidden

    def _encode_texts(
        self,
        texts: List[str],
        max_length: int,
        signal_ids: Optional[List[List[int]]],
        stage: str,
        is_user: bool,
    ) -> torch.Tensor:
        tokenized = self._tokenize(texts, max_length)
        input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
        embeds, sink_positions = self._replace_sink_embeddings(input_ids, signal_ids)

        if self.cfg.use_split_layer:
            mid = self.num_layers // 2
            if self.is_qwen:
                hidden = self._run_qwen_layers(
                    embeds,
                    attention_mask,
                    0,
                    mid,
                    sink_positions if self.cfg.use_inter_sink_attention else None,
                )
            else:
                hidden = self._run_bert_layers(
                    None,
                    embeds,
                    attention_mask,
                    0,
                    mid,
                    add_embeddings=True,
                    sink_positions=sink_positions if self.cfg.use_inter_sink_attention else None,
                )
        else:
            hidden = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state

        if stage == "sink_only" and is_user and self.cfg.use_sink:
            pool_mask = torch.zeros_like(attention_mask)
            for row_idx, positions in enumerate(sink_positions):
                if positions.numel() == 0:
                    pool_mask[row_idx, 0] = 1
                else:
                    pool_mask[row_idx, positions] = 1
            return masked_mean(hidden, pool_mask)
        return masked_mean(hidden, attention_mask)

    def _feature_interaction(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        states = torch.stack(vectors, dim=1)
        if not self.cfg.use_split_layer:
            return states
        attention_mask = torch.ones(states.size()[:2], dtype=torch.long, device=states.device)
        mid = self.num_layers // 2
        if self.is_qwen:
            return self._run_qwen_layers(states, attention_mask, mid, self.num_layers)
        return self._run_bert_layers(None, states, attention_mask, mid, self.num_layers, add_embeddings=False)

    def forward(self, batch: Dict, stage: str = "all_tokens") -> torch.Tensor:
        pos_behaviors, pos_signals = self._prepare_behaviors(batch["positive"], batch["title"])
        neg_behaviors, neg_signals = self._prepare_behaviors(batch["negative"], batch["title"])

        texts = {
            "positive": self._format_user_texts(self.cfg.positive_prompt, pos_behaviors),
            "negative": self._format_user_texts(self.cfg.negative_prompt, neg_behaviors),
            "title": [self.cfg.title_prompt + str(x) for x in batch["title"]],
            "genres": [self.cfg.genres_prompt + str(x) for x in batch["genres"]],
        }
        signal_map = {"positive": pos_signals, "negative": neg_signals, "title": None, "genres": None}
        length_map = {
            "positive": self.cfg.max_user_tokens,
            "negative": self.cfg.max_user_tokens,
            "title": self.cfg.max_item_tokens,
            "genres": self.cfg.max_item_tokens,
        }

        pooled = {}
        for name in FEATURES:
            pooled[name] = self._encode_texts(
                texts[name],
                length_map[name],
                signal_map[name],
                stage,
                is_user=name in {"positive", "negative"},
            )
        interacted = self._feature_interaction([pooled[name] for name in FEATURES])
        projected = [self.projectors[name](interacted[:, idx, :]) for idx, name in enumerate(FEATURES)]
        fused = self.feature_fusion(torch.cat(projected, dim=-1))
        return self.ctr_head(fused)

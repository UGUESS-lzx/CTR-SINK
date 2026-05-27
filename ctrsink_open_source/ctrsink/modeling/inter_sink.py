import math
import types
from typing import List, Optional

import torch
import torch.nn as nn

try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
except Exception:
    apply_rotary_pos_emb = None
    repeat_kv = None

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
        repeat_kv as qwen3_repeat_kv,
    )
except Exception:
    qwen3_apply_rotary_pos_emb = None
    qwen3_repeat_kv = None


class TokenInteractionModel(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x = self.norm(x)
        q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        return self.dropout(torch.softmax(scores, dim=-1)).mean(dim=1)


def _expand_sink_bias(
    hidden_states: torch.Tensor,
    special_item_pos: Optional[List[torch.Tensor]],
    interaction: TokenInteractionModel,
) -> Optional[torch.Tensor]:
    if special_item_pos is None or len(special_item_pos) == 0:
        return None

    batch_size, seq_len, _ = hidden_states.shape
    max_special = max((len(pos) for pos in special_item_pos), default=0)
    if max_special == 0:
        return None

    device = hidden_states.device
    pos_tensor = torch.zeros(batch_size, max_special, dtype=torch.long, device=device)
    valid_mask = torch.zeros(batch_size, max_special, dtype=torch.bool, device=device)
    for batch_idx, pos in enumerate(special_item_pos):
        if len(pos) == 0:
            continue
        pos = torch.as_tensor(pos, dtype=torch.long, device=device)
        pos = pos[(pos >= 0) & (pos < seq_len)]
        if pos.numel() == 0:
            continue
        pos_tensor[batch_idx, : pos.numel()] = pos
        valid_mask[batch_idx, : pos.numel()] = True

    flat_batch = torch.arange(batch_size, device=device).repeat_interleave(max_special)
    flat_pos = pos_tensor.reshape(-1)
    special_feats = hidden_states[flat_batch, flat_pos].view(batch_size, max_special, -1)
    special_feats = special_feats * valid_mask.unsqueeze(-1).to(special_feats.dtype)
    sink_bias = interaction(special_feats)
    sink_bias = sink_bias * valid_mask.unsqueeze(1).to(sink_bias.dtype) * valid_mask.unsqueeze(2).to(sink_bias.dtype)

    expanded = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=sink_bias.dtype)
    for batch_idx, pos in enumerate(special_item_pos):
        if len(pos) == 0:
            continue
        pos = torch.as_tensor(pos, dtype=torch.long, device=device)
        pos = pos[(pos >= 0) & (pos < seq_len)]
        if pos.numel() == 0:
            continue
        block = sink_bias[batch_idx, : pos.numel(), : pos.numel()]
        rows = pos.unsqueeze(1).expand(-1, pos.numel())
        cols = pos.unsqueeze(0).expand(pos.numel(), -1)
        expanded[batch_idx, rows, cols] = block
    return expanded


def _qwen2_attention_forward_with_sink(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    special_item_pos: Optional[List[torch.Tensor]] = None,
):
    if apply_rotary_pos_emb is None or repeat_kv is None:
        raise RuntimeError("Qwen2 attention patch requires transformers Qwen2 internals.")

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    else:
        cos, sin = position_embeddings
        try:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        except TypeError:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    expanded = _expand_sink_bias(hidden_states, special_item_pos, self.sink_interaction)
    if expanded is not None:
        attn_weights = attn_weights + expanded.unsqueeze(1) * self.sink_scale

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def enable_qwen2_inter_sink_attention(model, layer_count: int) -> None:
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Expected a Qwen2Model-like object with .layers.")
    for layer in layers[:layer_count]:
        attn = layer.self_attn
        if hasattr(attn, "sink_interaction"):
            continue
        dropout = getattr(attn, "attention_dropout", 0.0)
        attn.sink_interaction = TokenInteractionModel(attn.hidden_size, attn.num_heads, dropout)
        attn.sink_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        attn.forward = types.MethodType(_qwen2_attention_forward_with_sink, attn)


def _qwen3_attention_forward_with_sink(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    special_item_pos: Optional[List[torch.Tensor]] = None,
    **kwargs,
):
    if qwen3_apply_rotary_pos_emb is None or qwen3_repeat_kv is None:
        raise RuntimeError("Qwen3 attention patch requires transformers Qwen3 internals.")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = qwen3_repeat_kv(key_states, self.num_key_value_groups)
    value_states = qwen3_repeat_kv(value_states, self.num_key_value_groups)

    scale = getattr(self, "scaling", self.head_dim**-0.5)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scale
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    expanded = _expand_sink_bias(hidden_states, special_item_pos, self.sink_interaction)
    if expanded is not None:
        attn_weights = attn_weights + expanded.unsqueeze(1) * self.sink_scale

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights,
        p=0.0 if not self.training else getattr(self, "attention_dropout", 0.0),
        training=self.training,
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def enable_qwen3_inter_sink_attention(model, layer_count: int) -> None:
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Expected a Qwen3Model-like object with .layers.")
    for layer in layers[:layer_count]:
        attn = layer.self_attn
        if hasattr(attn, "sink_interaction"):
            continue
        hidden_size = getattr(attn.config, "hidden_size", getattr(attn, "hidden_size", None))
        num_heads = getattr(attn.config, "num_attention_heads", getattr(attn, "num_heads", None))
        if hidden_size is None or num_heads is None:
            raise ValueError("Could not infer Qwen3 hidden size or attention head count.")
        dropout = getattr(attn, "attention_dropout", 0.0)
        attn.sink_interaction = TokenInteractionModel(hidden_size, num_heads, dropout)
        attn.sink_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        attn.forward = types.MethodType(_qwen3_attention_forward_with_sink, attn)


def _bert_attention_forward_with_sink(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    special_item_pos: Optional[List[torch.Tensor]] = None,
):
    mixed_query_layer = self.query(hidden_states)

    is_cross_attention = encoder_hidden_states is not None
    if is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

    query_layer = self.transpose_for_scores(mixed_query_layer)
    use_cache = getattr(self, "is_decoder", False)
    present_key_value = (key_layer, value_layer) if use_cache else None

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    if not is_cross_attention:
        expanded = _expand_sink_bias(hidden_states, special_item_pos, self.sink_interaction)
        if expanded is not None:
            attention_scores = attention_scores + expanded.unsqueeze(1) * self.sink_scale

    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    if present_key_value is not None:
        outputs = outputs + (present_key_value,)
    return outputs


def enable_bert_inter_sink_attention(model, layer_count: int) -> None:
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError("Expected a BERT/RoBERTa-like model with .encoder.layer.")
    for layer in layers[:layer_count]:
        attn = layer.attention.self
        if hasattr(attn, "sink_interaction"):
            continue
        hidden_size = getattr(attn, "all_head_size", attn.query.out_features)
        num_heads = getattr(attn, "num_attention_heads")
        dropout = getattr(attn.dropout, "p", 0.0)
        attn.sink_interaction = TokenInteractionModel(hidden_size, num_heads, dropout)
        attn.sink_scale = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        attn.forward = types.MethodType(_bert_attention_forward_with_sink, attn)

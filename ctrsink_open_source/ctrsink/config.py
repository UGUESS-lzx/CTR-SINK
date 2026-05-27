import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    train_file: str = ""
    valid_file: str = ""
    output_dir: str = "outputs/run"
    backbone: str = "Qwen/Qwen2-0.5B"
    retriever_model: str = "facebook/contriever"
    tuning_method: str = "lora"
    model_dtype: str = "auto"
    use_split_layer: bool = True
    use_retrieval: bool = False
    use_sink: bool = False
    use_cls_token: bool = False
    use_inter_sink_attention: bool = False
    inter_sink_layers: int = 0
    sink_signal: str = "temporal"
    sink_token: str = "[unused1]"
    sink_num_embeddings: int = 100
    sink_embedding_dim: int = 256
    random_seed: int = 42
    epochs: int = 3
    two_stage: bool = False
    stage1_epochs: int = 3
    stage2_epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    lr: float = 1e-5
    weight_decay: float = 5e-5
    warmup_ratio: float = 0.05
    max_user_tokens: int = 512
    max_item_tokens: int = 32
    num_behaviors: int = 50
    emb_output_size: int = 128
    ctr_hidden_size: int = 32
    num_workers: int = 2
    precision: str = "bf16"
    gradient_checkpointing: bool = False
    user_positive_col: str = "positive_movie_titles"
    user_negative_col: str = "negative_movie_titles"
    item_title_col: str = "title"
    item_genres_col: str = "genres"
    label_col: str = "click_label"
    positive_prompt: str = "用户打高分的电影序列为:"
    negative_prompt: str = "用户打低分的电影序列为:"
    title_prompt: str = "电影名称为:"
    genres_prompt: str = "电影类型为:"
    save_every_epoch: bool = False

    @classmethod
    def from_json(cls, path: Optional[str]) -> "TrainConfig":
        cfg = cls()
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg.update(data)
        return cfg

    def update(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> None:
        if self.two_stage and not self.use_sink:
            raise ValueError("two_stage requires use_sink.")
        if self.use_inter_sink_attention and not self.use_sink:
            raise ValueError("use_inter_sink_attention requires use_sink.")
        if self.use_inter_sink_attention and not self.use_split_layer:
            raise ValueError("use_inter_sink_attention requires use_split_layer.")
        if self.use_cls_token and self.use_inter_sink_attention:
            raise ValueError("use_cls_token is a preliminary ablation; do not mix it with inter-sink attention.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def total_epochs(self) -> int:
        return self.stage1_epochs + self.stage2_epochs if self.two_stage else self.epochs

    def stage_for_epoch(self, epoch_idx: int) -> str:
        if self.two_stage and epoch_idx < self.stage1_epochs:
            return "sink_only"
        return "all_tokens"

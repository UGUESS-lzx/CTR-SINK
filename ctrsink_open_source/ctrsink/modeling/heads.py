import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CtrHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.leaky_relu(self.fc1(x))).squeeze(-1)

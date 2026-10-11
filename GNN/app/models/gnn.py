from __future__ import annotations

from dataclasses import dataclass


def require_torch_geometric():
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import SAGEConv  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "GraphSAGE model requires torch and torch_geometric. "
            "Install dependencies from requirements.txt before training or inference."
        ) from exc
    return torch, nn, SAGEConv


@dataclass(frozen=True)
class GNNDimensions:
    node_features: int
    edge_features: int
    hidden_dim: int = 128
    dropout: float = 0.2


def build_model(dimensions: GNNDimensions):
    torch, nn, SAGEConv = require_torch_geometric()

    class GraphSAGEEdgeClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = SAGEConv(dimensions.node_features, dimensions.hidden_dim)
            self.conv2 = SAGEConv(dimensions.hidden_dim, dimensions.hidden_dim)
            self.dropout = nn.Dropout(dimensions.dropout)
            self.edge_mlp = nn.Sequential(
                nn.Linear(dimensions.hidden_dim * 2 + dimensions.edge_features, dimensions.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dimensions.dropout),
                nn.Linear(dimensions.hidden_dim, 1),
            )

        def encode(self, x, edge_index):
            h = self.conv1(x, edge_index).relu()
            h = self.dropout(h)
            h = self.conv2(h, edge_index).relu()
            return h

        def forward(self, x, edge_index, candidate_edge_index, edge_attr):
            h = self.encode(x, edge_index)
            src = h[candidate_edge_index[0]]
            dst = h[candidate_edge_index[1]]
            pair = torch.cat([src, dst, edge_attr], dim=-1)
            return self.edge_mlp(pair).squeeze(-1)

    return GraphSAGEEdgeClassifier()

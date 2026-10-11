import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None, reason="torch/PyG not installed")
def test_gnn_builds_when_dependencies_exist():
    from app.models.gnn import GNNDimensions, build_model

    model = build_model(GNNDimensions(node_features=10, edge_features=10, hidden_dim=16))
    assert model is not None

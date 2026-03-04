import pytest
import torch

from frisk.models import GCNEncoder


def _toy_graph():
    x = torch.randn(5, 4)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2, 3], [1, 2, 3, 4, 0, 0, 1, 2]],
        dtype=torch.long,
    )
    return x, edge_index


@pytest.mark.parametrize("conv_type", ["gcn", "sage", "gat", "rgcn"])
def test_encoder_supports_multiple_conv_types(conv_type: str):
    x, edge_index = _toy_graph()
    model = GCNEncoder(
        in_dim=4,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        conv_type=conv_type,
        gat_heads=2,
        rgcn_num_relations=8,
    )
    edge_type = None
    if conv_type == "rgcn":
        edge_type = torch.randint(0, 4, (edge_index.size(1),), dtype=torch.long)
    out = model(x, edge_index, edge_type=edge_type)
    assert out.shape == (5, 8)
    assert torch.isfinite(out).all()


def test_encoder_rejects_invalid_conv_type():
    with pytest.raises(ValueError, match="conv_type"):
        GCNEncoder(in_dim=4, hidden_dim=8, conv_type="bad")  # type: ignore[arg-type]

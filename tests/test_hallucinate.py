import torch
import torch.nn as nn

from frisk.hallucinate import HallucinationConfig, hallucinate_negative


class _IdentityModel(nn.Module):
    def forward(self, x, edge_index, edge_weight=None):
        return x


def test_hallucinate_can_freeze_non_return_features():
    torch.manual_seed(0)
    model = _IdentityModel()
    x = torch.tensor(
        [
            [0.10, -0.20, 0.05, 0.01, 10.0, 100.0],
            [0.03, -0.04, 0.02, -0.01, 20.0, 200.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[0.25], [0.25]], dtype=torch.float32)
    batch = torch.tensor([0, 0], dtype=torch.long)

    cfg = HallucinationConfig(
        steps=4,
        lr=0.1,
        l2_weight=0.01,
        mean_weight=0.01,
        std_weight=0.01,
        corr_weight=0.01,
        clamp_std=3.0,
        goodness_temp=1.0,
        node_fraction=1.0,
        node_min=1,
        init_noise=0.0,
        return_slice_len=4,
        penalty_scope="returns",
        corr_scope="returns",
        freeze_non_return_features=True,
    )
    x_neg = hallucinate_negative(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        config=cfg,
    )
    assert torch.allclose(x_neg[:, 4:], x[:, 4:])
    assert not torch.allclose(x_neg[:, :4], x[:, :4])

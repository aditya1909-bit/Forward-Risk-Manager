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


def test_hallucinate_supports_sparse_periodic_corr_penalty():
    torch.manual_seed(0)
    model = _IdentityModel()
    x = torch.randn(6, 5)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4]],
        dtype=torch.long,
    )
    edge_attr = torch.full((edge_index.size(1), 1), 0.1, dtype=torch.float32)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    cfg = HallucinationConfig(
        steps=3,
        lr=0.05,
        l2_weight=0.01,
        mean_weight=0.01,
        std_weight=0.01,
        corr_weight=0.1,
        corr_every_n_steps=2,
        corr_edge_fraction=0.5,
        corr_edge_min=2,
    )
    x_neg = hallucinate_negative(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        config=cfg,
    )
    assert x_neg.shape == x.shape
    assert torch.isfinite(x_neg).all()


def test_hallucinate_supports_adaptive_lr_moment_penalties_and_target_early_stop():
    torch.manual_seed(0)
    model = _IdentityModel()
    x = torch.randn(4, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.full((edge_index.size(1), 1), 0.2, dtype=torch.float32)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    cfg = HallucinationConfig(
        steps=12,
        lr=0.1,
        l2_weight=0.01,
        mean_weight=0.01,
        std_weight=0.01,
        corr_weight=0.01,
        adaptive_lr=True,
        adaptive_lr_patience=1,
        adaptive_lr_decay=0.5,
        adaptive_lr_min=1e-4,
        early_stop_on_target_hit=True,
        target_hit_patience=1,
        moment_mean_weight=0.05,
        moment_var_weight=0.05,
        moment_skew_weight=0.05,
        return_slice_len=4,
        moment_scope="returns",
    )

    x_neg = hallucinate_negative(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        config=cfg,
        constraint_monitor_fn=lambda _: {"hit": True},
    )
    assert x_neg.shape == x.shape
    assert torch.isfinite(x_neg).all()

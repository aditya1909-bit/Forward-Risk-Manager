import torch

from frisk.training.objectives import (
    bootstrap_graph_latent_loss,
    compute_portfolio_head_loss,
    vicreg_graph_loss,
)


def test_compute_portfolio_head_loss_supports_delta_cara():
    head = torch.nn.Linear(4, 1)
    embeddings = torch.randn(6, 4)
    graph_idx = torch.arange(6, dtype=torch.long)
    targets = [0.03, -0.02, 0.01, 0.04, -0.01, 0.02]

    loss = compute_portfolio_head_loss(
        portfolio_head=head,
        embeddings=embeddings,
        graph_idx=graph_idx,
        portfolio_targets=targets,
        device=torch.device("cpu"),
        loss_type="delta_cara",
        cara_risk_aversion=3.0,
        baseline_exposure=1.0,
        delta_scale=0.5,
        max_abs_exposure=1.5,
    )

    assert loss is not None
    assert torch.isfinite(loss)


def test_bootstrap_graph_latent_loss_is_finite_for_matching_shapes():
    torch.manual_seed(0)
    predictor = torch.nn.Linear(5, 5)
    z_online = torch.randn(4, 5)
    z_target = z_online + 0.01 * torch.randn_like(z_online)

    loss = bootstrap_graph_latent_loss(predictor, z_online, z_target)

    assert loss is not None
    assert torch.isfinite(loss)


def test_vicreg_graph_loss_penalizes_collapsed_embeddings_more_than_spread_embeddings():
    collapsed = torch.zeros(8, 6)
    spread = torch.randn(8, 6)

    loss_collapsed = vicreg_graph_loss(
        collapsed,
        variance_weight=1.0,
        covariance_weight=1.0,
        variance_target=1.0,
    )
    loss_spread = vicreg_graph_loss(
        spread,
        variance_weight=1.0,
        covariance_weight=1.0,
        variance_target=1.0,
    )

    assert loss_collapsed is not None
    assert loss_spread is not None
    assert torch.isfinite(loss_collapsed)
    assert torch.isfinite(loss_spread)
    assert float(loss_collapsed) > float(loss_spread)

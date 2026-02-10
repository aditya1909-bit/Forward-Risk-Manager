import torch
import torch.nn.functional as F

from frisk.ff import (
    goodness,
    make_negative,
    pairwise_distance_forward_loss,
    permute_graph_embeddings,
    self_contrastive_loss,
    self_contrastive_retrieval_accuracy,
)


def _goodness_naive(h: torch.Tensor, batch: torch.Tensor, temperature: float) -> torch.Tensor:
    node_energy = (h * h).mean(dim=1)
    out = []
    for gid in batch.unique():
        idx = (batch == gid).nonzero(as_tuple=False).view(-1)
        e = node_energy[idx]
        out.append(temperature * torch.logsumexp(e / temperature, dim=0))
    return torch.stack(out, dim=0)


def test_goodness_matches_naive_unsorted_batch():
    torch.manual_seed(0)
    h = torch.randn(9, 5)
    batch = torch.tensor([2, 0, 2, 1, 0, 1, 1, 2, 0], dtype=torch.long)
    temp = 0.37

    g = goodness(h, batch, temperature=temp)
    g_ref = _goodness_naive(h, batch, temperature=temp)
    assert torch.allclose(g, g_ref, atol=1e-6)


def test_make_negative_shuffle_preserves_graph_membership():
    torch.manual_seed(0)
    x = torch.tensor([[1.0], [2.0], [10.0], [20.0], [100.0]])
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    out = make_negative(x, batch, mode="shuffle")

    vals_g0 = set(out[:2, 0].tolist())
    vals_g1 = set(out[2:, 0].tolist())
    assert vals_g0 == {1.0, 2.0}
    assert vals_g1 == {10.0, 20.0, 100.0}


def test_make_negative_time_flip_keeps_summary_slice():
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 9.0, 99.0],
            [5.0, 6.0, 7.0, 8.0, 8.0, 88.0],
        ]
    )
    batch = torch.tensor([0, 1], dtype=torch.long)
    out = make_negative(
        x,
        batch,
        mode="time_flip",
        window_len=4,
        summary_dim=2,
    )

    assert torch.equal(out[:, :4], torch.tensor([[4.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0]]))
    assert torch.equal(out[:, 4:], x[:, 4:])


def test_permute_graph_embeddings_deranges_rows_for_n_gt_1():
    torch.manual_seed(0)
    z = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out = permute_graph_embeddings(z)

    assert out.shape == z.shape
    assert not torch.any(torch.all(out == z, dim=1))


def test_self_contrastive_loss_reports_pos_above_neg_for_similar_views():
    torch.manual_seed(0)
    z_a = F.normalize(torch.randn(6, 8), dim=1)
    z_b = F.normalize(z_a + 0.01 * torch.randn_like(z_a), dim=1)

    loss, pos_sim, neg_sim = self_contrastive_loss(z_a, z_b, temperature=0.2)

    assert float(loss) >= 0.0
    assert float(pos_sim) > float(neg_sim)


def test_self_contrastive_retrieval_accuracy_prefers_matching_pairs():
    torch.manual_seed(0)
    z_a = F.normalize(torch.randn(8, 12), dim=1)
    z_b_close = F.normalize(z_a + 0.02 * torch.randn_like(z_a), dim=1)
    z_b_perm = z_b_close[torch.randperm(z_b_close.size(0))]

    acc_close = self_contrastive_retrieval_accuracy(z_a, z_b_close)
    acc_perm = self_contrastive_retrieval_accuracy(z_a, z_b_perm)

    assert float(acc_close) > float(acc_perm)
    assert 0.0 <= float(acc_close) <= 1.0


def test_pairwise_distance_forward_loss_prefers_farther_negatives():
    z_pos = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    z_neg_close = z_pos + 0.01
    z_neg_far = z_pos + 5.0

    loss_close = pairwise_distance_forward_loss(z_pos, z_neg_close, margin=0.1)
    loss_far = pairwise_distance_forward_loss(z_pos, z_neg_far, margin=0.1)

    assert float(loss_close) > float(loss_far)

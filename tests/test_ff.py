import torch
import torch.nn.functional as F

from frisk.ff import (
    ff_loss,
    goodness,
    make_negative,
    pairwise_distance_forward_loss,
    permute_graph_embeddings,
    rank_spread_loss,
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


def test_goodness_mean_reducer_matches_manual_graph_means():
    h = torch.tensor(
        [
            [1.0, 3.0],
            [2.0, 2.0],
            [4.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    g = goodness(h, batch, reducer="mean")
    expected = torch.tensor([4.5, 5.0], dtype=torch.float32)
    assert torch.allclose(g, expected, atol=1e-6)


def test_goodness_layernorm_changes_energy_scale_but_keeps_shape():
    torch.manual_seed(0)
    h = torch.randn(6, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    g_plain = goodness(h, batch, temperature=0.5)
    g_ln = goodness(h, batch, temperature=0.5, norm="layernorm")
    assert g_plain.shape == g_ln.shape == (2,)
    assert not torch.allclose(g_plain, g_ln)


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


def test_make_negative_block_bootstrap_keeps_tail_slice():
    torch.manual_seed(0)
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 9.0, 99.0],
            [10.0, 20.0, 30.0, 40.0, 8.0, 88.0],
        ]
    )
    batch = torch.tensor([0, 1], dtype=torch.long)
    out = make_negative(
        x,
        batch,
        mode="block_bootstrap",
        window_len=4,
        summary_dim=2,
    )

    assert out.shape == x.shape
    assert torch.equal(out[:, 4:], x[:, 4:])
    for i in range(x.size(0)):
        src = set(x[i, :4].tolist())
        for v in out[i, :4].tolist():
            assert v in src


def test_make_negative_cross_asset_mix_stays_within_group_range():
    torch.manual_seed(0)
    x = torch.tensor([[1.0], [2.0], [10.0], [20.0], [100.0]])
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    out = make_negative(x, batch, mode="cross_asset_mix")

    for gid in batch.unique():
        idx = (batch == gid).nonzero(as_tuple=False).view(-1)
        lo = float(x[idx].min().item())
        hi = float(x[idx].max().item())
        assert torch.all(out[idx] >= lo - 1e-6)
        assert torch.all(out[idx] <= hi + 1e-6)


def test_make_negative_phase_randomize_keeps_tail_and_moments():
    torch.manual_seed(0)
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 9.0, 99.0],
            [2.0, 1.0, 0.0, -1.0, 8.0, 88.0],
        ]
    )
    batch = torch.tensor([0, 1], dtype=torch.long)
    out = make_negative(
        x,
        batch,
        mode="phase_randomize",
        window_len=4,
        summary_dim=2,
    )

    assert out.shape == x.shape
    assert torch.equal(out[:, 4:], x[:, 4:])
    assert torch.allclose(out[:, :4].mean(dim=1), x[:, :4].mean(dim=1), atol=1e-4)
    assert torch.allclose(out[:, :4].std(dim=1), x[:, :4].std(dim=1), atol=1e-4)


def test_make_negative_sector_swap_respects_graph_and_sector_groups():
    x = torch.tensor(
        [
            [1.0, 10.0, 0.0],
            [2.0, 10.0, 0.0],
            [3.0, 20.0, 0.0],
            [4.0, 20.0, 0.0],
            [101.0, 10.0, 0.0],
            [102.0, 10.0, 0.0],
            [103.0, 20.0, 0.0],
            [104.0, 20.0, 0.0],
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    out = make_negative(
        x,
        batch,
        mode="sector_swap",
        sector_idx=1,
    )
    for gid in [0, 1]:
        for sec in [10.0, 20.0]:
            idx = ((batch == gid) & (x[:, 1] == sec)).nonzero(as_tuple=False).view(-1)
            src_vals = set(x[idx, 0].tolist())
            out_vals = set(out[idx, 0].tolist())
            assert out_vals.issubset(src_vals)


def test_make_negative_factor_hard_is_seed_deterministic_and_no_cross_graph():
    x = torch.tensor(
        [
            [1.0, 2.0, 0.1, 0.2],
            [2.0, 3.0, 0.2, 0.1],
            [3.0, 4.0, 0.3, 0.4],
            [101.0, 102.0, 0.1, 0.2],
            [102.0, 103.0, 0.2, 0.1],
            [103.0, 104.0, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    torch.manual_seed(7)
    out_a = make_negative(
        x,
        batch,
        mode="factor_hard",
        window_len=2,
        factor_start_idx=2,
        factor_dim=2,
        hard_topk=2,
    )
    torch.manual_seed(7)
    out_b = make_negative(
        x,
        batch,
        mode="factor_hard",
        window_len=2,
        factor_start_idx=2,
        factor_dim=2,
        hard_topk=2,
    )
    assert torch.equal(out_a, out_b)

    for gid in [0, 1]:
        idx = (batch == gid).nonzero(as_tuple=False).view(-1)
        src_rows = {tuple(row.tolist()) for row in x[idx, :2]}
        out_rows = {tuple(row.tolist()) for row in out_a[idx, :2]}
        assert out_rows.issubset(src_rows)


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


def test_ff_loss_margin_penalizes_small_positive_negative_gap():
    g_pos = torch.tensor([1.1, 1.0, 1.2], dtype=torch.float32)
    g_neg = torch.tensor([1.0, 0.95, 1.05], dtype=torch.float32)
    base = ff_loss(g_pos, g_neg, target=1.0, margin=0.0, margin_weight=1.0)
    with_margin = ff_loss(g_pos, g_neg, target=1.0, margin=0.3, margin_weight=1.0)
    assert float(with_margin) > float(base)


def test_ff_loss_symba_prefers_larger_positive_negative_gap():
    g_pos = torch.tensor([2.0, 2.2, 2.1], dtype=torch.float32)
    g_neg_close = torch.tensor([1.9, 2.0, 1.8], dtype=torch.float32)
    g_neg_far = torch.tensor([0.2, 0.1, 0.0], dtype=torch.float32)
    loss_close = ff_loss(g_pos, g_neg_close, loss_type="symba")
    loss_far = ff_loss(g_pos, g_neg_far, loss_type="symba")
    assert float(loss_close) > float(loss_far)


def test_rank_spread_loss_smaller_when_top_bottom_gap_is_larger():
    scores_small_gap = torch.tensor([0.5, 0.52, 0.49, 0.51, 0.5], dtype=torch.float32)
    scores_big_gap = torch.tensor([0.9, 0.85, 0.5, 0.2, 0.1], dtype=torch.float32)
    loss_small = rank_spread_loss(scores_small_gap, top_frac=0.4, margin=0.1)
    loss_big = rank_spread_loss(scores_big_gap, top_frac=0.4, margin=0.1)
    assert float(loss_small) > float(loss_big)

import torch

from frisk.ff import goodness, make_negative


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

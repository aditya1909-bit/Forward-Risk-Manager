import numpy as np

from frisk.eval_metrics import ff_binary_metrics


def test_ff_binary_metrics_perfect_separation():
    g_pos = np.array([2.0, 2.5, 3.0], dtype=float)
    g_neg = np.array([-1.0, -0.5, 0.0], dtype=float)
    out = ff_binary_metrics(g_pos, g_neg, threshold=0.5, ece_bins=10)

    assert out["eval_auroc"] > 0.99
    assert out["eval_auprc"] > 0.99
    assert out["eval_brier"] < 0.1
    assert 0.0 <= out["eval_ece"] <= 1.0

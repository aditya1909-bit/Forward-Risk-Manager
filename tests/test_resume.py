from __future__ import annotations

import random
from pathlib import Path
import warnings

import torch

from frisk.resume import (
    capture_rng_state,
    load_module_state_for_resume,
    load_resume_payload,
    module_state_for_resume,
    move_optimizer_state_to_device,
    restore_rng_state,
    resume_fingerprint,
    save_resume_payload,
)


def test_resume_payload_roundtrip_restores_module_and_optimizer(tmp_path: Path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(3, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    saved_params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fingerprint = resume_fingerprint({"job": "unit-test", "epochs": 3})
    ckpt_path = tmp_path / "resume.pt"
    save_resume_payload(
        ckpt_path,
        fingerprint=fingerprint,
        status="in_progress",
        epoch_completed=2,
        model_states={"model": module_state_for_resume(model)},
        optimizer_state=optimizer.state_dict(),
        metadata={"marker": 7},
        epoch_history=[{"epoch": 1, "train_loss": 0.5}],
        rng_state=capture_rng_state(),
    )

    model_restored = torch.nn.Linear(4, 2)
    optimizer_restored = torch.optim.Adam(model_restored.parameters(), lr=1e-3)
    payload = load_resume_payload(ckpt_path, expected_fingerprint=fingerprint)

    assert payload is not None
    load_module_state_for_resume(model_restored, payload["model_states"]["model"])
    optimizer_restored.load_state_dict(payload["optimizer_state"])
    move_optimizer_state_to_device(optimizer_restored, torch.device("cpu"))

    for key, tensor in model_restored.state_dict().items():
        assert torch.equal(tensor, saved_params[key])
    assert payload["metadata"]["marker"] == 7
    assert payload["epoch_completed"] == 2
    assert payload["epoch_history"][0]["epoch"] == 1


def test_resume_payload_respects_fingerprint(tmp_path: Path):
    ckpt_path = tmp_path / "resume.pt"
    save_resume_payload(
        ckpt_path,
        fingerprint=resume_fingerprint({"job": "expected"}),
        status="completed",
        epoch_completed=4,
        metadata={"ok": True},
    )

    assert load_resume_payload(ckpt_path, expected_fingerprint=resume_fingerprint({"job": "other"})) is None


def test_resume_payload_ignores_corrupt_checkpoint(tmp_path: Path):
    ckpt_path = tmp_path / "resume.pt"
    ckpt_path.write_bytes(b"not a torch checkpoint")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = load_resume_payload(ckpt_path)

    assert payload is None
    assert any("Failed to load resume checkpoint" in str(w.message) for w in caught)


def test_rng_state_capture_and_restore():
    random.seed(11)
    torch.manual_seed(11)

    saved_state = capture_rng_state()
    expected_random = random.random()
    expected_torch = torch.rand(3)

    restore_rng_state(saved_state)

    assert random.random() == expected_random
    assert torch.equal(torch.rand(3), expected_torch)

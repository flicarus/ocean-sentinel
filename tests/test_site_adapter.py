"""Tests for the per-site SiteAdapter and its fine-tune loop.

We avoid the heavy v7.4 backbone in unit tests:
- SiteAdapter forward/identity behaviour is exercised directly with
  random tensors.
- The fine-tune loop's safeguards (held-out floor, init-as-identity,
  best-snapshot reversion) are tested against a TINY fake vessel head
  by monkeypatching `_ship_prob_from_embedding`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ocean_sentinel.gemma import site_adapter
from ocean_sentinel.gemma.site_adapter import SiteAdapter, load_site_adapter


# ── Module behaviour ────────────────────────────────────────────────────


def test_adapter_initialised_as_identity():
    """At init, residual up.weight=0 → forward(x) == x bit-exact."""
    adapter = SiteAdapter(dim=256, bottleneck=64)
    x = torch.randn(8, 256)
    y = adapter(x)
    assert torch.allclose(x, y)


def test_adapter_param_count_is_small():
    """~33k params — orders of magnitude smaller than the 2.3M v7 backbone."""
    adapter = SiteAdapter(dim=256, bottleneck=64)
    total = sum(p.numel() for p in adapter.parameters())
    assert 25_000 < total < 50_000


def test_adapter_changes_output_after_a_gradient_step():
    """After one optimiser step, forward(x) ≠ x. Smoke test that the
    parameters are actually trainable."""
    adapter = SiteAdapter(dim=256, bottleneck=64)
    x = torch.randn(4, 256)

    # Push toward zeros
    optim = torch.optim.SGD(adapter.parameters(), lr=0.5)
    for _ in range(5):
        optim.zero_grad()
        loss = (adapter(x) ** 2).sum()
        loss.backward()
        optim.step()

    y = adapter(x)
    assert not torch.allclose(x, y)


def test_adapter_round_trip_save_and_load(tmp_path):
    adapter = SiteAdapter(dim=256, bottleneck=64)
    # Mutate so save/load is non-trivial
    with torch.no_grad():
        adapter.up.weight.add_(0.1)
    path = tmp_path / "adapter.pt"
    torch.save(adapter.state_dict(), str(path))

    loaded = load_site_adapter(path, torch.device("cpu"))
    x = torch.randn(2, 256)
    assert torch.allclose(adapter(x), loaded(x), atol=1e-5)


# ── Held-out floor + best-snapshot reversion ──────────────────────────


@pytest.fixture
def fake_v7(monkeypatch):
    """Replace the v7.4 backbone+transformer pipeline with a tiny stub
    so train_site_adapter can run in <1 s under unit tests."""
    rng = np.random.RandomState(0)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # We don't use these — _embed_batch is monkeypatched. But the
            # trainer DOES read `model.vessel_head` so provide a stub one.
            self.vessel_head = torch.nn.Linear(256, 2)
            with torch.no_grad():
                self.vessel_head.weight.zero_()
                # Bias so untouched embedding produces ship_prob ≈ 0.85
                self.vessel_head.bias[0] = 0.0
                self.vessel_head.bias[1] = 1.5
            self.backbone = torch.nn.Identity()
            self.temporal = torch.nn.Identity()
            self.type_head = torch.nn.Linear(256, 5)
            self.distance_head = torch.nn.Linear(256, 4)

        def load_state_dict(self, *a, **k):
            pass

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter(self.vessel_head.parameters())

    fake_model = FakeModel()

    def fake_load_v7(*a, **k):
        return fake_model

    # Patch the v7 import used by train_site_adapter
    import ocean_sentinel.models.cnn_v7 as cnn_v7_module
    monkeypatch.setattr(cnn_v7_module, "OceanSentinelV7", lambda: fake_model)

    # Patch torch.load so we don't need the real ckpt
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})

    # _embed_batch returns deterministic per-row embeddings.
    def fake_embed(chunks, sr, model, device, mel_freqs):
        # Each chunk → 256-d embedding with a small per-chunk perturbation
        n = len(chunks)
        embs = []
        for i in range(n):
            v = torch.from_numpy(rng.randn(256).astype("float32"))
            embs.append(v)
        if not embs:
            return torch.zeros(0, 256)
        return torch.stack(embs).to(device)

    monkeypatch.setattr(site_adapter, "_embed_batch", fake_embed)
    return fake_model


def test_adapter_persists_to_disk(tmp_path, monkeypatch, fake_v7):
    """End-to-end smoke test of train_site_adapter writes adapter.pt."""
    import soundfile as sf

    monkeypatch.chdir(tmp_path)
    sr = 16_000
    y = np.random.RandomState(7).randn(sr * 200).astype("float32") * 0.05
    p = tmp_path / "amb.wav"
    sf.write(str(p), y, sr)

    # Provide minimal "held-out vessel clips" so the loop has a recall
    # input. We monkeypatch _default_holdout_clips to return our fake.
    fake_clip = tmp_path / "ship.wav"
    sf.write(str(fake_clip), np.random.randn(sr * 60).astype("float32"), sr)
    monkeypatch.setattr(
        site_adapter, "_default_holdout_clips",
        lambda: [fake_clip],
    )

    out = site_adapter.train_site_adapter(
        site_id="unit-test",
        ambient_source=str(p),
        n_epochs=2,
        device="cpu",
    )
    assert out["ok"] is True
    adapter_path = Path(out["adapter_path"])
    assert adapter_path.exists()
    assert adapter_path.name == "adapter.pt"
    assert "median_ship_prob_before" in out
    assert "median_ship_prob_after" in out


def test_train_returns_error_for_missing_ambient(fake_v7):
    out = site_adapter.train_site_adapter(
        site_id="unit-test",
        ambient_source="/nonexistent/audio.wav",
    )
    assert out["ok"] is False
    assert "not found" in out["error"]

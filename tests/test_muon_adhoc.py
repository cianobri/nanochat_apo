import os
import subprocess
import sys

os.environ["NANOCHAT_DISABLE_COMPILE"] = "1"

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW


def test_gpt_setup_optimizer_accepts_muon_adhoc():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        window_pattern="L",
    )
    model = GPT(config)

    optimizer = model.setup_optimizer(
        optimizer="adamw_muon",
        muon_orthogonalization="muon_adhoc",
        ns_steps=6,
    )

    muon_groups = [group for group in optimizer.param_groups if group["kind"] == "muon"]
    assert muon_groups
    assert all(group["orthogonalization"] == "muon_adhoc" for group in muon_groups)


def test_gpt_setup_optimizer_rejects_unknown_muon_method():
    model = GPT(GPTConfig(sequence_len=16, vocab_size=64, n_layer=1, n_head=2, n_kv_head=2, n_embd=16))

    with pytest.raises(ValueError, match="Unknown Muon orthogonalization method"):
        model.setup_optimizer(optimizer="adamw_muon", muon_orthogonalization="not_a_method")


def test_muon_adhoc_step_updates_tall_and_wide_params():
    tall = torch.nn.Parameter(torch.randn(4, 2))
    wide = torch.nn.Parameter(torch.randn(2, 4))
    tall_before = tall.detach().clone()
    wide_before = wide.detach().clone()
    tall.grad = torch.randn_like(tall)
    wide.grad = torch.randn_like(wide)

    optimizer = MuonAdamW([
        dict(
            kind="muon", params=[tall], lr=0.01, momentum=0.9, ns_steps=5,
            orthogonalization="muon_adhoc", muon_norm_iters=True,
            orthogonalization_dtype="float32", ortho_order=1, ortho_grid=17,
            ortho_newton=2, beta2=0.9, weight_decay=0.0,
        ),
        dict(
            kind="muon", params=[wide], lr=0.01, momentum=0.9, ns_steps=5,
            orthogonalization="muon_adhoc", muon_norm_iters=True,
            orthogonalization_dtype="float32", ortho_order=1, ortho_grid=17,
            ortho_newton=2, beta2=0.9, weight_decay=0.0,
        ),
    ])

    optimizer.step()

    assert torch.isfinite(tall).all()
    assert torch.isfinite(wide).all()
    assert not torch.equal(tall, tall_before)
    assert not torch.equal(wide, wide_before)


def test_base_train_help_lists_muon_adhoc():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.base_train", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "NANOCHAT_DISABLE_COMPILE": "1", "PYTHONIOENCODING": "utf-8"},
    )

    assert "muon_adhoc" in result.stdout

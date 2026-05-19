import os
import subprocess
import sys

os.environ["NANOCHAT_DISABLE_COMPILE"] = "1"

import torch

import nanochat.optim as optim
from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW


def test_uagq_routes_to_uagq_function():
    assert optim._get_muon_step_fn({"orthogonalization": "uagq"}) is optim.muon_step_uagq_fused


def test_uagq_error_step_runs_for_stacked_batch():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 2) * 0.2

    actual = optim._uagq_error_step(x, ortho_grid=17, ortho_newton=2)

    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()


def test_uagq_step_updates_tall_and_wide_params():
    torch.manual_seed(1)
    tall = torch.nn.Parameter(torch.randn(4, 2))
    wide = torch.nn.Parameter(torch.randn(2, 4))
    tall_before = tall.detach().clone()
    wide_before = wide.detach().clone()
    tall.grad = torch.randn_like(tall)
    wide.grad = torch.randn_like(wide)

    optimizer = MuonAdamW([
        dict(
            kind="muon", params=[tall], lr=0.01, momentum=0.9, ns_steps=2,
            orthogonalization="uagq", muon_norm_iters=True,
            orthogonalization_dtype="float32", ortho_order=1, ortho_grid=17,
            ortho_newton=2, beta2=0.9, weight_decay=0.0,
        ),
        dict(
            kind="muon", params=[wide], lr=0.01, momentum=0.9, ns_steps=2,
            orthogonalization="uagq", muon_norm_iters=True,
            orthogonalization_dtype="float32", ortho_order=1, ortho_grid=17,
            ortho_newton=2, beta2=0.9, weight_decay=0.0,
        ),
    ])

    optimizer.step()

    assert torch.isfinite(tall).all()
    assert torch.isfinite(wide).all()
    assert not torch.equal(tall, tall_before)
    assert not torch.equal(wide, wide_before)


def test_gpt_setup_optimizer_accepts_uagq():
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
        muon_orthogonalization="uagq",
        ns_steps=2,
    )

    muon_groups = [group for group in optimizer.param_groups if group["kind"] == "muon"]
    assert muon_groups
    assert all(group["orthogonalization"] == "uagq" for group in muon_groups)


def test_base_train_help_lists_uagq():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.base_train", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "NANOCHAT_DISABLE_COMPILE": "1", "PYTHONIOENCODING": "utf-8"},
    )

    assert "uagq" in result.stdout

import os

os.environ["NANOCHAT_DISABLE_COMPILE"] = "1"

import pytest
import torch

import nanochat.optim as optim
from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW


def test_ls2_routes_to_ls2_function():
    assert optim._get_muon_step_fn({"orthogonalization": "ls2"}) is optim.muon_step_ls2_fused


def test_ls2_error_step_runs_for_stacked_batch():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 2) * 0.2

    actual = optim._ls2_error_step(x)

    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("shape", [(5, 2), (2, 5)])
def test_ls2_step_updates_params_and_preserves_shape(shape):
    torch.manual_seed(1)
    params = [torch.nn.Parameter(torch.randn(*shape)) for _ in range(2)]
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.randn_like(p)

    optimizer = MuonAdamW([
        dict(
            kind="muon", params=params, lr=0.01, momentum=0.9, ns_steps=1,
            orthogonalization="ls2", muon_norm_iters=True,
            orthogonalization_dtype="float32", ortho_order=1, ortho_grid=17,
            ortho_newton=2, beta2=0.9, weight_decay=0.0,
        ),
    ])

    optimizer.step()

    for p, p_before in zip(params, before):
        assert p.shape == p_before.shape
        assert torch.isfinite(p).all()
        assert not torch.equal(p, p_before)


def test_gpt_setup_optimizer_accepts_ls2():
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
        muon_orthogonalization="ls2",
        ns_steps=1,
    )

    muon_groups = [group for group in optimizer.param_groups if group["kind"] == "muon"]
    assert muon_groups
    assert all(group["orthogonalization"] == "ls2" for group in muon_groups)

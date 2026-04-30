import os

os.environ["NANOCHAT_DISABLE_COMPILE"] = "1"

import pytest
import torch

import nanochat.optim as optim
from nanochat.optim import MuonAdamW


@pytest.mark.parametrize("ortho_order", [1, 2])
def test_adaptive_poly_step_updates_tall_and_wide_params(ortho_order):
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
            orthogonalization="adaptive_poly", muon_norm_iters=True,
            orthogonalization_dtype="float32", ortho_order=ortho_order, ortho_grid=17,
            ortho_newton=2, beta2=0.9, weight_decay=0.0,
        ),
        dict(
            kind="muon", params=[wide], lr=0.01, momentum=0.9, ns_steps=2,
            orthogonalization="adaptive_poly", muon_norm_iters=True,
            orthogonalization_dtype="float32", ortho_order=ortho_order, ortho_grid=17,
            ortho_newton=2, beta2=0.9, weight_decay=0.0,
        ),
    ])

    optimizer.step()

    assert torch.isfinite(tall).all()
    assert torch.isfinite(wide).all()
    assert not torch.equal(tall, tall_before)
    assert not torch.equal(wide, wide_before)


def test_error_moments_index_zero_is_batch_ones():
    e = torch.randn(3, 2, 2)
    moments = optim._error_moments(e, 2)

    assert torch.equal(moments[0], torch.ones(3, dtype=e.dtype, device=e.device))


def test_adaptive_poly_first_order_beta_near_identity_is_near_half():
    x = torch.diag(torch.tensor([0.995, 0.99, 0.985])).unsqueeze(0)
    e = torch.eye(3).unsqueeze(0) - x.mT @ x
    moments = optim._error_moments(e, 6)

    beta = optim._optimal_beta_first_order_from_error_moments(moments, 0.0, 1.0, 33, 4)

    assert torch.allclose(beta, torch.full_like(beta, 0.5), atol=0.03)


def test_gso_fixed_gamma_beta_is_finite_for_batch():
    x = torch.randn(2, 5, 3) * 0.2
    g = x.mT @ x
    eye = torch.eye(g.size(-1), dtype=g.dtype, device=g.device)
    e = eye - g
    moments = optim._error_moments(e, 10)
    gamma = torch.full(x.shape[:-2], 0.375, dtype=x.dtype, device=x.device)

    beta = optim._optimal_beta_with_fixed_gamma_from_error_moments(moments, gamma, 0.0, 2.0, 17, 2)

    assert beta.shape == gamma.shape
    assert torch.isfinite(beta).all()


def test_adaptive_poly_second_order_gamma_fixed_matches_quintic_newton_schulz():
    x = torch.randn(2, 5, 3) * 0.2
    g = x.mT @ x
    eye = torch.eye(g.size(-1), dtype=g.dtype, device=g.device)
    e = eye - g
    e2 = e @ e
    g2 = g @ g

    error_basis = x @ (eye + 0.5 * e + 0.375 * e2)
    gram_basis = x @ (1.875 * eye - 1.25 * g + 0.375 * g2)

    assert torch.allclose(error_basis, gram_basis, atol=1e-6, rtol=1e-6)


def test_gso_step_uses_beta_for_e_and_gamma_for_e_squared(monkeypatch):
    x = torch.randn(2, 5, 3) * 0.2
    gamma_prev = torch.full((2,), 0.375)
    beta_value = torch.full((2,), 0.25)
    gamma_value = torch.full((2,), 0.125)

    def fake_beta(moments, gamma, beta_min, beta_max, ortho_grid, ortho_newton):
        assert torch.equal(gamma, gamma_prev.to(dtype=gamma.dtype, device=gamma.device))
        return beta_value.to(dtype=moments[0].dtype, device=moments[0].device)

    def fake_gamma(moments, beta, gamma_min, gamma_max, ortho_grid, ortho_newton):
        assert torch.equal(beta, beta_value.to(dtype=beta.dtype, device=beta.device))
        assert gamma_min == 0.0
        assert gamma_max == 1.5
        return gamma_value.to(dtype=moments[0].dtype, device=moments[0].device)

    monkeypatch.setattr(optim, "_optimal_beta_with_fixed_gamma_from_error_moments", fake_beta)
    monkeypatch.setattr(optim, "_optimal_gamma_with_fixed_beta_from_error_moments", fake_gamma)

    actual, gamma = optim._gso_error_step(x, gamma_prev, 17, 2)

    g = x.mT @ x
    eye = torch.eye(g.size(-1), dtype=g.dtype, device=g.device)
    e = eye - g
    e2 = e @ e
    expected = x @ (eye + beta_value[..., None, None] * e + gamma_value[..., None, None] * e2)

    assert torch.allclose(actual, expected)
    assert torch.equal(gamma, gamma_value.to(dtype=gamma.dtype, device=gamma.device))


@pytest.mark.parametrize(("orthogonalization", "step_name"), [
    ("adaptive_poly", "_adaptive_poly_first_order_error_step"),
    ("ls2", "_ls2_error_step"),
    ("gso", "_gso_error_step"),
])
@pytest.mark.parametrize("muon_norm_iters", [False, True])
def test_muon_norm_iters_reaches_adaptive_methods(monkeypatch, orthogonalization, step_name, muon_norm_iters):
    captured_norms = []
    original_step = getattr(optim, step_name)

    def wrapped_step(x, *args):
        captured_norms.append(x.norm(dim=(-2, -1)).detach().clone())
        return original_step(x, *args)

    monkeypatch.setattr(optim, step_name, wrapped_step)

    p = torch.nn.Parameter(torch.zeros(4, 2))
    p.grad = torch.full_like(p, 3.0)
    optimizer = MuonAdamW([
        dict(
            kind="muon", params=[p], lr=0.01, momentum=0.0, ns_steps=1,
            orthogonalization=orthogonalization, muon_norm_iters=muon_norm_iters,
            orthogonalization_dtype="float32", ortho_order=1, ortho_grid=17,
            ortho_newton=2, beta2=0.9, weight_decay=0.0,
        ),
    ])

    optimizer.step()

    assert len(captured_norms) == 1
    expected_norm = torch.ones_like(captured_norms[0]) if muon_norm_iters else p.grad.norm(dim=(-2, -1))
    assert torch.allclose(captured_norms[0], expected_norm, atol=1e-5, rtol=1e-5)

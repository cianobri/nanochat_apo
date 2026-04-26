"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.

Adapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import os

import torch
import torch.distributed as dist
from torch import Tensor

from nanochat.common import COMPUTE_DTYPE


# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""


def adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update.

    This function may be wrapped with torch.compile below unless
    NANOCHAT_DISABLE_COMPILE=1 is set.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)

    # Update running averages
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)

    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t

    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _finish_muon_step(
    g: Tensor,
    stacked_params: Tensor,
    second_momentum_buffer: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    red_dim: int,
) -> None:
    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)

    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()

    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)

    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()

    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


def muon_step_polar_express_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
    muon_norm_iters: bool,
    ortho_order: int,
    ortho_grid: int,
    ortho_newton: int,
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update.

    This function may be wrapped with torch.compile below unless
    NANOCHAT_DISABLE_COMPILE=1 is set.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar Express
    # Cast to bf16 for speed when available; skip cast otherwise.
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    if muon_norm_iters:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)

    if g.size(-2) > g.size(-1):  # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    _finish_muon_step(X, stacked_params, second_momentum_buffer, lr_t, wd_t, beta2_t, red_dim)


def muon_step_newton_schulz_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
    muon_norm_iters: bool,
    ortho_order: int,
    ortho_grid: int,
    ortho_newton: int,
) -> None:
    """
    Fused Muon step: momentum -> Newton-Schulz -> variance_reduction -> cautious_update.

    This function may be wrapped with torch.compile below unless
    NANOCHAT_DISABLE_COMPILE=1 is set.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Newton-Schulz iteration for the zeroth power / orthogonalized update.
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    if muon_norm_iters:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-6)

    if g.size(-2) > g.size(-1):  # Tall matrix
        for _ in range(ns_steps):
            A = X.mT @ X
            if ortho_order == 1:
                X = 1.5 * X - 0.5 * (X @ A)
            elif ortho_order == 2:
                A2 = A @ A
                X = X @ (1.875 * torch.eye(A.size(-1), dtype=A.dtype, device=A.device) - 1.25 * A + 0.375 * A2)
            else:
                raise ValueError(f"ortho_order must be 1 or 2, got {ortho_order}")
    else:  # Wide matrix
        for _ in range(ns_steps):
            A = X @ X.mT
            if ortho_order == 1:
                X = 1.5 * X - 0.5 * (A @ X)
            elif ortho_order == 2:
                A2 = A @ A
                X = (1.875 * torch.eye(A.size(-1), dtype=A.dtype, device=A.device) - 1.25 * A + 0.375 * A2) @ X
            else:
                raise ValueError(f"ortho_order must be 1 or 2, got {ortho_order}")

    _finish_muon_step(X, stacked_params, second_momentum_buffer, lr_t, wd_t, beta2_t, red_dim)


def _trace_mean(x: Tensor) -> Tensor:
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / x.size(-1)


def _error_moments(e: Tensor, max_k: int) -> list[Tensor]:
    moments: list[Tensor] = [torch.empty(0, device=e.device, dtype=e.dtype)]
    p = e
    moments.append(_trace_mean(p))
    for _ in range(2, max_k + 1):
        p = p @ e
        moments.append(_trace_mean(p))
    return moments


def _eval_quartic(c0: Tensor, c1: Tensor, c2: Tensor, c3: Tensor, c4: Tensor, beta: Tensor) -> Tensor:
    return (((c4 * beta + c3) * beta + c2) * beta + c1) * beta + c0


def _quartic_minimizer_on_interval(
    c0: Tensor,
    c1: Tensor,
    c2: Tensor,
    c3: Tensor,
    c4: Tensor,
    lo: float,
    hi: float,
    n_grid: int = 17,
    n_newton: int = 2,
) -> Tensor:
    """
    Approximate minimizer of

        J(beta) = c0 + c1 beta + c2 beta^2 + c3 beta^3 + c4 beta^4

    over [lo, hi]. Uses a fixed grid to select a local bracket, followed by
    safeguarded Newton steps on J'(beta).
    """

    beta_grid = torch.linspace(lo, hi, n_grid, dtype=c0.dtype, device=c0.device)
    beta_grid = beta_grid.reshape(*((1,) * c0.ndim), n_grid)

    values = _eval_quartic(
        c0.unsqueeze(-1),
        c1.unsqueeze(-1),
        c2.unsqueeze(-1),
        c3.unsqueeze(-1),
        c4.unsqueeze(-1),
        beta_grid,
    )
    idx = values.argmin(dim=-1)

    grid = beta_grid.expand(*c0.shape, n_grid)
    delta = (hi - lo) / float(n_grid - 1)
    beta0 = grid.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    left = (beta0 - delta).clamp_min(lo)
    right = (beta0 + delta).clamp_max(hi)
    beta = beta0.clamp(left, right)

    eps = 32.0 * torch.finfo(beta.dtype).eps
    for _ in range(n_newton):
        jp = c1 + 2.0 * c2 * beta + 3.0 * c3 * beta.square() + 4.0 * c4 * beta.pow(3)
        jpp = 2.0 * c2 + 6.0 * c3 * beta + 12.0 * c4 * beta.square()
        safe = jpp.abs() > eps
        denom = torch.where(safe, jpp, torch.ones_like(jpp))
        beta_new = beta - jp / denom
        beta_new = beta_new.clamp(left, right)
        beta = torch.where(safe, beta_new, beta)

    candidates = torch.stack([left, beta, right], dim=-1)
    candidate_values = _eval_quartic(
        c0.unsqueeze(-1),
        c1.unsqueeze(-1),
        c2.unsqueeze(-1),
        c3.unsqueeze(-1),
        c4.unsqueeze(-1),
        candidates,
    )
    best = candidate_values.argmin(dim=-1)
    return candidates.gather(-1, best.unsqueeze(-1)).squeeze(-1)


def _adaptive_poly_first_order_step(x: Tensor, eye: Tensor, ortho_grid: int, ortho_newton: int) -> Tensor:
    g = x.mT @ x
    e = eye - g
    m = _error_moments(e, 6)

    c0 = m[2]
    c1 = -4.0 * m[2] + 4.0 * m[3]
    c2 = 4.0 * m[2] - 10.0 * m[3] + 6.0 * m[4]
    c3 = 4.0 * m[3] - 8.0 * m[4] + 4.0 * m[5]
    c4 = m[4] - 2.0 * m[5] + m[6]

    beta = _quartic_minimizer_on_interval(c0, c1, c2, c3, c4, 0.0, 1.0, ortho_grid, ortho_newton)
    correction = eye + beta[..., None, None] * e
    return x @ correction


def _adaptive_poly_second_order_step(x: Tensor, eye: Tensor, ortho_grid: int, ortho_newton: int) -> Tensor:
    g = x.mT @ x
    e = eye - g
    e2 = e @ e
    m = _error_moments(e, 10)

    c0 = (9.0 / 16.0) * m[4] + (3.0 / 8.0) * m[5] + (1.0 / 16.0) * m[6]
    c1 = -3.0 * m[4] + 0.5 * m[5] + 2.0 * m[6] + 0.5 * m[7]
    c2 = 4.0 * m[4] - 4.0 * m[5] - 4.5 * m[6] + 3.0 * m[7] + 1.5 * m[8]
    c3 = 4.0 * m[6] - 6.0 * m[7] + 2.0 * m[9]
    c4 = m[8] - 2.0 * m[9] + m[10]

    beta = _quartic_minimizer_on_interval(c0, c1, c2, c3, c4, 0.0, 0.8, ortho_grid, ortho_newton)
    correction = eye + 0.5 * e + beta[..., None, None] * e2
    return x @ correction


def muon_step_adaptive_poly_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
    muon_norm_iters: bool,
    ortho_order: int,
    ortho_grid: int,
    ortho_newton: int,
) -> None:
    """
    Fused Muon step: momentum -> adaptive polynomial orthogonalization
    -> variance_reduction -> cautious_update.
    """

    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g.float()
    if muon_norm_iters:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-6)

    transposed = g.size(-2) <= g.size(-1)
    if transposed:
        X = X.mT

    eye = torch.eye(X.size(-1), dtype=X.dtype, device=X.device).expand(*X.shape[:-2], X.size(-1), X.size(-1))
    for _ in range(ns_steps):
        if ortho_order == 1:
            X = _adaptive_poly_first_order_step(X, eye, ortho_grid, ortho_newton)
        elif ortho_order == 2:
            X = _adaptive_poly_second_order_step(X, eye, ortho_grid, ortho_newton)
        else:
            raise ValueError(f"ortho_order must be 1 or 2, got {ortho_order}")

    if transposed:
        X = X.mT

    _finish_muon_step(X, stacked_params, second_momentum_buffer, lr_t, wd_t, beta2_t, red_dim)


def _get_muon_step_fn(group: dict):
    orthogonalization = group.get("orthogonalization", "polar_express")
    if orthogonalization == "polar_express":
        return muon_step_polar_express_fused
    if orthogonalization == "newton_schulz":
        return muon_step_newton_schulz_fused
    if orthogonalization == "adaptive_poly":
        return muon_step_adaptive_poly_fused
    raise ValueError(f"Unknown Muon orthogonalization method: {orthogonalization}")


# -----------------------------------------------------------------------------
# Optional torch.compile wrapping.
#
# Local Windows CPU tests often fail because torch.compile/Inductor needs cl.exe.
# Set NANOCHAT_DISABLE_COMPILE=1 locally.
#
# On CUDA machines, leave it unset for the fast compiled path.

if os.environ.get("NANOCHAT_DISABLE_COMPILE", "0") != "1":
    adamw_step_fused = torch.compile(adamw_step_fused, dynamic=False, fullgraph=True)
    muon_step_polar_express_fused = torch.compile(muon_step_polar_express_fused, dynamic=False, fullgraph=True)
    muon_step_newton_schulz_fused = torch.compile(muon_step_newton_schulz_fused, dynamic=False, fullgraph=True)
    muon_step_adaptive_poly_fused = torch.compile(muon_step_adaptive_poly_fused, dynamic=False, fullgraph=True)


# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'orthogonalization',
              'muon_norm_iters', 'ortho_order', 'ortho_grid', 'ortho_newton',
              'beta2', 'weight_decay'
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})

        # 0-D CPU tensors avoid torch.compile recompilation when values change.
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            adamw_step_fused(
                p,
                grad,
                exp_avg,
                exp_avg_sq,
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group, stacked for efficiency.
        """
        params: list[Tensor] = group["params"]
        if not params:
            return

        p0 = params[0]
        state = self.state[p0]
        num_params = len(params)
        shape, device, dtype = p0.shape, p0.device, p0.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]

        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_fn = _get_muon_step_fn(group)
        muon_step_fn(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
            bool(group.get("muon_norm_iters", True)),
            group.get("ortho_order", 1),
            group.get("ortho_grid", 17),
            group.get("ortho_newton", 2),
        )

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")


# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})

        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group."""
        param_infos = {}

        for p in group["params"]:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0, (
                    f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) "
                    f"divisible by world_size ({world_size})"
                )
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)

        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group."""
        params = group["params"]
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size

        p0 = params[0]
        shape, device, dtype = p0.shape, p0.device, p0.dtype

        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[: len(params)].copy_(grad_stack)

        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()

        return dict(
            future=future,
            grad_chunk=grad_chunk,
            stacked_grads=stacked_grads,
            chunk_size=chunk_size,
        )

    def _compute_adamw(
        self,
        group: dict,
        info: dict,
        gather_list: list,
        rank: int,
        world_size: int,
    ) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info["param_infos"]

        for p in group["params"]:
            pinfo = param_infos[p]
            pinfo["future"].wait()
            grad_slice = pinfo["grad_slice"]
            state = self.state[p]

            if pinfo["is_small"]:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)

            state["step"] += 1

            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            adamw_step_fused(
                p_slice,
                grad_slice,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

            if not pinfo["is_small"]:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        info["future"].wait()

        params = group["params"]
        chunk_size = info["chunk_size"]
        grad_chunk = info["grad_chunk"]

        p0 = params[0]
        shape, device, dtype = p0.shape, p0.device, p0.dtype

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        state = self.state[p0]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)

        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)

        red_dim = -1 if shape[-2] >= shape[-1] else -2
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group["weight_decay"])

            muon_step_fn = _get_muon_step_fn(group)
            muon_step_fn(
                grad_chunk[:num_owned],
                stacked_owned,
                state["momentum_buffer"][:num_owned],
                state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                group["ns_steps"],
                red_dim,
                bool(group.get("muon_norm_iters", True)),
                group.get("ortho_order", 1),
                group.get("ortho_grid", 17),
                group.get("ortho_newton", 2),
            )

            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                torch._foreach_copy_(
                    info["params"],
                    list(info["stacked_params"][: len(info["params"])].unbind(0)),
                )

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_infos: list[dict] = []

        for group in self.param_groups:
            if group["kind"] == "adamw":
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group["kind"] == "muon":
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        gather_list: list[dict] = []

        for group, info in zip(self.param_groups, reduce_infos):
            if group["kind"] == "adamw":
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group["kind"] == "muon":
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        self._finish_gathers(gather_list)

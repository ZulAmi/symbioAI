"""
Influence Function Approximation for Efficient Causal Buffer Selection.

Implements Koh & Liang (2017) influence functions as a fast alternative to
the checkpoint/restore TRUE causality method. Target overhead: ~3x vs vanilla
instead of ~10x for TRUE causality.

Theory
------
The influence of training point z on test loss L(z_test) is:

    I(z, z_test) = -∇_θ L(z_test)ᵀ H⁻¹ ∇_θ L(z)

where H = ∇²_θ L_train is the Hessian of the training loss.

Interpretation for continual learning:
  - Negative influence → sample reduces old-task loss → beneficial (replay it)
  - Positive influence → sample increases old-task loss → harmful (skip it)

H⁻¹v is approximated via the LiSSA algorithm (stochastic conjugate gradient),
which requires only Hessian-vector products (no explicit H storage).

Reference
---------
Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via
influence functions. ICML.

LiSSA: Agarwal et al. (2017). Second-order stochastic optimization for
machine learning in linear time. JMLR.

Usage
-----
Register as use_causal_sampling=4 in DerppCausal to ablate against TRUE
causality (mode 3). Typical speedup: 3–5x over mode 3.

Example::

    approx = InfluenceFunctionApproximator(model, n_cg_iterations=10)
    indices = approx.rank_buffer_samples(
        buffer_samples=[(x_i, y_i) for ...],
        old_task_data={0: (X0, y0), 1: (X1, y1)},
        top_k=128,
    )
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfluenceFunctionApproximator:
    """
    Approximates TRUE causal effects using influence functions.

    Runs in O(n_cg_iterations * |params|) per candidate instead of 2 full
    forward/backward passes — enabling ~3x speedup over checkpoint/restore.

    Args:
        model: The continual learning backbone (e.g. ResNet-18).
        damping: Tikhonov regularisation for H stability (λ in H + λI).
        n_cg_iterations: LiSSA recursion depth — more = more accurate, slower.
        scale: LiSSA scaling factor (controls numerical stability).
        max_params: If model has more params than this, restrict influence to
            the last `max_params` parameters (classifier head only) for speed.
    """

    def __init__(
        self,
        model: nn.Module,
        damping: float = 0.01,
        n_cg_iterations: int = 10,
        scale: float = 25.0,
        max_params: int = 100_000,
    ) -> None:
        self.model = model
        self.damping = damping
        self.n_cg_iterations = n_cg_iterations
        self.scale = scale
        self.max_params = max_params

        # Only differentiate through a subset of params if model is large
        self._param_names = self._select_params()

    def _select_params(self) -> list[str]:
        """Select parameters to differentiate through (last layers if too large)."""
        all_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        total = sum(p.numel() for _, p in all_params)
        if total <= self.max_params:
            return [n for n, _ in all_params]

        # Use only the final linear layer (classifier head) for efficiency
        head_params = [(n, p) for n, p in all_params if "classifier" in n or "linear" in n]
        if head_params:
            logger.debug(
                "Model has %d parameters; restricting influence to classifier head (%d params)",
                total, sum(p.numel() for _, p in head_params),
            )
            return [n for n, _ in head_params]
        return [n for n, _ in all_params[-2:]]  # last 2 layers as fallback

    def _get_params(self) -> list[torch.Tensor]:
        """Return the selected differentiable parameters as a flat list."""
        return [p for n, p in self.model.named_parameters() if n in self._param_names]

    def _hvp(
        self,
        loss: torch.Tensor,
        params: list[torch.Tensor],
        v: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Hessian-vector product Hv via two backward passes.

        Uses the identity: Hv = ∇(∇L · v)
        """
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

        grad_v = sum((g * vi).sum() for g, vi in zip(grads, v))
        hvp = torch.autograd.grad(grad_v, params, retain_graph=False, allow_unused=True)
        return [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp, params)]

    def compute_ihvp(
        self,
        test_loss: torch.Tensor,
        train_data_iter: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[torch.Tensor]:
        """
        Compute H⁻¹v via LiSSA (stochastic CG) where v = ∇ L(z_test).

        Args:
            test_loss: Loss on the test/old-task data (already computed).
            train_data_iter: Mini-batches from training data for Hessian estimate.

        Returns:
            List of tensors matching parameter shapes.
        """
        params = self._get_params()

        # v = ∇ L(z_test)
        v = torch.autograd.grad(test_loss, params, retain_graph=False, allow_unused=True)
        v = [g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(v, params)]

        # LiSSA: H⁻¹v ≈ (1/scale) Σ_t (I - H/scale)^t v
        estimate = [vi.clone() for vi in v]
        current = [vi.clone() for vi in v]

        for t, (bx, by) in enumerate(train_data_iter):
            if t >= self.n_cg_iterations:
                break
            bx = bx.to(next(self.model.parameters()).device)
            by = by.to(bx.device)

            self.model.zero_grad()
            out = self.model(bx) if not hasattr(self.model, "net") else self.model.net(bx)
            loss_t = F.cross_entropy(out, by)

            hvp = self._hvp(loss_t, params, current)

            # Recursion: current = v - (H/scale)·current + damping·current
            current = [
                vi - h / self.scale + self.damping * ci
                for vi, h, ci in zip(v, hvp, current)
            ]
            estimate = [e + c for e, c in zip(estimate, current)]

        ihvp = [e / self.scale for e in estimate]
        return [i.detach() for i in ihvp]

    def estimate_influence(
        self,
        candidate: tuple[torch.Tensor, torch.Tensor],
        old_task_data: dict[int, tuple[torch.Tensor, torch.Tensor]],
        train_data_iter: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> float:
        """
        Estimate causal influence of `candidate` on old-task loss.

        I(z_candidate, z_test) = -∇L(z_test)ᵀ H⁻¹ ∇L(z_candidate)

        Negative → beneficial (reduces forgetting).
        Positive → harmful (increases forgetting).

        Args:
            candidate: (x, y) buffer sample to evaluate.
            old_task_data: {task_id: (X, y)} for forgetting measurement.
            train_data_iter: Optional mini-batches for Hessian estimate.
                Defaults to using the candidate itself (cheap approximation).

        Returns:
            Scalar influence score.
        """
        params = self._get_params()
        device = next(self.model.parameters()).device

        # Build combined old-task loss
        all_x, all_y = [], []
        for task_x, task_y in old_task_data.values():
            n = min(20, len(task_x))
            all_x.append(task_x[:n].to(device).float())
            all_y.append(task_y[:n].to(device).long())
        if not all_x:
            return 0.0

        batch_x = torch.cat(all_x)
        batch_y = torch.cat(all_y)

        self.model.zero_grad()
        with torch.enable_grad():
            out = self.model(batch_x) if not hasattr(self.model, "net") else self.model.net(batch_x)
            test_loss = F.cross_entropy(out, batch_y)

        # Default train_data_iter: just use the candidate sample itself
        if train_data_iter is None:
            cx, cy = candidate
            train_data_iter = [(cx.unsqueeze(0).float(), cy.unsqueeze(0).long())]

        ihvp = self.compute_ihvp(test_loss, train_data_iter)

        # ∇L(z_candidate)
        cx, cy = candidate
        cx = cx.unsqueeze(0).to(device).float()
        cy = cy.unsqueeze(0).to(device).long()

        self.model.zero_grad()
        with torch.enable_grad():
            out_c = self.model(cx) if not hasattr(self.model, "net") else self.model.net(cx)
            cand_loss = F.cross_entropy(out_c, cy)

        cand_grads = torch.autograd.grad(cand_loss, params, allow_unused=True)
        cand_grads = [
            g.detach() if g is not None else torch.zeros_like(p)
            for g, p in zip(cand_grads, params)
        ]

        # I = -∇L(test)ᵀ H⁻¹ ∇L(candidate)
        influence = -sum((i * g).sum() for i, g in zip(ihvp, cand_grads))
        return float(influence)

    def rank_buffer_samples(
        self,
        buffer_samples: list[tuple[torch.Tensor, torch.Tensor]],
        old_task_data: dict[int, tuple[torch.Tensor, torch.Tensor]],
        top_k: int = 128,
    ) -> list[int]:
        """
        Rank buffer samples by influence on old-task loss; return top-k indices.

        Most negative influence = most beneficial (reduces forgetting most).

        Args:
            buffer_samples: [(x_i, y_i)] candidate list.
            old_task_data: {task_id: (X, y)} for forgetting measurement.
            top_k: Number of samples to select.

        Returns:
            List of indices into buffer_samples, sorted best-first.
        """
        scores: list[tuple[int, float]] = []

        for idx, sample in enumerate(buffer_samples):
            try:
                score = self.estimate_influence(sample, old_task_data)
            except Exception as e:
                logger.warning("Influence estimation failed for sample %d: %s", idx, e)
                score = 0.0
            scores.append((idx, score))

        # Sort by ascending influence (most negative = most beneficial first)
        scores.sort(key=lambda x: x[1])
        return [idx for idx, _ in scores[:top_k]]

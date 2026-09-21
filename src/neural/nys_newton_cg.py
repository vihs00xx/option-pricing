"""Nyström-preconditioned Newton-CG refinement for PINNs.

Implements the NysNewton-CG optimiser of Rathore et al. (2024,
arXiv:2402.01868) as a full-batch second-order refinement stage:

1. compute the gradient and a random Nystrom sketch of the Hessian,
2. build a positive-definite preconditioner from the sketch,
3. solve the damped Newton system with preconditioned conjugate
   gradients using Hessian-vector products,
4. take a backtracking line-search step.

The paper forms the Gauss-Newton approximation with forward-mode
products. Torch 2.8 cannot run forward-mode ``jvp`` through the
second-order autograd graph a Black--Scholes PINN requires, so this
implementation uses the exact Hessian via double backward instead.
A small damping keeps the system positive definite when residuals are
not yet small. Hessian accuracy and loss decrease are unit-tested.
"""

import torch


def flatten(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten(vector, templates):
    result = []
    offset = 0
    for template in templates:
        count = template.numel()
        result.append(vector[offset:offset + count].view_as(template))
        offset += count
    return result


class NysNewtonCG:
    """Damped Newton-CG with a Nystrom sketch preconditioner."""

    def __init__(self, params, loss_fn, sketch_rank=32, cg_max_steps=30,
                 cg_tolerance=1e-6, damping=1e-6, seed=0, max_line_search=10):
        self.params = [p for p in params if p.requires_grad]
        self.loss_fn = loss_fn
        self.sketch_rank = min(sketch_rank, sum(p.numel() for p in self.params))
        self.cg_max_steps = cg_max_steps
        self.cg_tolerance = cg_tolerance
        self.damping = damping
        self.seed = seed
        self.max_line_search = max_line_search
        self.step_count = 0

    def _hessian_vector(self, grads, vector):
        dot = sum(torch.sum(g * v) for g, v in zip(grads, vector))
        return torch.autograd.grad(dot, self.params, retain_graph=True)

    def _apply_preconditioner(self, sketch_orthonormal, vector):
        return sketch_orthonormal @ (sketch_orthonormal.T @ vector)

    def step(self):
        """One optimiser step; returns diagnostics about the step."""
        loss = self.loss_fn()
        if not torch.isfinite(loss):
            return {"loss_before": float(loss.detach()), "accepted": False,
                    "reason": "non_finite_loss"}
        grads = torch.autograd.grad(
            loss, self.params, create_graph=True, retain_graph=True
        )
        gradient = flatten(grads).detach()
        size = gradient.numel()
        generator = torch.Generator().manual_seed(self.seed + self.step_count)
        omega = torch.randn(size, self.sketch_rank, generator=generator,
                            dtype=gradient.dtype)
        sketch_columns = []
        for column in range(self.sketch_rank):
            direction = unflatten(omega[:, column], self.params)
            product = self._hessian_vector(grads, direction)
            sketch_columns.append(flatten([p.detach() for p in product]))
        sketch = torch.stack(sketch_columns, dim=1)
        gram = omega.T @ sketch
        gram = 0.5 * (gram + gram.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        floor = max(float(eigenvalues.max()) * 1e-10, 1e-14)
        keep = eigenvalues > floor
        damping = max(self.damping, 1e-14)
        if int(torch.sum(keep)) > 0:
            # Basis of the sketched top subspace: H ~ B B^T with
            # B = S V D^{-1/2}. With B = Q diag(sigma) R^T, the
            # preconditioner (H_sketch + damping I)^{-1} acts on a
            # vector as Q diag(1 / (sigma^2 + damping)) Q^T v plus the
            # complement term (v - Q Q^T v) / damping, which matches the
            # damping of the Newton system outside the sketch span and
            # keeps the preconditioner positive definite.
            basis = sketch @ (
                eigenvectors[:, keep] / torch.sqrt(eigenvalues[keep])
            )
            left, sigma, _ = torch.linalg.svd(basis, full_matrices=False)
            sigma_squared = sigma**2
            preconditioner = lambda vector: self._apply_preconditioner(
                left, sigma_squared, damping, vector
            )
        else:
            preconditioner = lambda vector: vector / damping
        direction = self._conjugate_gradient(grads, gradient, preconditioner)
        if direction is None:
            return {"loss_before": float(loss.detach()), "accepted": False,
                    "reason": "cg_failed"}
        alpha = self._line_search(loss, direction)
        if alpha is None:
            return {"loss_before": float(loss.detach()), "accepted": False,
                    "reason": "line_search_failed"}
        with torch.no_grad():
            offset = 0
            for parameter in self.params:
                count = parameter.numel()
                parameter.add_(
                    direction[offset:offset + count].view_as(parameter),
                    alpha=alpha,
                )
                offset += count
        self.step_count += 1
        new_loss = float(self.loss_fn().detach())
        return {
            "loss_before": float(loss.detach()), "loss_after": new_loss,
            "accepted": True, "step_size": alpha,
            "sketch_rank": self.sketch_rank,
        }

    def _apply_preconditioner(self, left, sigma_squared, damping, vector):
        """(H_sketch + damping I)^{-1} v via the sketch eigendecomposition."""
        projection = left.T @ vector
        in_span = left @ (projection / (sigma_squared + damping))
        complement = (vector - left @ projection) / damping
        return in_span + complement

    def _conjugate_gradient(self, grads, gradient, preconditioner):
        damping = self.damping

        def system_product(vector):
            direction = unflatten(vector, self.params)
            product = self._hessian_vector(grads, direction)
            flat = flatten([p.detach() for p in product])
            return flat + damping * vector

        x = torch.zeros_like(gradient)
        residual = -gradient
        z = preconditioner(residual)
        p = z
        rz = torch.dot(residual, z)
        norm_g = torch.norm(gradient)
        for _ in range(self.cg_max_steps):
            if torch.norm(residual) <= self.cg_tolerance * (norm_g + 1e-30):
                break
            ap = system_product(p)
            denominator = torch.dot(p, ap)
            if float(denominator) <= 0.0:
                break
            alpha = rz / denominator
            x = x + alpha * p
            residual = residual - alpha * ap
            z = preconditioner(residual)
            rz_new = torch.dot(residual, z)
            if not torch.isfinite(rz_new) or float(rz_new) <= 0.0:
                break
            p = z + (rz_new / rz) * p
            rz = rz_new
        if float(torch.norm(x)) > 0.0 and torch.all(torch.isfinite(x)):
            return x.detach()
        # CG broke down (rank-deficient sketched Hessian): fall back to
        # the damped steepest-descent direction; the line search still
        # enforces a decrease.
        return (-gradient).detach()

    def _line_search(self, loss, direction):
        reference = float(loss.detach())
        alpha = 1.0
        for _ in range(self.max_line_search):
            with torch.no_grad():
                offset = 0
                for parameter in self.params:
                    count = parameter.numel()
                    parameter.add_(
                        direction[offset:offset + count].view_as(parameter),
                        alpha=alpha,
                    )
                    offset += count
            # The loss may require internal autograd (PDE residuals with
            # second derivatives), so it must be evaluated with grad mode
            # enabled and detached afterwards.
            candidate = float(self.loss_fn().detach())
            with torch.no_grad():
                offset = 0
                for parameter in self.params:
                    count = parameter.numel()
                    parameter.add_(
                        direction[offset:offset + count].view_as(parameter),
                        alpha=-alpha,
                    )
                    offset += count
            if candidate < reference:
                return alpha
            alpha *= 0.5
        return None

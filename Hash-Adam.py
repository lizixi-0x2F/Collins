"""
Hash-Adam (Count-Sketch Optimizer)
==================================
A memory-efficient Adam variant that compresses optimizer states (m, v)
into compact hash tables via 2-Universal Hashing + Count-Sketch.

Optimizer state memory: O(K) per parameter instead of O(N).
The tiny hash tables can fit in GPU L1/L2 cache, smashing the memory wall.

Theory: First moment (m) uses Count-Sketch with sign hashing s(i) ∈ {-1, +1}
        for unbiased estimation with Chernoff-bound guarantees.
        Second moment (v) uses direct hash accumulation (no sign hash) —
        collisions safely overestimate v, acting as implicit regularization.
"""

import math
import torch
from torch.optim import Optimizer


class HashAdam(Optimizer):
    """
    Adam with Count-Sketch compressed momentum states.

    Args:
        params: model parameters
        lr: learning rate (default: 1e-3)
        betas: coefficients for running averages (default: (0.9, 0.999))
        eps: term for numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay (AdamW-style) (default: 0.0)
        compress_ratio: compression ratio for hash tables (default: 32)
            K = max(numel // compress_ratio, min_K)
            Chernoff bound optimal: ~34x for eps=0.5, M=0.1, beta1=0.9
        min_K: minimum hash table size (default: 1024)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, compress_ratio=32, min_K=1024):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.compress_ratio = compress_ratio
        self.min_K = min_K
        # Mersenne prime for 2-universal hashing
        self._P = 2147483647  # 2^31 - 1
        super().__init__(params, defaults)

    def _init_state(self, p):
        """Initialize per-parameter hash state."""
        state = self.state[p]
        N = p.numel()
        K = max(N // self.compress_ratio, self.min_K)

        state['step'] = 0
        state['K'] = K
        # Compressed momentum buffers — the memory savings core
        state['hash_m'] = torch.zeros(K, dtype=torch.float32, device=p.device)
        state['hash_v'] = torch.zeros(K, dtype=torch.float32, device=p.device)

        # 2-Universal hash coefficients (random, fixed per param)
        # h(i) = ((a*i + b) mod P) mod K  — position hash
        # s(i) = 2*((c*i + d) mod P mod 2) - 1  — sign hash {-1, +1}
        state['a'] = torch.randint(1, self._P, (1,)).item()
        state['b'] = torch.randint(0, self._P, (1,)).item()
        state['c'] = torch.randint(1, self._P, (1,)).item()
        state['d'] = torch.randint(0, self._P, (1,)).item()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("HashAdam does not support sparse gradients")

                # Lazy state init
                if len(self.state[p]) == 0:
                    self._init_state(p)

                state = self.state[p]
                state['step'] += 1
                step = state['step']
                K = state['K']
                hash_m = state['hash_m']
                hash_v = state['hash_v']
                a, b = state['a'], state['b']
                c, d = state['c'], state['d']
                P = self._P

                # Flatten gradient
                grad_flat = grad.reshape(-1)
                N = grad_flat.numel()

                # --- Decoupled weight decay (AdamW style) ---
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # --- Compute hash indices (fully parallel on GPU) ---
                indices = torch.arange(N, device=p.device, dtype=torch.long)

                # Position hash: h(i) → [0, K)
                h_idx = ((a * indices + b) % P) % K

                # Sign hash: s(i) → {-1, +1}
                s_val = (((c * indices + d) % P) % 2).to(grad_flat.dtype) * 2.0 - 1.0

                # --- Count-Sketch: aggregate gradient into hash buckets ---
                # For m (1st moment): use sign hash for unbiased estimation
                #   sketch[h(i)] += s(i) * g_i
                signed_grad = grad_flat * s_val
                sketched_m = torch.zeros(K, dtype=torch.float32, device=p.device)
                sketched_m.scatter_add_(0, h_idx, signed_grad.float())

                # For v (2nd moment): direct accumulation WITHOUT sign hash
                #   sketch[h(i)] += g_i^2
                # Rationale: g_i^2 >= 0 always. Using sign hash would allow
                # reconstructed v to go negative/near-zero, making the Adam
                # denominator vanish (÷ eps ≈ 1e-8) → catastrophic gradient
                # explosion. Direct accumulation means collisions OVERESTIMATE v
                # (safe: larger denominator → smaller, more conservative updates).
                grad_sq = (grad_flat * grad_flat).float()
                sketched_v = torch.zeros(K, dtype=torch.float32, device=p.device)
                sketched_v.scatter_add_(0, h_idx, grad_sq)

                # --- EMA update in compressed space (O(K) ops, cache-friendly) ---
                hash_m.mul_(beta1).add_(sketched_m, alpha=1.0 - beta1)
                hash_v.mul_(beta2).add_(sketched_v, alpha=1.0 - beta2)

                # --- Reconstruct approximate moments via gather ---
                # m: undo sign hash for unbiased first-moment estimate
                approx_m = hash_m[h_idx] * s_val
                # v: direct lookup (always non-negative, may overestimate)
                approx_v = hash_v[h_idx]

                # --- Bias correction ---
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step

                # --- Adam update ---
                step_size = lr / bc1
                denom = (approx_v.sqrt() / math.sqrt(bc2)).add_(eps)

                # Update in-place
                p.data.reshape(-1).addcdiv_(approx_m, denom, value=-step_size)

        return loss

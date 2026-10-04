"""
Faster arithmetic for the NOVA recurrence  s_t = f_t * s_{t-1} + i_t * u_t  (same maths, same weights).

The core computes the gradient of this recurrence with two Python loops over the sequence (254 passes
per layer for 127 tokens). On a GPU every pass costs a handful of kernel launches, so the loops, not the
arithmetic, set the training speed. Here both directions are done by doubling: log2(T) vectorised steps.

    states = FastScan.apply(f, i, u, initial_state)      # drop-in for blocks_scan.ScanRefBackward.apply

Nothing uses this yet: evo.engine.speed_bench measures it against the current core first.
"""

from __future__ import annotations

import torch


def scan_linear(a: torch.Tensor, b: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """All prefixes of s_t = a_t * s_{t-1} + b_t along dim 1 (zero start).

    Returns (A, S): A_t = a_0 * ... * a_t and S_t = s_t. With reverse=True the recurrence runs from the
    end: s_t = a_t * s_{t+1} + b_t."""
    if reverse:
        a, b = a.flip(1), b.flip(1)
    k, n = 1, a.shape[1]
    while k < n:
        # position t absorbs the finished block that ends at t-k
        b = torch.cat([b[:, :k], a[:, k:] * b[:, :-k] + b[:, k:]], dim=1)
        a = torch.cat([a[:, :k], a[:, k:] * a[:, :-k]], dim=1)
        k *= 2
    if reverse:
        a, b = a.flip(1), b.flip(1)
    return a, b


class FastScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, i, u, initial_state):
        prod, states = scan_linear(f, i * u)
        if initial_state is not None:
            states = prod * initial_state.unsqueeze(1) + states
        ctx.save_for_backward(f, i, u, initial_state, states)
        return states

    @staticmethod
    def backward(ctx, grad_states):
        f, i, u, initial_state, states = ctx.saved_tensors
        # d_t = g_t + f_{t+1} * d_{t+1}: the same kind of recurrence, backwards in time
        f_next = torch.cat([f[:, 1:], torch.zeros_like(f[:, :1])], dim=1)
        _, d = scan_linear(f_next, grad_states, reverse=True)
        first = torch.zeros_like(states[:, :1]) if initial_state is None else initial_state.unsqueeze(1)
        prev = torch.cat([first, states[:, :-1]], dim=1)
        grad_initial = None if initial_state is None else d[:, 0] * f[:, 0]
        return d * prev, d * u, d * i, grad_initial

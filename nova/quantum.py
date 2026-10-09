"""
NOVA-Q: a quantum-inspired memory for the generation-8 core (mixer "Q"). A separate line from the champion: its
cores are released as NOVAQ-..., never as NOVA8-....

What it is (simulated on an ordinary card - not a quantum computer):

  state     every head keeps a complex unit vector psi of `dim` numbers (default 4 heads x 256)
  writing   every token turns the state: psi_t = U(x_t) psi_{t-1}, with U a unitary that is diagonal in a learned
            basis - every component gets a phase e^(i theta), theta = (base frequency + what the token adds).
            A unitary keeps the length of psi, so nothing fades by itself: there is no built-in forgetting.
  reading   like a measurement: the token chooses its own basis (a phase mask phi(x_t)), then the probabilities
            of `dim` outcomes are |M e^(i phi) psi_t|^2 (Born rule, normalised to sum to one); the state is not
            collapsed (we read the expectation, not one random outcome).

Because the turns of one head commute, the whole text is read at once: the phase after t tokens is the running
sum of the angles, Phi_t = Phi_{t-1} + theta_t. The carried state is that phase (heads x dim real numbers, the
same size after any length of text); psi = psi_0 e^(i Phi).

Honest limits: a state that keeps its length does not fade, but what was written can still become unreachable -
later turns mix it with everything else, and the reading sees only `dim` probabilities. "Does not shrink" is not
"remembers perfectly"; the recall test (nova.recall) measures how much survives at a distance.

Building a Q core from a trained one (`transplant`): the token table, the norms and the per-token layers (the
gated MLP of every block) are copied; the memory blocks named in the new pattern as "Q" are new, their output
starts at zero, so at step 0 the new core is the old one without those memories.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

TWO_PI = 2.0 * math.pi


class QMixer(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dim: int = 256, max_turn: float = math.pi) -> None:
        super().__init__()
        self.heads, self.dim, self.max_turn = heads, dim, max_turn
        n = heads * dim
        self.turn = nn.Linear(d_model, n)                 # theta(x): what a token turns, per component
        self.basis = nn.Linear(d_model, n, bias=False)    # phi(x): the basis a token reads in
        self.freq = nn.Parameter(torch.empty(heads, dim))  # base frequencies: the state turns a little with every token
        self.psi0 = nn.Parameter(torch.randn(heads, dim, 2) / math.sqrt(2 * dim))  # starting state (re, im)
        self.meas = nn.Parameter(torch.randn(heads, dim, dim, 2) / math.sqrt(2 * dim))  # measurement M (re, im)
        self.out = nn.Linear(n, d_model, bias=False)
        with torch.no_grad():
            # time scales from 1 to ~10 000 tokens, as in rotary positions
            self.freq.copy_(torch.logspace(0, -4, dim).expand(heads, dim) * torch.rand(heads, dim).add(0.5))
            nn.init.normal_(self.turn.weight, std=0.02)
            nn.init.zeros_(self.turn.bias)
            nn.init.normal_(self.basis.weight, std=0.02)

    def angles(self, u: torch.Tensor) -> torch.Tensor:
        b, t, _ = u.shape
        return self.freq.float() + self.max_turn * torch.tanh(self.turn(u).float()).view(b, t, self.heads, self.dim)

    def start(self) -> torch.Tensor:
        p = self.psi0.float()
        return p / p.pow(2).sum(dim=(-1, -2), keepdim=True).sqrt()           # unit length per head

    def read(self, phase: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """phase [b, t, heads, dim] -> outcome probabilities [b, t, heads * dim]."""
        b, t = phase.shape[:2]
        p0 = self.start()
        mask = phase + self.basis(u).float().view(b, t, self.heads, self.dim)
        c, s = torch.cos(mask), torch.sin(mask)
        re = c * p0[..., 0] - s * p0[..., 1]
        im = s * p0[..., 0] + c * p0[..., 1]
        mr, mi = self.meas[..., 0].float(), self.meas[..., 1].float()                 # [heads, out, in]
        a_re = torch.einsum("bthi,hoi->btho", re, mr) - torch.einsum("bthi,hoi->btho", im, mi)
        a_im = torch.einsum("bthi,hoi->btho", re, mi) + torch.einsum("bthi,hoi->btho", im, mr)
        prob = a_re.pow(2) + a_im.pow(2)
        prob = prob / (prob.sum(dim=-1, keepdim=True) + 1e-9)                         # Born rule
        return (prob * self.dim - 1.0).reshape(b, t, self.heads * self.dim)           # centred: uniform -> 0

    def forward(self, u: torch.Tensor, state=None):
        with torch.autocast(device_type=u.device.type, enabled=False):
            theta = self.angles(u)
            run = torch.cumsum(theta.double(), dim=1)
            if state is not None:
                run = run + state.double().unsqueeze(1)
            phase = torch.remainder(run, TWO_PI).float()
            y = self.read(phase, u)
        last = phase[:, -1]
        return self.out(y.to(u.dtype)), (last.detach() if not self.training else last)


def norm_of_state(mixer: QMixer, phase: torch.Tensor) -> torch.Tensor:
    """Length of psi for a carried phase (always 1 per head: the turns keep it)."""
    p0 = mixer.start()
    c, s = torch.cos(phase), torch.sin(phase)
    re = c * p0[..., 0] - s * p0[..., 1]
    im = s * p0[..., 0] + c * p0[..., 1]
    return (re.pow(2) + im.pow(2)).sum(dim=-1).sqrt()


def transplant(ckpt: dict, pattern: str, q_heads: int = 4, q_dim: int = 256, seed: int = 0) -> dict:
    """A Q core from a trained generation-8 checkpoint: same width, MLP and token table; the blocks marked "Q" in
    `pattern` get a new Q memory (output starts at zero), the others keep their memory if the kind is unchanged."""
    from evo.engine.architecture_factory import build_model

    old_cfg = ckpt["config"]
    old_pattern = old_cfg["pattern"]
    if len(pattern) != len(old_pattern):
        raise ValueError(f"pattern {pattern} must have as many blocks as {old_pattern}")
    for a, b in zip(old_pattern, pattern):
        if b != "Q" and b != a:
            raise ValueError(f"a block can only stay what it was or become Q ({old_pattern} -> {pattern})")
    cfg = {**old_cfg, "pattern": pattern, "q_heads": q_heads, "q_dim": q_dim}
    torch.manual_seed(seed)
    model = build_model(cfg)
    new = model.state_dict()
    old = ckpt["model_state_dict"]
    copied, fresh = 0, 0
    for k in new:
        i = int(k.split(".")[1]) if k.startswith("blocks.") else -1
        if i >= 0 and ".mixer." in k and pattern[i] == "Q":
            fresh += new[k].numel()
            continue
        if k in old and old[k].shape == new[k].shape:
            new[k] = old[k].to(new[k].dtype)
            copied += new[k].numel()
        else:
            fresh += new[k].numel()
    for i, kind in enumerate(pattern):
        if kind == "Q":
            new[f"blocks.{i}.mixer.out.weight"].zero_()
    model.load_state_dict(new)
    return {"config": cfg, "model_state_dict": model.state_dict(), "transplant": {
        "from": ckpt.get("name") or ckpt.get("candidate") or "checkpoint", "old_pattern": old_pattern, "pattern": pattern,
        "copied_parameters": copied, "new_parameters": fresh}}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="build a NOVA-Q core from a trained generation-8 core")
    ap.add_argument("--from", dest="src", required=True, help="checkpoint or release file (nova_model_fp32.pt)")
    ap.add_argument("--pattern", required=True, help="e.g. QQQQQQQ (pure Q) or NQNQNQN (mixed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args(argv)
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    ck["model_state_dict"] = {k: v.float() if v.is_floating_point() else v for k, v in ck["model_state_dict"].items()}
    out = transplant(ck, args.pattern, args.heads, args.dim)
    torch.save(out, args.out)
    print(json.dumps(out["transplant"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

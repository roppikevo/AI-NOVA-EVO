"""
Writing token by token with the NOVA core at constant cost per token.

The plain model call with a carried state is not enough for exact stepping: every block also looks at the
last few inputs through its local convolution. Generation therefore used to re-read the whole window for
every new token. The stepper keeps, per block, the recurrent state AND the last (kernel - 1) inputs, so
one token costs one small step however long the text already is - with the same logits as a full pass.

    st = Stepper(model)
    logits = st.prime(prompt_ids)        # read the prompt in one batched pass
    logits = st.step(next_id)            # then one token at a time
"""

from __future__ import annotations

import os

import torch


class Stepper:
    def __init__(self, model: torch.nn.Module) -> None:
        self.m = model.eval()
        self.layers = []
        for blk in model.blocks:
            if blk.d_state != blk.d_model:
                raise ValueError("Stepper supports blocks with d_state == d_model")
            w = torch.cat([blk.forget_proj.weight, blk.input_proj.weight, blk.fusion_proj.weight]).detach()
            b = torch.cat([blk.forget_proj.bias, blk.input_proj.bias, blk.fusion_proj.bias]).detach()
            self.layers.append({"blk": blk, "w": w, "b": b, "cw": blk.local_conv.weight.detach()[:, 0, :].t().contiguous(),
                                "cb": blk.local_conv.bias.detach(), "k": blk.conv_kernel, "d": blk.d_state})
        self.state: list[torch.Tensor] = []
        self.buf: list[torch.Tensor] = []

    def reset(self, batch: int = 1) -> None:
        p = self.m.embedding.weight
        self.state = [torch.zeros(batch, l["d"], dtype=p.dtype, device=p.device) for l in self.layers]
        self.buf = [torch.zeros(batch, l["k"] - 1, l["d"], dtype=p.dtype, device=p.device) for l in self.layers]

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.m.final_norm(x)
        if getattr(self.m, "embed_out", None) is not None:
            x = self.m.embed_out(x)
        return self.m.lm_head(x)

    @torch.no_grad()
    def prime(self, ids: torch.Tensor) -> torch.Tensor:
        """Read a whole prompt [batch, T] in one pass; returns the logits after its last token."""
        self.reset(ids.shape[0])
        x = self.m.embedding(ids)
        if getattr(self.m, "embed_in", None) is not None:
            x = self.m.embed_in(x)
        for n, l in enumerate(self.layers):
            u = l["blk"].norm(x)
            keep = l["k"] - 1
            if keep:
                tail = u[:, -keep:]
                self.buf[n] = torch.cat([self.buf[n][:, tail.shape[1]:], tail], dim=1)
            x, self.state[n] = l["blk"](x)
        return self._head(x[:, -1])

    @torch.no_grad()
    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """One new token per sequence: ids [batch] -> logits [batch, vocab]."""
        if not self.state:
            self.reset(ids.shape[0])
        x = self.m.embedding(ids)
        if getattr(self.m, "embed_in", None) is not None:
            x = self.m.embed_in(x)
        for n, l in enumerate(self.layers):
            blk, d = l["blk"], l["d"]
            u = blk.norm(x)
            window = torch.cat([self.buf[n], u.unsqueeze(1)], dim=1)           # [batch, k, d], oldest first
            c = (window * l["cw"]).sum(dim=1) + l["cb"]
            gates = torch.sigmoid(torch.addmm(l["b"], u, l["w"].t()) / blk.temperature)
            f, i, g = gates[:, :d], gates[:, d:2 * d], gates[:, 2 * d:]
            s = f * self.state[n] + i * u
            x = x + blk.output_proj(g * s + (1.0 - g) * c)
            self.state[n] = s
            if l["k"] > 1:
                self.buf[n] = window[:, 1:]
        return self._head(x)

    def state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.state + self.buf)


@torch.no_grad()
def generate_ids(model: torch.nn.Module, prompt: list[int], max_new: int, temperature: float = 0.0, eos: int = 3) -> list[int]:
    """Greedy or sampled continuation with the stepper (batch 1)."""
    st = Stepper(model)
    logits = st.prime(torch.tensor([prompt]))
    out: list[int] = []
    for _ in range(max_new):
        last = logits[0].float()
        nxt = int(torch.multinomial(torch.softmax(last / temperature, -1), 1)) if temperature > 0 else int(last.argmax())
        if nxt == eos:
            break
        out.append(nxt)
        logits = st.step(torch.tensor([nxt]))
    return out


def supports(model: torch.nn.Module) -> bool:
    """True for a NOVA scan core the stepper can drive exactly (anything else falls back to the window)."""
    blocks = getattr(model, "blocks", None)
    if not blocks or not all(hasattr(model, a) for a in ("embedding", "final_norm", "lm_head")):
        return False
    need = ("norm", "local_conv", "forget_proj", "input_proj", "fusion_proj", "output_proj", "temperature", "conv_kernel")
    return all(all(hasattr(b, a) for a in need) and getattr(b, "d_state", 0) == getattr(b, "d_model", -1) for b in blocks)


class Writer:
    """Next-token logits for a text that grows one token at a time.

    A NOVA core is driven by the stepper (constant cost per token); any other model re-reads its last
    `window` tokens, as generation always did. NOVA_SLOW_GEN=1 forces the old way (to compare)."""

    def __init__(self, model: torch.nn.Module, ids: list[int], device: str = "cpu", window: int = 128) -> None:
        self.model, self.device, self.window = model, device, window
        self.fast = supports(model) and not os.environ.get("NOVA_SLOW_GEN")
        # generation 8 carries its whole state through the plain model call
        self.carry = bool(getattr(model, "carries_state", False)) and not os.environ.get("NOVA_SLOW_GEN")
        with torch.no_grad():
            if self.carry:
                out, self.states = model(torch.tensor([ids], device=device))
                self.logits = out[0, -1]
            elif self.fast:
                self.st = Stepper(model)
                self.logits = self.st.prime(torch.tensor([ids], device=device))[0]
            else:
                self.x = torch.tensor([ids], device=device)
                self.logits = self._window()

    def _window(self) -> torch.Tensor:
        out = self.model(self.x)
        return (out[0] if isinstance(out, (tuple, list)) else out)[0, -1]

    @torch.no_grad()
    def push(self, token: int) -> torch.Tensor:
        """Append a token; returns the logits for the one after it."""
        if self.carry:
            out, self.states = self.model(torch.tensor([[token]], device=self.device), self.states)
            self.logits = out[0, -1]
        elif self.fast:
            self.logits = self.st.step(torch.tensor([token], device=self.device))[0]
        else:
            self.x = torch.cat([self.x, torch.tensor([[token]], device=self.device)], dim=1)[:, -self.window:]
            self.logits = self._window()
        return self.logits

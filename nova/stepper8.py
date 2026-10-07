"""
Writing token by token with a generation-8 core, fused: the same numbers as the model's own one-token call,
with fewer and larger operations.

The model's own call (`model(token, states)`) goes block by block through modules built for whole sequences.
For one token on a processor that is mostly overhead: many small products where a few large ones do. Here every
block's weights are joined once (the three gates of the memory; the two halves of the non-linear layer; a slot
block's write and query) and a token costs four products per block. Blocks this file has no fused form for
(the recurrent unit, hash, matrix memory, window) run through their own one-token step, so any pattern works.

    st = Stepper8(model)
    logits = st.prime(prompt_ids)        # read the prompt in one batched pass
    logits = st.step(next_id)            # then one token at a time
    st.states                            # the same state the model's own call carries
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from nova.core8 import MAX_DECAY, Gen7Mixer, SlotMixer, state_bytes

F_MIN = math.exp(-MAX_DECAY)


class Stepper8:
    def __init__(self, model: torch.nn.Module) -> None:
        self.m = model.eval()
        self.layers = []
        with torch.no_grad():
            for blk in model.blocks:
                mx = blk.mixer
                layer = {"blk": blk, "n1": blk.norm1.weight, "n2": blk.norm2.weight, "eps1": blk.norm1.eps, "eps2": blk.norm2.eps,
                         "ab_w": torch.cat([blk.fc_a.weight, blk.fc_b.weight]).contiguous(), "ab_b": torch.cat([blk.fc_a.bias, blk.fc_b.bias]),
                         "h": blk.fc_a.out_features, "kind": "other"}
                if isinstance(mx, Gen7Mixer) and mx.kernel > 1:
                    d, temp = mx.width, mx.temperature.float()
                    bias = mx.gates.bias.float().clone()
                    bias[:d] += mx.forget_bias                                    # the forget gate's offset, folded in
                    layer.update({"kind": "N", "d": d, "gw": (mx.gates.weight.float() / temp).contiguous(), "gb": bias / temp,
                                  "cw": mx.conv.weight[:, 0, :].t().contiguous(), "cb": mx.conv.bias})
                elif isinstance(mx, SlotMixer):
                    layer.update({"kind": "S", "wq_w": torch.cat([mx.write.weight, mx.query.weight]).contiguous(),
                                  "wq_b": torch.cat([mx.write.bias, mx.write.bias.new_zeros(mx.query.out_features)]),
                                  "nw": mx.write.out_features})
                self.layers.append(layer)
        self.states: list | None = None

    @torch.no_grad()
    def prime(self, ids: torch.Tensor) -> torch.Tensor:
        """Read a whole prompt [batch, T] in one pass; returns the logits after its last token."""
        logits, self.states = self.m(ids)
        return logits[:, -1]

    @torch.no_grad()
    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """One new token per sequence: ids [batch] -> logits [batch, vocab]."""
        if self.states is None:
            return self.prime(ids[:, None])
        m = self.m
        x = m.embedding(ids)
        states = self.states
        for n, l in enumerate(self.layers):
            blk, mx = l["blk"], l["blk"].mixer
            u = F.rms_norm(x, (x.shape[-1],), l["n1"], l["eps1"])
            if l["kind"] == "N":
                s, buf = states[n]
                d = l["d"]
                window = torch.cat([buf, u.unsqueeze(1)], dim=1)                 # [batch, kernel, d], oldest first
                c = (window * l["cw"]).sum(dim=1) + l["cb"]
                gates = torch.sigmoid(torch.addmm(l["gb"], u, l["gw"].t()))
                s = gates[:, :d].clamp(min=F_MIN) * s + gates[:, d:2 * d] * u
                g = gates[:, 2 * d:]
                x = x + mx.out(g * s + (1.0 - g) * c)
                states[n] = (s, window[:, 1:])
            elif l["kind"] == "S":
                table = states[n]
                wq = torch.addmm(l["wq_b"], u, l["wq_w"].t())
                w = wq[:, :l["nw"]]
                share = (torch.softmax(w[:, mx.ds:mx.ds + mx.slots], dim=-1) * torch.sigmoid(w[:, -1:])).unsqueeze(-1)
                table = (1.0 - share.clamp(max=1.0 - F_MIN)) * table + share * w[:, None, :mx.ds]
                read, _ = mx.read(wq[:, l["nw"]:].view(-1, mx.heads, mx.dk), table)
                x = x + mx.out(read.reshape(-1, mx.heads * mx.dv))
                states[n] = table
            else:
                y, states[n] = mx(u.unsqueeze(1), states[n])
                x = x + y[:, 0]
            ab = torch.addmm(l["ab_b"], F.rms_norm(x, (x.shape[-1],), l["n2"], l["eps2"]), l["ab_w"].t())
            h = l["h"]
            x = x + blk.fc_out(F.gelu(ab[:, :h]) * ab[:, h:])
        return m.lm_head(m.final_norm(x))

    def state_bytes(self) -> int:
        return state_bytes(self.states) if self.states is not None else 0


def supports(model: torch.nn.Module) -> bool:
    """True for a generation-8 core in full precision (the fused step works on plain float tensors)."""
    return bool(getattr(model, "carries_state", False)) and hasattr(model, "blocks") and model.embedding.weight.dtype == torch.float32 \
        and all(hasattr(b, "fc_a") and hasattr(b, "norm1") for b in model.blocks)

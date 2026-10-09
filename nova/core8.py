"""
NOVA core, generation 8: recurrent blocks with a fixed-size state, built to close the gap to a transformer
of the same size without giving up what the NOVA core is for (constant memory and constant cost per token).

What the generation-7 block lacked, measured against a same-size transformer (9-12 % higher loss):
no non-linear layer after the memory, a memory that is only a running average of its inputs, and no way
to look something up by content. Generation 8 adds these, each as a part with a state that does NOT grow
with the text:

  every block      x = x + mixer(norm(x));  x = x + gated_mlp(norm(x))
  mixer "L"  LRU   a gated linear recurrent unit (after Griffin/Hawk, De et al. 2024): a short causal
                   convolution, then h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * x_t) with a decay a_t
                   between 0 and 1 chosen per channel and per token; multiplied with a second, non-recurrent
                   branch.   state: one vector + the last (kernel - 1) inputs
  mixer "M"  MEM   matrix memory: every head keeps a small matrix S_t = f_t * S_{t-1} + i_t * k_t v_t^T and
                   reads it with a query, out_t = q_t^T S_t (gated retention; content-based recall).
                   state: heads x d_head x d_head numbers
  mixer "W"  WIN   attention over the last `window` tokens only (rotary positions).
                   state: the last (window - 1) keys and values

  mixer "H"  HASH  a table of slots addressed by content (a learned hash table): a token writes its value
                   under an address computed from its context and reads what is stored under addresses of its
                   own; newer writes replace older ones.   state: slots x slot_dim numbers
  mixer "N"        the generation-7 memory as it was (for comparison: what the non-linear layer alone adds)
  mixer "Q"        NOVA-Q: a complex unit state turned by every token, read like a measurement (nova/quantum.py).
                   state: heads x dim phases

A model is a string of mixers, e.g. "LLLLLLL" (pure recurrent), "LLMLLML", "LLWLLWL".

Training reads a whole sequence at once: the LRU recurrence is evaluated in closed form inside short
chunks (plain float32 tensor operations, no Python loop over tokens), the matrix memory and the window as
a masked T x T product. Writing calls the same code with one token and the carried state:

    logits, states = model(input_ids)             # training / scoring / reading a prompt
    logits, states = model(next_token, states)    # one token; the state has the same size after any text

Both ways give the same logits (tests/test_core8.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nova.transformer_lm import rope

MAX_DECAY = 4.5      # per token: a_t >= exp(-4.5) = 0.011 (the state is practically replaced in one step)
SCAN_CHUNK = 16      # exp(MAX_DECAY * SCAN_CHUNK) = e^72 fits into float32
READ_CHUNK = 128     # outside training, longer texts are read in pieces of this many tokens (e^-L of the hash table stays in float64)


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight - as in the generation-7 core, in one fused operation where available."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)
        return self.weight * (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps))


def two_linear(first: nn.Linear, second: nn.Linear, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two linear layers on the same input as one product (same numbers, one operation on the card)."""
    if x.shape[0] * x.shape[1] <= 64:              # a few tokens: joining the weights would cost more than it saves
        return first(x), second(x)
    y = F.linear(x, torch.cat([first.weight, second.weight]), torch.cat([first.bias, second.bias]))
    return y[..., :first.out_features], y[..., first.out_features:]


def lru_scan(log_a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    """h_t = exp(log_a_t) * h_{t-1} + b_t along dim 1 for [batch, time, ...]; log_a in [-MAX_DECAY, 0].

    No loop over tokens and none over chunks: the sequence is cut into chunks of SCAN_CHUNK tokens,
      * inside every chunk (all chunks at once), with L_t = sum of log_a from the chunk's start to t,
            inner_t = e^{L_t} * sum_{k<=t} b_k e^{-L_k}                    (the chunk as if it started from zero)
      * the state at the start of chunk j is a decayed sum of the ends of the chunks before it,
            start_j = sum_{i<j} e^{S_{j-1} - S_i} * inner_end_i (+ e^{S_{j-1}} h0),   S = running total of log_a
        (all exponents here are <= 0),
      * h_t = inner_t + e^{L_t} * start_j.
    About 15 tensor operations whatever the length; e^{-L} stays below e^{MAX_DECAY * SCAN_CHUNK} = e^72."""
    batch, time, chunk = b.shape[0], b.shape[1], SCAN_CHUNK
    n = (time + chunk - 1) // chunk
    extra = n * chunk - time
    if extra:                                             # fill the last chunk with "keep the state, add nothing"
        log_a = F.pad(log_a, [0, 0] * (log_a.ndim - 2) + [0, extra])
        b = F.pad(b, [0, 0] * (b.ndim - 2) + [0, extra])
    L = log_a.reshape(batch, n, chunk, *log_a.shape[2:]).cumsum(dim=2)
    decay = torch.exp(L)
    inner = decay * (b.reshape(batch, n, chunk, *b.shape[2:]) * torch.exp(-L)).cumsum(dim=2)
    if n > 1 or h0 is not None:
        S = L[:, :, -1].cumsum(dim=1)                                     # [batch, n, ...]: total up to the end of chunk j
        before = torch.cat([torch.zeros_like(S[:, :1]), S[:, :-1]], dim=1)  # the same up to the start of chunk j
        start = None
        if n > 1:
            earlier = torch.ones(n, n, dtype=torch.bool, device=b.device).tril(diagonal=-1)
            earlier = earlier.reshape(1, n, n, *([1] * (S.ndim - 2)))
            weight = torch.exp((before.unsqueeze(2) - S.unsqueeze(1)).masked_fill(~earlier, float("-inf")))
            start = (weight * inner[:, :, -1].unsqueeze(1)).sum(dim=2)
        if h0 is not None:
            carried = torch.exp(before) * h0.unsqueeze(1)
            start = carried if start is None else start + carried
        inner = inner + decay * start.unsqueeze(2)
    out = inner.reshape(batch, n * chunk, *b.shape[2:])
    return out[:, :time] if extra else out


def lru_scan_reference(log_a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    """The recurrence token by token (for tests)."""
    h = torch.zeros_like(b[:, 0]) if h0 is None else h0
    out = []
    for t in range(b.shape[1]):
        h = torch.exp(log_a[:, t]) * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def turn(x: torch.Tensor, offset: int) -> torch.Tensor:
    """Rotary positions for [batch, heads, time, head_dim], exact for any offset (angles in float64, modulo 2 pi)."""
    t, half = x.shape[-2], x.shape[-1] // 2
    freq = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float64, device=x.device) / half))
    ang = torch.remainder(torch.arange(offset, offset + t, dtype=torch.float64, device=x.device)[:, None] * freq[None, :], 2 * math.pi)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:2 * half]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class BlockLinear(nn.Module):
    """A linear layer whose weight is block-diagonal (cheap gates: channels are mixed inside small groups)."""

    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        self.blocks, self.size = blocks, width // blocks
        self.weight = nn.Parameter(torch.randn(blocks, self.size, self.size) * (1.0 / math.sqrt(self.size)))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = torch.einsum("...gi,goi->...go", x.reshape(*shape[:-1], self.blocks, self.size), self.weight.to(x.dtype))
        return y.reshape(shape) + self.bias.to(x.dtype)


def two_gates(first: BlockLinear, second: BlockLinear, x: torch.Tensor) -> torch.Tensor:
    """Both block-diagonal gates in one product: [2, ..., width]."""
    shape = x.shape
    w = torch.cat([first.weight, second.weight], dim=1).to(x.dtype)            # [blocks, 2 * size, size]
    y = torch.einsum("...gi,goi->...go", x.reshape(*shape[:-1], first.blocks, first.size), w)
    y = y.reshape(*shape[:-1], first.blocks, 2, first.size).movedim(-2, 0).reshape(2, *shape)
    return y + torch.stack([first.bias, second.bias]).to(x.dtype).reshape(2, *([1] * (len(shape) - 1)), shape[-1])


class LruMixer(nn.Module):
    def __init__(self, d_model: int, expand: float = 1.0, kernel: int = 4, gate_blocks: int = 16, c: float = 8.0) -> None:
        super().__init__()
        width = max(gate_blocks, int(round(d_model * expand / gate_blocks)) * gate_blocks)
        self.width, self.kernel, self.c = width, kernel, c
        self.left = nn.Linear(d_model, width)
        self.right = nn.Linear(d_model, width)
        self.conv = nn.Conv1d(width, width, kernel_size=kernel, groups=width)
        self.gate_a = BlockLinear(width, gate_blocks)
        self.gate_x = BlockLinear(width, gate_blocks)
        # decay at full recurrence gate, a^c, spread uniformly over [0.9, 0.999]: memories of ~10 to ~1000 tokens
        target = torch.empty(width).uniform_(0.9, 0.999)
        base = target ** (1.0 / c)
        self.lam = nn.Parameter(torch.log(base) - torch.log1p(-base))
        self.out = nn.Linear(width, d_model)

    def step(self, u: torch.Tensor, state):
        """One token with a carried state: the same numbers as forward(), without the machinery for sequences."""
        h, buf = state
        u = u[:, 0]
        left, r = F.gelu(self.left(u)), self.right(u)
        window = torch.cat([buf.to(r.dtype), r.unsqueeze(1)], dim=1)                 # [batch, kernel, width], oldest first
        x = (window * self.conv.weight[:, 0, :].t()).sum(dim=1) + self.conv.bias
        gate_a, gate_x = torch.sigmoid(two_gates(self.gate_a, self.gate_x, x.float()))
        log_a = (-self.c * F.softplus(-self.lam.float()) * gate_a).clamp(min=-MAX_DECAY)
        h = torch.exp(log_a) * h.float() + torch.sqrt(-torch.expm1(2.0 * log_a) + 1e-6) * gate_x * x.float()
        return self.out(left * h.to(left.dtype)).unsqueeze(1), (h, window[:, 1:])

    def forward(self, u: torch.Tensor, state=None):
        b, t, _ = u.shape
        if t == 1 and state is not None and not self.training and self.kernel > 1:
            return self.step(u, state)
        left, r = two_linear(self.left, self.right, u)
        left = F.gelu(left)
        keep = self.kernel - 1
        buf = r.new_zeros(b, keep, self.width) if state is None else state[1].to(r.dtype)
        window = torch.cat([buf, r], dim=1)
        x = self.conv(window.transpose(1, 2)).transpose(1, 2)
        with torch.autocast(device_type=u.device.type, enabled=False):      # the recurrence needs full precision
            xf = x.float()
            gate_a, gate_x = torch.sigmoid(two_gates(self.gate_a, self.gate_x, xf))
            log_a = (-self.c * F.softplus(-self.lam.float()) * gate_a).clamp(min=-MAX_DECAY)
            gain = torch.sqrt(-torch.expm1(2.0 * log_a) + 1e-6)             # sqrt(1 - a^2): the state keeps its scale
            h = lru_scan(log_a, gain * gate_x * xf, None if state is None else state[0].float())
        y = self.out(left * h.to(left.dtype))
        return y, (h[:, -1], window[:, window.shape[1] - keep:].detach() if keep else buf)


class Gen7Mixer(nn.Module):
    """The generation-7 memory unchanged (running average of the inputs blended with a local convolution);
    as a mixer "N" it shows what the non-linear layer alone adds to the old core."""

    def __init__(self, d_model: int, kernel: int = 5, forget_bias: float = 1.125, temperature: float = 2.0) -> None:
        super().__init__()
        self.width, self.kernel = d_model, kernel
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel, groups=d_model)
        self.gates = nn.Linear(d_model, 3 * d_model)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.forget_bias = forget_bias
        self.out = nn.Linear(d_model, d_model)

    def step(self, u: torch.Tensor, state):
        """One token with a carried state: the same numbers as forward(), without the machinery for sequences."""
        s, buf = state
        u, d = u[:, 0], self.width
        window = torch.cat([buf.to(u.dtype), u.unsqueeze(1)], dim=1)                 # [batch, kernel, width], oldest first
        c = (window * self.conv.weight[:, 0, :].t()).sum(dim=1) + self.conv.bias
        temperature = self.temperature.float()
        pre = self.gates(u).float() / temperature
        g = torch.sigmoid(pre[:, 2 * d:])
        s = torch.sigmoid(pre[:, :d] + self.forget_bias / temperature).clamp(min=math.exp(-MAX_DECAY)) * s.float() + torch.sigmoid(pre[:, d:2 * d]) * u.float()
        return self.out((g * s + (1.0 - g) * c.float()).to(u.dtype)).unsqueeze(1), (s, window[:, 1:])

    def forward(self, u: torch.Tensor, state=None):
        b, t, d = u.shape
        if t == 1 and state is not None and not self.training and self.kernel > 1:
            return self.step(u, state)
        keep = self.kernel - 1
        buf = u.new_zeros(b, keep, d) if state is None else state[1].to(u.dtype)
        window = torch.cat([buf, u], dim=1)
        c = self.conv(window.transpose(1, 2)).transpose(1, 2)
        with torch.autocast(device_type=u.device.type, enabled=False):
            pre = self.gates(u).float() / self.temperature.float()
            f, i, g = pre[..., :d] + self.forget_bias / self.temperature.float(), pre[..., d:2 * d], torch.sigmoid(pre[..., 2 * d:])
            s = lru_scan(F.logsigmoid(f).clamp(min=-MAX_DECAY), torch.sigmoid(i) * u.float(), None if state is None else state[0].float())
            mixed = g * s + (1.0 - g) * c.float()
        return self.out(mixed.to(u.dtype)), (s[:, -1], window[:, window.shape[1] - keep:].detach() if keep else buf)


class MemMixer(nn.Module):
    """Matrix memory per head with input and forget gates; read by a query (no softmax, no growing cache)."""

    def __init__(self, d_model: int, heads: int = 8, positions: bool = True) -> None:
        super().__init__()
        if d_model % heads or (d_model // heads) % 2:
            raise ValueError(f"d_model {d_model} must split into {heads} heads of even size")
        self.heads, self.dh, self.positions = heads, d_model // heads, positions
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.gates = nn.Linear(d_model, 2 * heads)
        self.out = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():      # forget gates start at different time scales: about 2 ... 512 tokens
            horizon = torch.logspace(math.log2(2.0), math.log2(512.0), heads, base=2.0)
            keep = 1.0 - 1.0 / horizon
            self.gates.bias[:heads] = torch.log(keep) - torch.log1p(-keep)
            self.gates.bias[heads:] = 2.0
            self.gates.weight.mul_(0.1)

    def forward(self, u: torch.Tensor, state=None):
        b, t, d = u.shape
        q, k, v = self.qkv(u).view(b, t, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)     # [b, heads, t, dh]
        scale = 1.0 / math.sqrt(self.dh)
        past = 0 if state is None else int(state[1])
        if self.positions:        # keys are stored already turned by their position, so the memory knows "how far back"
            q, k = turn(q, past), turn(k, past)
        state = None if state is None else state[0]
        with torch.autocast(device_type=u.device.type, enabled=False):
            g = self.gates(u).float().transpose(1, 2)                         # [b, 2*heads, t]
            log_f = F.logsigmoid(g[:, :self.heads]).cumsum(dim=-1)            # [b, heads, t], decreasing
            gate_in = torch.sigmoid(g[:, self.heads:])
            diff = log_f.unsqueeze(-1) - log_f.unsqueeze(-2)                  # [b, heads, t(query), t(key)]
            causal = torch.ones(t, t, dtype=torch.bool, device=u.device).tril()
            weight = torch.exp(diff.masked_fill(~causal, float("-inf"))) * gate_in.unsqueeze(-2)
            qf, kf, vf = q.float(), k.float(), v.float()
            y = ((qf @ kf.transpose(-1, -2)) * scale * weight) @ vf           # [b, heads, t, dh]
            last = torch.exp(log_f[..., -1:] - log_f) * gate_in               # what is left of every token at the end
            memory = (kf * last.unsqueeze(-1)).transpose(-1, -2) @ vf         # [b, heads, dh, dh]
            if state is not None:
                old = state.float()
                decay = torch.exp(log_f)
                y = y + decay.unsqueeze(-1) * (qf @ old) * scale
                memory = memory + decay[..., -1, None, None] * old
            y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-6)   # per head: only the direction counts
        y = y.to(u.dtype).transpose(1, 2).reshape(b, t, d)
        return self.out(y), (memory.detach(), torch.tensor(past + t))


class SlotMixer(nn.Module):
    """A small table of memory slots. Every token writes its value into the slots its own content points to
    (a soft address), and reads by comparing a query with what the slots hold - recall by content from a
    memory of fixed size. A slot keeps what it has until something is written over it.

    Two options, both at the same size of the table:
      key_dim > 0   a slot is split into a key (what the query is compared with) and a value (what is read out);
                    by default the whole slot is both
      sharp         every reading head learns how sharply it chooses among the slots (starts as without it)"""

    def __init__(self, d_model: int, slots: int = 16, heads: int = 4, slot_dim: int = 0, key_dim: int = 0, sharp: bool = False) -> None:
        super().__init__()
        self.slots, self.heads, self.ds = slots, heads, slot_dim or d_model // 4
        if not 0 <= key_dim < self.ds:
            raise ValueError(f"key_dim {key_dim} must be smaller than the slot ({self.ds})")
        self.split = key_dim > 0
        self.dk, self.dv = (key_dim, self.ds - key_dim) if self.split else (self.ds, self.ds)
        self.write = nn.Linear(d_model, self.ds + slots + 1)          # value, address, write gate
        self.query = nn.Linear(d_model, heads * self.dk, bias=False)
        self.out = nn.Linear(heads * self.dv, d_model, bias=False)
        self.sharp = nn.Parameter(torch.zeros(heads)) if sharp else None

    def read(self, q: torch.Tensor, table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """What the queries [.., heads, dk] read from the table [.., slots, ds], and how they spread over the slots."""
        keys, values = (table[..., :self.dk], table[..., self.dk:]) if self.split else (table, table)
        match = q @ keys.transpose(-1, -2) / math.sqrt(self.dk)                                             # [.., heads, slots]
        if self.sharp is not None:
            match = match * torch.exp(self.sharp.float()).unsqueeze(-1)
        attention = torch.softmax(match, dim=-1)
        return attention @ values, attention

    def step(self, u: torch.Tensor, table: torch.Tensor):
        u = u[:, 0]
        w = self.write(u).float()
        share = (torch.softmax(w[:, self.ds:self.ds + self.slots], dim=-1) * torch.sigmoid(w[:, -1:])).unsqueeze(-1)   # [batch, slots, 1]
        keep = 1.0 - share.clamp(max=1.0 - math.exp(-MAX_DECAY))
        table = keep * table.float() + share * w[:, None, :self.ds]
        read, _ = self.read(self.query(u).float().view(-1, self.heads, self.dk), table)                     # [batch, heads, dv]
        return self.out(read.reshape(-1, 1, self.heads * self.dv).to(u.dtype)), table

    def forward(self, u: torch.Tensor, state=None):
        b, t, _ = u.shape
        if t == 1 and state is not None and not self.training:
            return self.step(u, state)
        w = self.write(u)
        q = self.query(u).view(b, t, self.heads, self.dk)
        with torch.autocast(device_type=u.device.type, enabled=False):
            w = w.float()
            value = w[..., :self.ds]
            share = torch.softmax(w[..., self.ds:self.ds + self.slots], dim=-1) * torch.sigmoid(w[..., -1:])   # [b, t, slots]
            log_keep = torch.log1p(-share.clamp(max=1.0 - math.exp(-MAX_DECAY))).unsqueeze(-1)                  # [b, t, slots, 1]
            table = lru_scan(log_keep, share.unsqueeze(-1) * value.unsqueeze(-2), None if state is None else state.float())
            read, _ = self.read(q.float(), table)                                                               # [b, t, heads, dv]
        return self.out(read.reshape(b, t, self.heads * self.dv).to(u.dtype)), table[:, -1].detach() if not self.training else table[:, -1]


class HashMixer(nn.Module):
    """A table of memory slots addressed by content, like a learned hash table.

    Writing: every token computes an address (a soft choice among the slots) from its own context and blends its
    value into the slots it points to; what was there fades by as much as is written over it.
    Reading: every token computes addresses of its own (one per head) and takes what those slots hold.
    A token can so find what was stored under the same address earlier - for instance what followed the same
    word last time - however far back that was, and the table has a fixed size.

    In training the table is never built for every position: reading slot j at time t gives
        sum_{k<=t} r_tj * w_kj * exp(L_tj - L_kj) * v_k,     L = running sum of log(1 - w),
    which is a T x T matrix product of (r * e^L) and (w * e^-L) - the same shape of work as attention, but the
    'keys' live in a fixed set of slots and newer writes replace older ones. Token by token the same numbers
    come from the stored table (float64 inside: e^-L can be large)."""

    def __init__(self, d_model: int, slots: int = 128, heads: int = 4, slot_dim: int = 0) -> None:
        super().__init__()
        self.slots, self.heads, self.ds = slots, heads, slot_dim or d_model // 4
        self.write = nn.Linear(d_model, slots + 1)                    # address and write gate
        self.value = nn.Linear(d_model, self.ds, bias=False)
        self.read = nn.Linear(d_model, heads * slots)
        self.out = nn.Linear(heads * self.ds, d_model, bias=False)
        self.sharp = nn.Parameter(torch.tensor(1.0))                  # sharpness of the addresses

    def step(self, u: torch.Tensor, table: torch.Tensor):
        u = u[:, 0]
        w_raw = self.write(u).float()
        sharp = F.softplus(self.sharp.float()) + 0.5
        share = torch.softmax(w_raw[:, :self.slots] * sharp, dim=-1) * torch.sigmoid(w_raw[:, -1:])       # [batch, slots]
        keep = 1.0 - share.clamp(max=1.0 - math.exp(-MAX_DECAY))
        table = keep.unsqueeze(-1) * table.float() + share.unsqueeze(-1) * self.value(u).float().unsqueeze(1)
        read = torch.softmax(self.read(u).float().view(-1, self.heads, self.slots) * sharp, dim=-1)     # [batch, heads, slots]
        got = (read @ table).reshape(-1, 1, self.heads * self.ds)
        return self.out(got.to(u.dtype)), table

    def forward(self, u: torch.Tensor, state=None):
        b, t, _ = u.shape
        if t == 1 and state is not None and not self.training:
            return self.step(u, state)
        w_raw, r_raw, value = self.write(u), self.read(u), self.value(u)
        with torch.autocast(device_type=u.device.type, enabled=False):
            sharp = F.softplus(self.sharp.double()) + 0.5
            w_raw, value = w_raw.double(), value.double()
            share = torch.softmax(w_raw[..., :self.slots] * sharp, dim=-1) * torch.sigmoid(w_raw[..., -1:])      # [b, t, slots]
            log_keep = torch.log1p(-share.clamp(max=1.0 - math.exp(-MAX_DECAY)))
            L = log_keep.cumsum(dim=1)                                                                           # [b, t, slots] <= 0
            read = torch.softmax(r_raw.double().view(b, t, self.heads, self.slots) * sharp, dim=-1).transpose(1, 2)  # [b, heads, t, slots]
            P = read * torch.exp(L).unsqueeze(1)                       # what a reader at time t sees of slot j ...
            Q = share * torch.exp(-L)                                  # ... of what was written at time k
            mix = P @ Q.transpose(1, 2).unsqueeze(1)                   # [b, heads, t(reader), t(writer)]
            mix = mix.masked_fill(~torch.ones(t, t, dtype=torch.bool, device=u.device).tril(), 0.0)
            got = mix @ value.unsqueeze(1)                             # [b, heads, t, ds]
            stored = Q.transpose(1, 2) @ value                         # [b, slots, ds]: everything written, undecayed
            if state is not None:
                old = state.double()
                got = got + P @ old.unsqueeze(1)
                stored = stored + old
            table = torch.exp(L[:, -1]).unsqueeze(-1) * stored
            got = got.transpose(1, 2).reshape(b, t, self.heads * self.ds)
        return self.out(got.to(u.dtype)), table.float().detach() if not self.training else table.float()


class WinMixer(nn.Module):
    """Attention over the last `window` tokens (the current one included); rotary positions."""

    def __init__(self, d_model: int, heads: int = 8, window: int = 32) -> None:
        super().__init__()
        if d_model % heads or (d_model // heads) % 2:
            raise ValueError(f"d_model {d_model} must split into {heads} heads of even size")
        self.heads, self.dh, self.window = heads, d_model // heads, window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, u: torch.Tensor, state=None):
        b, t, d = u.shape
        q, k, v = self.qkv(u).view(b, t, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        if state is not None:
            k, v = torch.cat([state[0].to(k.dtype), k], dim=2), torch.cat([state[1].to(v.dtype), v], dim=2)
        past = k.shape[2] - t
        pos_q = torch.arange(past, past + t, device=u.device)
        pos_k = torch.arange(past + t, device=u.device)
        mask = (pos_k[None, :] <= pos_q[:, None]) & (pos_k[None, :] > pos_q[:, None] - self.window)
        y = F.scaled_dot_product_attention(rope(q, past), rope(k, 0), v, attn_mask=mask)
        keep = self.window - 1
        start = max(0, k.shape[2] - keep)
        return self.out(y.transpose(1, 2).reshape(b, t, d)), (k[:, :, start:].detach(), v[:, :, start:].detach())


class Block8(nn.Module):
    def __init__(self, d_model: int, kind: str, mlp_hidden: int, heads: int, window: int, lru_expand: float, lru_kernel: int,
                 mem_positions: bool = True, slots: int = 16, hash_slots: int = 128, slot_key: int = 0, slot_sharp: bool = False,
                 q_heads: int = 4, q_dim: int = 256) -> None:
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(d_model)
        if kind == "L":
            self.mixer = LruMixer(d_model, expand=lru_expand, kernel=lru_kernel)
        elif kind == "N":
            self.mixer = Gen7Mixer(d_model)
        elif kind == "M":
            self.mixer = MemMixer(d_model, heads=heads, positions=mem_positions)
        elif kind == "S":
            self.mixer = SlotMixer(d_model, slots=slots, heads=max(1, heads // 2), key_dim=slot_key, sharp=slot_sharp)
        elif kind == "H":
            self.mixer = HashMixer(d_model, slots=hash_slots, heads=max(1, heads // 2))
        elif kind == "W":
            self.mixer = WinMixer(d_model, heads=heads, window=window)
        elif kind == "Q":
            from nova.quantum import QMixer

            self.mixer = QMixer(d_model, heads=q_heads, dim=q_dim)
        else:
            raise ValueError(f"unknown mixer '{kind}' (L, H, S, M, W, N or Q)")
        self.norm2 = RMSNorm(d_model)
        self.fc_a = nn.Linear(d_model, mlp_hidden)
        self.fc_b = nn.Linear(d_model, mlp_hidden)
        self.fc_out = nn.Linear(mlp_hidden, d_model)

    def forward(self, x: torch.Tensor, state=None):
        y, state = self.mixer(self.norm1(x), state)
        x = x + y
        a, b = two_linear(self.fc_a, self.fc_b, self.norm2(x))
        return x + self.fc_out(F.gelu(a) * b), state


class Nova8Model(nn.Module):
    carries_state = True      # model(token, states) continues exactly; the state does not grow with the text

    def __init__(self, vocab_size: int, d_model: int, pattern: str, mlp_hidden: int, heads: int = 8, window: int = 32,
                 lru_expand: float = 1.0, lru_kernel: int = 4, mem_positions: bool = True, slots: int = 16, hash_slots: int = 128, pad_token_id: int = 0,
                 slot_key: int = 0, slot_sharp: bool = False, q_heads: int = 4, q_dim: int = 256) -> None:
        super().__init__()
        self.pattern = pattern
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([Block8(d_model, kind, mlp_hidden, heads, window, lru_expand, lru_kernel, mem_positions, slots, hash_slots,
                                            slot_key, slot_sharp, q_heads, q_dim) for kind in pattern])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        keep = {id(m.gates) for m in self.modules() if isinstance(m, MemMixer)}
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)) and id(m) not in keep:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
        small = 0.02 / math.sqrt(2 * len(pattern))       # residual branches start small
        for blk in self.blocks:
            nn.init.normal_(blk.fc_out.weight, mean=0.0, std=small)
            nn.init.normal_(blk.mixer.out.weight, mean=0.0, std=small)
        with torch.no_grad():
            self.embedding.weight[pad_token_id].zero_()

    def forward(self, input_ids: torch.Tensor, states: list | None = None):
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids [batch, seq], got {tuple(input_ids.shape)}")
        if not self.training and input_ids.shape[1] > READ_CHUNK:
            # a long text is read in pieces with the carried state: same result, time and memory linear in its length
            parts = []
            for s in range(0, input_ids.shape[1], READ_CHUNK):
                logits, states = self._read(input_ids[:, s:s + READ_CHUNK], states)
                parts.append(logits)
            return torch.cat(parts, dim=1), states
        return self._read(input_ids, states)

    def _read(self, input_ids: torch.Tensor, states: list | None = None):
        x = self.embedding(input_ids)
        new_states = []
        for i, blk in enumerate(self.blocks):
            x, st = blk(x, None if states is None else states[i])
            new_states.append(st)
        return self.lm_head(self.final_norm(x)), new_states

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def state_bytes(states) -> int:
    if torch.is_tensor(states):
        return states.numel() * states.element_size()
    return sum(state_bytes(s) for s in states) if isinstance(states, (list, tuple)) else 0


def build_nova8(config: dict) -> Nova8Model:
    d = int(config["d_model"])
    pattern = str(config.get("pattern") or "L" * int(config.get("num_layers", 6)))
    return Nova8Model(vocab_size=int(config["vocab_size"]), d_model=d, pattern=pattern,
                      mlp_hidden=int(config.get("mlp_hidden") or 3 * d), heads=int(config.get("heads") or max(1, d // 64)),
                      window=int(config.get("window", 32)), lru_expand=float(config.get("lru_expand", 1.0)),
                      lru_kernel=int(config.get("lru_kernel", 4)), mem_positions=bool(config.get("mem_positions", True)), slots=int(config.get("slots", 16)), hash_slots=int(config.get("hash_slots", 128)),
                      slot_key=int(config.get("slot_key", 0)), slot_sharp=bool(config.get("slot_sharp", False)),
                      q_heads=int(config.get("q_heads", 4)), q_dim=int(config.get("q_dim", 256)))

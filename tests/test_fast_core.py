"""Faster arithmetic for the NOVA core: same results as the current code (no GPU, no server)."""

import torch

from evo.engine import speed_bench as sb
from evo.engine.architecture_factory import build_model
from nova.blocks_scan import ScanRefBackward
from nova.fast_scan import FastScan, scan_linear
from nova.stepper import Stepper, generate_ids

CFG = {"vocab_size": 16384, "d_model": 48, "d_state": 48, "num_layers": 3, "conv_kernel": 5, "forget_bias": 1.125,
       "learnable_initial_state": False}


def gates(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.sigmoid(torch.randn(shape, generator=g, dtype=torch.float64)).requires_grad_()
    i = torch.sigmoid(torch.randn(shape, generator=g, dtype=torch.float64)).requires_grad_()
    u = torch.randn(shape, generator=g, dtype=torch.float64).requires_grad_()
    return f, i, u, torch.randn(shape, generator=g, dtype=torch.float64)


def test_scan_linear_is_the_recurrence_in_both_directions():
    f, i, u, _ = gates((2, 13, 3))
    a, b = f.detach(), (i * u).detach()
    s, want = torch.zeros(2, 3, dtype=torch.float64), []
    for t in range(13):
        s = a[:, t] * s + b[:, t]
        want.append(s)
    assert torch.allclose(scan_linear(a, b)[1], torch.stack(want, 1))
    s, back = torch.zeros(2, 3, dtype=torch.float64), []
    for t in range(12, -1, -1):
        s = a[:, t] * s + b[:, t]
        back.append(s)
    assert torch.allclose(scan_linear(a, b, reverse=True)[1], torch.stack(back[::-1], 1))


def test_fast_scan_matches_the_core_in_values_and_gradients():
    for shape in ((3, 50, 8), (2, 127, 5), (1, 1, 4)):
        f, i, u, w = gates(shape)
        ref = ScanRefBackward.apply(f, i, u, None)
        new = FastScan.apply(f, i, u, None)
        assert torch.allclose(ref, new)
        ga = torch.autograd.grad((ref * w).sum(), [f, i, u])
        gb = torch.autograd.grad((new * w).sum(), [f, i, u])
        assert all(torch.allclose(x, y) for x, y in zip(ga, gb))


def test_fast_scan_gradients_are_exact_with_an_initial_state():
    f, i, u, _ = gates((2, 9, 4))
    s0 = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda *a: FastScan.apply(*a), (f, i, u, s0))


def model():
    torch.manual_seed(0)
    m = build_model(CFG).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim > 1:
                p.add_(torch.randn_like(p) * 0.05)
    return m


def test_stepper_gives_the_logits_of_a_full_pass():
    m, x = model(), torch.randint(12, 16384, (2, 30))
    with torch.no_grad():
        full = m(x)[0]
    st = Stepper(m)
    stepped = torch.stack([st.step(x[:, t]) for t in range(30)], 1)
    assert float((stepped - full).abs().max()) < 1e-4
    for cut in (1, 3, 17):                                   # prompt read in one pass, then steps
        st = Stepper(m)
        outs = [st.prime(x[:, :cut])] + [st.step(x[:, t]) for t in range(cut, 30)]
        assert float((torch.stack(outs, 1) - full[:, cut - 1:]).abs().max()) < 1e-4
    assert st.state_bytes() == 2 * 3 * (48 + 4 * 48) * 4


def test_stepper_generation_equals_rereading_the_window():
    m, prompt = model(), [4, 20, 21, 22, 23]
    ids, want = list(prompt), []
    with torch.no_grad():
        for _ in range(12):
            nxt = int(m(torch.tensor([ids]))[0][0, -1].argmax())
            want.append(nxt)
            ids.append(nxt)
    assert generate_ids(m, prompt, 12, eos=-1) == want


def test_speed_bench_runs_on_the_cpu():
    s = sb.scan_bench("cpu", shape=(2, 20, 8), repeats=1)
    assert s["max_abs_diff"] < 1e-4 and s["reference_ms"] > 0 and s["fast_ms"] > 0
    t = sb.train_step_bench(CFG, 2, "cpu", repeats=1, seq=16)
    assert t["reference_first_loss"] == t["fast_first_loss"] and t["speedup"] > 0
    r = {"device": "cpu", "scan": [s], "train_step": [t],
         "write": {"threads": 1, "read_tok_s": {"nova": 10, "transformer": 20},
                   "by_context": {"127": {"nova_today_window_tok_s": 1.0, "nova_old_step_tok_s": 3.0, "nova_stepper_tok_s": 9.0, "nova_state_kb": 1.0,
                                          "transformer_cache_tok_s": 5.0, "transformer_state_kb": 9.0}}}}
    assert "writing after 127 tokens" in sb.text(r) and "whole training step" in sb.text(r)


# ------------------------------------------------------------------ the stepper inside generation

class Tok:
    """Minimal tokenizer: ids are words, good enough to compare two ways of generating."""
    has_think = False

    def lang_id(self, lang):
        return 8 if lang == "py" else 4

    def encode(self, text):
        return [12 + (ord(c) % 200) for c in text]

    def decode(self, ids, skip_special=True):
        return "".join(chr(97 + i % 26) if i % 7 else "\n    " for i in ids)


def test_supports_only_the_nova_core():
    from nova.stepper import supports

    assert supports(model())
    assert not supports(build_model({**CFG, "arch": "transformer", "num_layers": 2, "n_heads": 4}))
    assert not supports(torch.nn.Linear(2, 2))


def test_writer_fast_and_window_give_the_same_tokens(monkeypatch):
    from nova.stepper import Writer

    m, ids = model(), [4, 30, 31, 32]
    monkeypatch.delenv("NOVA_SLOW_GEN", raising=False)
    fast = Writer(m, ids)
    monkeypatch.setenv("NOVA_SLOW_GEN", "1")
    slow = Writer(m, ids)
    assert fast.fast and not slow.fast
    for _ in range(20):
        a, b = int(fast.logits.argmax()), int(slow.logits.argmax())
        assert a == b and float((fast.logits - slow.logits).abs().max()) < 1e-4
        fast.push(a)
        slow.push(b)


def test_code_school_and_generate_write_the_same_with_and_without_the_stepper(monkeypatch):
    from evo.learning import code_school as cs
    from evo.learning.code_tasks import task_bank
    from nova.generate import generate

    m, tok, task = model(), Tok(), task_bank()[0]
    monkeypatch.delenv("NOVA_SLOW_GEN", raising=False)
    body_fast, text_fast = cs.write_body(m, tok, task, max_tokens=24), generate(m, tok, "ab", "sk", max_new_tokens=24)
    monkeypatch.setenv("NOVA_SLOW_GEN", "1")
    assert cs.write_body(m, tok, task, max_tokens=24) == body_fast
    assert generate(m, tok, "ab", "sk", max_new_tokens=24) == text_fast


def test_compare_speed_uses_the_stepper_for_nova_and_the_cache_for_the_transformer():
    from evo.engine import compare_arch as ca

    a = ca.cpu_speed(model(), 16384, prompt_len=12, gen=3, threads=1, repeats=1)
    b = ca.cpu_speed(build_model({**CFG, "arch": "transformer", "num_layers": 2, "n_heads": 4}), 16384, prompt_len=12, gen=3,
                     threads=1, repeats=1)
    assert a["state_kb_after_writing"] == round(3 * (48 + 4 * 48) * 4 / 1024, 1)        # state + window, constant
    assert b["state_kb_after_writing"] > a["state_kb_after_writing"] and a["write_tokens_per_s"] > 0

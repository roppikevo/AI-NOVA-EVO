"""Generation 8 of the NOVA core: same logits token by token as in one pass, a state that does not grow."""

import pytest
import torch

from evo.engine.architecture_factory import build_model
from nova import core8

BASE = {"arch": "nova8", "vocab_size": 200, "d_model": 32, "mlp_hidden": 48, "heads": 4, "window": 5, "hash_slots": 24}
PATTERNS = ["LL", "NL", "LM", "LS", "LH", "LW", "LMSHWN", "NSW", "NN"]


def _model(pattern, seed=0):
    torch.manual_seed(seed)
    m = build_model({**BASE, "pattern": pattern}).eval()
    with torch.no_grad():      # make the residual branches matter (they start near zero)
        for blk in m.blocks:
            blk.fc_out.weight.mul_(20)
            blk.mixer.out.weight.mul_(20)
    return m


def test_scan_equals_the_recurrence_step_by_step():
    torch.manual_seed(0)
    for t, lo in ((70, -core8.MAX_DECAY), (33, -0.01), (16, -core8.MAX_DECAY), (1, -1.0), (129, -2.0)):
        log_a = torch.empty(3, t, 7).uniform_(lo, 0.0)
        if t == 16:
            log_a.fill_(-core8.MAX_DECAY)                       # the hardest case for the closed form
        b, h0 = torch.randn(3, t, 7), torch.randn(3, 7)
        for start in (h0, None):
            got = core8.lru_scan(log_a, b, start)
            assert torch.allclose(got, core8.lru_scan_reference(log_a, b, start), atol=1e-4, rtol=1e-4)
            assert torch.isfinite(got).all()


def test_scan_with_a_shared_decay_per_slot_and_its_gradient():
    torch.manual_seed(1)
    log_a = torch.empty(2, 50, 5, 1).uniform_(-3.0, 0.0).requires_grad_(True)       # one decay per slot, many values
    b = torch.randn(2, 50, 5, 6, requires_grad=True)
    h0 = torch.randn(2, 5, 6)
    got, want = core8.lru_scan(log_a, b, h0), core8.lru_scan_reference(log_a, b, h0)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)
    ga = torch.autograd.grad(got.pow(2).sum(), [log_a, b])
    gw = torch.autograd.grad(want.pow(2).sum(), [log_a, b])
    assert all(torch.allclose(x, y, atol=2e-3, rtol=2e-3) and torch.isfinite(x).all() for x, y in zip(ga, gw))


def test_two_gates_in_one_product_equal_two_separate_gates():
    torch.manual_seed(2)
    a, b = core8.BlockLinear(32, 4), core8.BlockLinear(32, 4)
    with torch.no_grad():
        a.bias.normal_(); b.bias.normal_()
    x = torch.randn(3, 9, 32)
    both = core8.two_gates(a, b, x)
    assert torch.allclose(both[0], a(x), atol=1e-6) and torch.allclose(both[1], b(x), atol=1e-6)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_token_by_token_gives_the_same_logits_as_one_pass(pattern):
    m = _model(pattern)
    x = torch.randint(12, 200, (2, 41))
    with torch.no_grad():
        full, _ = m(x)
        parts, states, pos = [], None, 0
        for size in (17, 1, 1, 6, 1, 15):                        # a prompt, then single tokens and pieces
            out, states = m(x[:, pos:pos + size], states)
            parts.append(out)
            pos += size
    stepped = torch.cat(parts, dim=1)
    assert (full - stepped).abs().max() < 2e-4 * max(1.0, float(full.abs().max()))


@pytest.mark.parametrize("pattern", PATTERNS)
def test_no_look_ahead(pattern):
    m = _model(pattern)
    x = torch.randint(12, 200, (1, 30))
    y = x.clone()
    y[0, 20:] = torch.randint(12, 200, (10,))
    with torch.no_grad():
        a, b = m(x)[0], m(y)[0]
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-5) and not torch.allclose(a[:, 20:], b[:, 20:], atol=1e-3)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_the_state_does_not_grow_with_the_text(pattern):
    m = _model(pattern)
    with torch.no_grad():
        _, short = m(torch.randint(12, 200, (1, 12)))
        _, long = m(torch.randint(12, 200, (1, 300)))
    assert core8.state_bytes(short) == core8.state_bytes(long) > 0


def test_it_learns_and_runs_in_half_precision():
    torch.manual_seed(0)
    m = build_model({**BASE, "pattern": "LMW"})
    data = torch.randint(12, 60, (8, 24))
    data[:, 12:] = data[:, :12]                                  # the second half repeats the first: needs memory
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        logits, _ = m(data[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 200), data[:, 1:].reshape(-1))
        first = first or float(loss.detach())
        opt.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
    assert float(loss) < 0.6 * first
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(m(data)[0].float()).all()


def test_checkpoint_round_trip_and_writer(tmp_path):
    from nova.generate import load_checkpoint_model
    from nova.stepper import Writer

    cfg = {**BASE, "pattern": "LW"}
    m = _model("LW")
    torch.save({"config": cfg, "model_state_dict": m.state_dict()}, tmp_path / "m.pt")
    again, _ = load_checkpoint_model(tmp_path / "m.pt")
    ids = [4, 20, 31, 45, 50, 61]
    w = Writer(again.eval(), ids[:3])
    assert w.carry
    for t in ids[3:]:
        w.push(t)
    with torch.no_grad():
        want = m(torch.tensor([ids]))[0][0, -1]
    assert torch.allclose(w.logits, want, atol=1e-4)


def test_a_long_text_is_read_in_pieces_with_the_same_result():
    m = _model("LMSHW")
    x = torch.randint(12, 200, (1, 2 * core8.READ_CHUNK + 37))
    with torch.no_grad():
        pieces, states = m(x)
        whole, _ = m._read(x)
    assert (pieces - whole).abs().max() < 2e-4 * max(1.0, float(whole.abs().max()))
    assert core8.state_bytes(states) == core8.state_bytes(m(x[:, :40])[1])


@pytest.mark.parametrize("options", [{"slot_key": 3}, {"slot_sharp": True}, {"slot_key": 5, "slot_sharp": True, "slots": 6}])
def test_slot_options_keep_the_exact_token_by_token_reading_and_the_size_of_the_state(options):
    torch.manual_seed(0)
    m = build_model({**BASE, "pattern": "NSS", **options}).eval()
    plain = build_model({**BASE, "pattern": "NSS", "slots": options.get("slots", 16)}).eval()
    with torch.no_grad():
        for blk in m.blocks:
            blk.fc_out.weight.mul_(20)
            blk.mixer.out.weight.mul_(20)
        for blk in m.blocks[1:]:
            blk.mixer.query.weight.mul_(30)                          # sharp reading: the choice among slots matters
            if blk.mixer.sharp is not None:
                blk.mixer.sharp.copy_(torch.tensor([0.7, -0.3]))
        x = torch.randint(12, 200, (2, 30))
        full, st_full = m(x)
        parts, states, pos = [], None, 0
        for size in (11, 1, 1, 7, 1, 9):
            out, states = m(x[:, pos:pos + size], states)
            parts.append(out)
            pos += size
        _, st_plain = plain(x)
    assert (full - torch.cat(parts, dim=1)).abs().max() < 2e-4 * max(1.0, float(full.abs().max()))
    assert core8.state_bytes(st_full) == core8.state_bytes(states) == core8.state_bytes(st_plain)     # the options cost no state
    mixer = m.blocks[1].mixer
    assert mixer.dk + (mixer.dv if mixer.split else 0) == mixer.ds and (mixer.sharp is not None) == bool(options.get("slot_sharp"))


def test_sharpness_starts_as_no_sharpness_and_old_weights_still_load():
    torch.manual_seed(0)
    plain = build_model({**BASE, "pattern": "NS"}).eval()
    sharp = build_model({**BASE, "pattern": "NS", "slot_sharp": True}).eval()
    missing = sharp.load_state_dict(plain.state_dict(), strict=False)
    assert missing.missing_keys == ["blocks.1.mixer.sharp"] and not missing.unexpected_keys
    x = torch.randint(12, 200, (2, 20))
    with torch.no_grad():
        assert torch.allclose(plain(x)[0], sharp(x)[0], atol=1e-6)

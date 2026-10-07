"""The fused one-token step of generation 8: the same numbers as the model's own call, for every pattern."""

import pytest
import torch

from evo.engine.architecture_factory import build_model
from nova import stepper8
from nova.stepper import Writer

BASE = {"arch": "nova8", "vocab_size": 200, "d_model": 32, "mlp_hidden": 48, "heads": 4, "window": 5, "hash_slots": 24, "slots": 6}


def _model(pattern, **options):
    torch.manual_seed(0)
    m = build_model({**BASE, "pattern": pattern, **options}).eval()
    with torch.no_grad():
        for blk in m.blocks:
            blk.fc_out.weight.mul_(20)
            blk.mixer.out.weight.mul_(20)
    return m


@pytest.mark.parametrize("pattern,options", [("NSNS", {}), ("NN", {}), ("NSW", {}), ("LHM", {}), ("NS", {"slot_key": 3, "slot_sharp": True})])
def test_fused_step_equals_the_models_own_step_and_one_pass(pattern, options):
    m = _model(pattern, **options)
    assert stepper8.supports(m)
    x = torch.randint(12, 200, (2, 40))
    with torch.no_grad():
        full, _ = m(x)
        _, states = m(x[:, :25])
        st = stepper8.Stepper8(m)
        got = [st.prime(x[:, :25])]
        own = []
        for t in range(25, 40):
            out, states = m(x[:, t:t + 1], states)
            own.append(out[:, 0])
            got.append(st.step(x[:, t]))
    scale = max(1.0, float(full.abs().max()))
    assert (torch.stack(got[1:], 1) - torch.stack(own, 1)).abs().max() < 1e-5 * scale          # the model's own one-token call
    assert (torch.stack(got, 1) - full[:, 24:]).abs().max() < 2e-4 * scale                      # and the single pass
    assert st.state_bytes() == stepper8.state_bytes(states) > 0


def test_the_writer_uses_the_fused_step_and_writes_the_same_tokens(monkeypatch):
    m = _model("NSNS")
    ids = [5, 17, 80, 33, 12]
    fused = Writer(m, list(ids))
    assert fused.fused is not None
    monkeypatch.setenv("NOVA_PLAIN_STEP", "1")
    plain = Writer(m, list(ids))
    assert plain.fused is None
    for _ in range(12):
        a, b = fused.logits, plain.logits
        assert torch.allclose(a, b, atol=1e-5) and int(a.argmax()) == int(b.argmax())
        tok = int(a.argmax())
        fused.push(tok)
        plain.push(tok)


def test_half_precision_weights_fall_back_to_the_plain_step():
    m = _model("NS").half()
    assert not stepper8.supports(m)

"""The probe of the slot memories: its own pass equals the model's, the ablations change what they should."""

import numpy as np
import torch

from evo.engine import slot_probe as sp
from evo.engine.architecture_factory import build_model

CFG = {"vocab_size": 120, "arch": "nova8", "d_model": 32, "pattern": "NSNS", "mlp_hidden": 48, "heads": 4, "slots": 8}


def _model():
    torch.manual_seed(0)
    m = build_model(CFG).eval()
    with torch.no_grad():
        for blk in m.blocks:                       # an untrained core barely uses its blocks: make them count
            blk.mixer.out.weight.mul_(30)
            blk.fc_out.weight.mul_(30)
    return m


def test_the_patched_pass_is_the_models_own_pass_and_leaves_no_trace():
    m = _model()
    rows = np.random.default_rng(0).integers(12, 120, size=(40, 24)).astype(np.int32)
    before = sp.loss(m, rows)
    r = sp.probe(m, rows, batch=16)
    assert r["slot_blocks"] == [1, 3] and r["patched_pass_differs_by"] < 1e-4
    assert abs(sp.loss(m, rows) - before) < 1e-6 and "forward" not in m.blocks[1].mixer.__dict__
    assert abs(r["loss"] - before) < 1e-4


def test_ablations_and_statistics():
    m = _model()
    rows = np.random.default_rng(1).integers(12, 120, size=(40, 24)).astype(np.int32)
    r = sp.probe(m, rows, batch=16)
    for i in ("1", "3"):
        b = r["blocks"][i]
        assert 0 < b["gate_mean"] < 1 and 1 <= b["address_slots_per_token"] <= 8 and 1 <= b["slots_in_use"] <= 8
        assert len(b["half_life_tokens"]) == 8 and all(h > 0 for h in b["half_life_tokens"])
        assert 1 <= b["read_slots_per_query"] <= 8 and b["output_vs_stream"] > 0
        assert abs(b["loss_off"] - r["loss"]) > 1e-4               # the block does something
    assert abs(r["loss_all_off"] - r["loss"]) > 1e-4
    assert all(np.isfinite(r[f"loss_all_{mode}"]) for mode in sp.MODES[1:])
    t = sp.text("m", r)
    assert "block 1" in t and "slots in use" in t and "uniform write" in t
    gen7 = build_model({"vocab_size": 120, "d_model": 32, "d_state": 32, "num_layers": 2, "conv_kernel": 5, "forget_bias": 1.125,
                        "learnable_initial_state": False})
    assert sp.probe(gen7, rows)["slot_blocks"] == [] and "no slot blocks" in sp.text("g", {"slot_blocks": []})


def test_the_modes_replace_exactly_one_part():
    m = _model()
    mixer = m.blocks[1].mixer
    with torch.no_grad():
        mixer.query.weight.mul_(60)                 # sharp reading, so that the choice among slots matters
        mixer.write.weight.mul_(20)
        u = torch.randn(2, 20, 32)
        normal, own = sp.slot_forward(mixer, u), mixer(u)[0]
        assert torch.allclose(normal, own, atol=1e-5)
        off, uread, uwrite = (sp.slot_forward(mixer, u, mode) for mode in ("off", "uniform_read", "uniform_write"))
    assert float(off.abs().max()) == 0 and not torch.allclose(uread, normal, atol=1e-4) and not torch.allclose(uwrite, normal, atol=1e-4)
    stats = {}
    with torch.no_grad():
        sp.slot_forward(mixer, u, "uniform_write", stats)
    assert abs(stats["address_slots_per_token"] - 8) < 1e-3 and abs(stats["slots_in_use"] - 8) < 1e-3     # every slot holds the same
    assert abs(stats["read_slots_per_query"] - 8) < 1e-2

"""The identity test: constant state is the hard condition; the rest is recorded."""

import torch

from evo.engine import identity as idn
from evo.engine.architecture_factory import build_model

V = 300


def _check(cfg):
    torch.manual_seed(0)
    return idn.check(build_model({"vocab_size": V, **cfg}), V)


def test_recurrent_cores_pass_as_nova_with_a_state_that_does_not_grow():
    gen7 = _check({"d_model": 32, "d_state": 32, "num_layers": 2, "conv_kernel": 5, "forget_bias": 1.125, "learnable_initial_state": False})
    nslot = _check({"arch": "nova8", "d_model": 32, "pattern": "NSNS", "mlp_hidden": 48, "heads": 4, "slots": 8})
    other = _check({"arch": "nova8", "d_model": 32, "pattern": "LHM", "mlp_hidden": 48, "heads": 4})
    for r in (gen7, nslot, other):
        assert r["pass"] and r["class"] == "NOVA" and r["constant_state"] and r["growth_bytes_per_token"] == 0
        assert r["causal"] and r["stepping_exact"] and not r["attends_over_stored_tokens"]
        assert len(set(r["state_bytes_after"].values())) == 1 and r["state_kb"] > 0


def test_a_fixed_window_passes_as_a_hybrid_and_a_growing_cache_fails():
    nsw = _check({"arch": "nova8", "d_model": 32, "pattern": "NSW", "mlp_hidden": 48, "heads": 4, "slots": 8, "window": 16})
    assert nsw["pass"] and nsw["class"] == "NOVA-HYBRID" and nsw["attends_over_stored_tokens"]
    wide = _check({"arch": "nova8", "d_model": 32, "pattern": "NW", "mlp_hidden": 48, "heads": 4, "window": 300})
    sizes = wide["state_bytes_after"]
    assert wide["pass"] and sizes["128"] < sizes["1024"] == sizes["4096"]         # it fills up, then stays
    tf = _check({"arch": "transformer", "d_model": 32, "num_layers": 2, "n_heads": 4})
    assert not tf["pass"] and tf["class"] == "NOT-NOVA" and tf["growth_bytes_per_token"] > 0 and tf["causal"]
    assert "FAIL" in idn.text("tf", tf) and "PASS" in idn.text("nsw", nsw) and "NOVA-HYBRID" in idn.text("nsw", nsw)

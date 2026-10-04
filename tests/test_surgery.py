"""In-place changes of the core's structure keep what the model has learned."""

import torch

from evo.engine.architecture_factory import build_model
from nova import surgery

CFG = {"vocab_size": 16384, "d_model": 48, "d_state": 48, "num_layers": 3, "conv_kernel": 5, "forget_bias": 1.125,
       "learnable_initial_state": False}


def checkpoint():
    torch.manual_seed(1)
    m = build_model(CFG)
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim > 1:
                p.add_(torch.randn_like(p) * 0.05)
    return {"config": dict(CFG), "model_state_dict": m.state_dict(), "optimizer": {"x": 1}, "candidate": "GEN7"}


def logits(ckpt, x):
    m = build_model(ckpt["config"]).eval()
    m.load_state_dict(ckpt["model_state_dict"])
    with torch.no_grad():
        return m(x)[0]


def test_a_new_layer_changes_nothing_until_it_learns():
    ck, x = checkpoint(), torch.randint(12, 16384, (2, 24))
    before = logits(ck, x)
    for pos in (None, 0, 1):
        new = surgery.add_layer(ck, pos)
        assert new["config"]["num_layers"] == 4 and "optimizer" not in new and new["candidate"] == "GEN7"
        assert float((logits(new, x) - before).abs().max()) < 1e-5
        assert surgery.parameters(new) > surgery.parameters(ck)
    assert ck["config"]["num_layers"] == 3 and len([k for k in ck["model_state_dict"] if k.startswith("blocks.3.")]) == 0


def test_the_new_layer_can_learn():
    new = surgery.add_layer(checkpoint())
    m = build_model(new["config"])
    m.load_state_dict(new["model_state_dict"])
    x = torch.randint(12, 16384, (4, 16))
    out = m(x[:, :-1])[0]
    loss = torch.nn.functional.cross_entropy(out.reshape(-1, 16384), x[:, 1:].reshape(-1))
    loss.backward()
    assert float(m.blocks[3].output_proj.weight.grad.abs().sum()) > 0


def test_a_wider_view_is_exact_and_a_narrower_one_keeps_the_nearest_taps():
    ck, x = checkpoint(), torch.randint(12, 16384, (2, 24))
    before = logits(ck, x)
    wide = surgery.set_kernel(ck, 7)
    assert wide["config"]["conv_kernel"] == 7 and float((logits(wide, x) - before).abs().max()) < 1e-5
    narrow = surgery.set_kernel(ck, 3)
    w_old, w_new = ck["model_state_dict"]["blocks.0.local_conv.weight"], narrow["model_state_dict"]["blocks.0.local_conv.weight"]
    assert w_new.shape[-1] == 3 and torch.equal(w_new, w_old[:, :, 2:])
    assert logits(narrow, x).shape == before.shape and surgery.set_kernel(wide, 5)["model_state_dict"]["blocks.0.local_conv.weight"].equal(w_old)


def test_apply_respects_limits_and_records_what_was_done():
    ck = checkpoint()
    new = surgery.apply(ck, {"op": "add_layer"})
    assert new["surgery"] == ["add_layer at 3: 3 -> 4 layers"] and surgery.genome(new)["layers"] == 4
    again = surgery.apply(new, {"op": "kernel", "delta": 2})
    assert again["surgery"][-1] == "kernel 5 -> 7" and len(again["surgery"]) == 2
    assert surgery.apply(ck, {"op": "add_layer"}, max_parameters=surgery.parameters(ck)) is None
    assert surgery.apply(ck, {"op": "kernel", "k": 5}) is None and surgery.apply(ck, {"op": "kernel", "k": 11}) is None
    nine = surgery.set_kernel(ck, 9)
    assert surgery.apply(nine, {"op": "kernel", "delta": 2}) is None
    g = surgery.genome(ck)
    assert g == {"layers": 3, "width": 48, "kernel": 5, "parameters": sum(p.numel() for p in build_model(CFG).parameters())}

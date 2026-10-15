"""Domain packages: the base never changes, a package starts as a no-op, base + package merge into an ordinary core."""

import pytest
import torch

from nova import adapters as A

CFG = {"vocab_size": 300, "arch": "nova8", "d_model": 32, "pattern": "NSN", "mlp_hidden": 48, "heads": 2, "slots": 4}


def core(seed=0):
    from evo.engine.architecture_factory import build_model
    torch.manual_seed(seed)
    return build_model(CFG).eval()


def test_a_fresh_package_changes_nothing_and_only_the_package_trains():
    m = core()
    base = {k: v.clone() for k, v in m.state_dict().items()}
    x = torch.randint(1, 300, (2, 20))
    with torch.no_grad():
        before = m(x)[0]
    names = A.inject(m, rank=4)
    assert names and all(n.endswith(("fc_a", "fc_b", "fc_out", "mixer.out")) for n in names)
    with torch.no_grad():
        assert torch.allclose(m(x)[0], before, atol=1e-6)                       # B starts at zero
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_" in n for n in trainable)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.5)
    m.train()
    for _ in range(3):
        loss = torch.nn.functional.cross_entropy(m(x[:, :-1])[0].reshape(-1, 300), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    merged = A.merged_state_dict(m)
    for k, v in base.items():                                                    # the base itself is untouched
        sd = m.state_dict()
        own = sd[k] if k in sd else sd[k.replace(".weight", ".base.weight").replace(".bias", ".base.bias")]
        assert torch.equal(own.cpu(), v)
    assert set(merged) == set(base)
    fresh = core(1)
    fresh.load_state_dict(merged)                                                # an ordinary core again
    with torch.no_grad():
        assert torch.allclose(fresh(x)[0], m(x)[0], atol=1e-4)
        assert not torch.allclose(fresh(x)[0], before, atol=1e-4)               # and it did learn something


def test_a_stored_package_fits_only_its_own_base(tmp_path):
    m = core()
    fp = A.base_fingerprint(m.state_dict())
    A.inject(m, rank=4)
    assert A.base_fingerprint(m.state_dict()) == fp                              # the frozen base is still the same
    with torch.no_grad():
        for n, p in m.named_parameters():
            if ".lora_b" in n:
                p.add_(0.01)
    path = A.save_package(tmp_path / "math.pt", A.package_state(m), fp, {"rank": 4, "name": "matematika"})
    pkg = A.read_package(path)
    x = torch.randint(1, 300, (1, 12))
    other = core()
    A.apply_package(other, pkg, base_state=core().state_dict())
    with torch.no_grad():
        assert torch.allclose(other(x)[0], m(x)[0], atol=1e-5)
    with pytest.raises(ValueError, match="another base"):
        A.apply_package(core(7), pkg, base_state=core(7).state_dict())
    assert '"matematika"' in A.describe(pkg)


def test_fused_layers_still_see_the_package():
    """core8 joins two linear layers into one product for speed; with a package the delta must not be skipped."""
    m = core()
    A.inject(m, rank=4)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if ".lora_b" in n:
                p.normal_(0, 0.05)
    x = torch.randint(1, 300, (8, 40))                                         # > 64 tokens: the fused path
    fresh = core(1)
    fresh.load_state_dict(A.merged_state_dict(m))
    with torch.no_grad():
        assert torch.allclose(fresh(x)[0], m(x)[0], atol=1e-4)
    assert m.blocks[0].fc_a.bias is m.blocks[0].fc_a.base.bias

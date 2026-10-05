"""Running text: the four ways of reading are measured on the same tokens and agree where they must."""

import numpy as np
import torch

from evo.engine import long_context as lc
from evo.engine.architecture_factory import build_model

BASE = {"vocab_size": 90, "d_model": 32}


def _loud(m):
    with torch.no_grad():
        for blk in m.blocks:
            for name in ("fc_out", "fc2", "proj"):
                if hasattr(blk, name):
                    getattr(blk, name).weight.mul_(20)
            if hasattr(blk, "mixer"):
                blk.mixer.out.weight.mul_(20)
    return m.eval()


def test_recurrent_core_carried_equals_two_rows_when_there_are_two_rows():
    torch.manual_seed(0)
    m = _loud(build_model({**BASE, "arch": "nova8", "pattern": "LSW", "mlp_hidden": 48, "heads": 4, "window": 6}))
    rows = np.random.default_rng(0).integers(12, 90, size=(2, 32)).astype(np.int32)
    r = lc.measure(m, rows, every=8)
    assert set(r) == {"row", "two_rows", "window", "carried"} and r["row"].shape == (1, 4)
    assert np.allclose(r["carried"], r["two_rows"], atol=1e-4)            # the same text was read before the row
    assert not np.allclose(r["row"], r["carried"], atol=1e-3)             # and it matters
    # "row" by hand
    x = torch.from_numpy(rows[1:2]).long()
    with torch.no_grad():
        logits = m(x[:, :-1])[0]
    want = [float(torch.nn.functional.cross_entropy(logits[0, p - 1:p], x[0, p:p + 1])) for p in (4, 12, 20, 28)]
    assert np.allclose(r["row"][0], want, atol=1e-4)


def test_transformer_has_no_carried_reading_and_the_window_is_the_31_tokens_before():
    torch.manual_seed(1)
    m = _loud(build_model({**BASE, "arch": "transformer", "num_layers": 2, "n_heads": 4}))
    rows = np.random.default_rng(1).integers(12, 90, size=(4, 32)).astype(np.int32)
    r = lc.measure(m, rows, every=16)
    assert set(r) == {"row", "two_rows", "window"} and r["window"].shape == (3, 2)
    flat = torch.from_numpy(rows.reshape(-1)).long()
    j = 2 * 32 + 8                                                         # row 2, position 8
    with torch.no_grad():
        logit = m(flat[j - 31:j][None])[0][0, -1]
    assert abs(r["window"][1, 0] - float(torch.nn.functional.cross_entropy(logit[None], flat[j:j + 1]))) < 1e-4
    s = lc.summary(r)
    assert "window_vs_row_percent" in s and "carried" not in s
    assert "sliding window" in lc.text({"m": s})

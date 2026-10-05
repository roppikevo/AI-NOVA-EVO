"""Training on running text: consecutive rows with the state carried from row to row."""

import json

import numpy as np
import torch

from evo.engine.architecture_factory import build_model
from evo.engine.long_train import detach_states, evaluate, evaluate_carried, long_train

CFG = {"arch": "nova8", "vocab_size": 60, "d_model": 32, "pattern": "LSW", "mlp_hidden": 48, "heads": 4, "window": 6}


class Spy:
    """A bulk corpus that remembers which rows were asked for."""

    def __init__(self, rows):
        self.rows, self.asked = rows, []
        self.shape, self.dtype = rows.shape, rows.dtype

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        self.asked.append(np.array(idx))
        return self.rows[idx]


def running_text(n_rows, seq=16, seed=0):
    """One long text with a period of 37 tokens, cut into rows: the start of a row is only predictable from the row before."""
    phrase = np.random.default_rng(seed).integers(12, 60, size=37)
    return np.resize(phrase, n_rows * seq).reshape(n_rows, seq).astype(np.int32)


def test_streams_read_consecutive_rows_and_other_material_fills_the_rest(tmp_path):
    torch.manual_seed(0)
    bulk = Spy(running_text(400))
    train = np.random.default_rng(1).integers(12, 60, size=(40, 16)).astype(np.int32)
    rep = long_train(build_model(CFG), train, train[:8], steps=30, out_dir=tmp_path, meta={"tag": "t"}, batch_size=4, warmup=1,
                     eval_every=15, save_every=100, bulk=bulk, bulk_frac=0.9, code=train[:3], code_frac=0.05, carry=4,
                     extra_val=running_text(64, seed=0), progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None)
    assert rep["steps_done"] == 30
    asked = bulk.asked
    assert len(asked) >= 16 and len(asked) < 30                      # most steps read the web text, some the rest
    runs = [asked[i:i + 4] for i in range(0, len(asked) - len(asked) % 4, 4)]
    assert all((run[k] == run[0] + k).all() for run in runs for k in range(4))   # four consecutive rows per stream
    rows = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert "web_carried" in rows[-1] and np.isfinite(rows[-1]["web_carried"])


def test_mixed_batches_hold_running_text_and_fresh_rows_in_every_step(tmp_path):
    torch.manual_seed(0)
    bulk = Spy(running_text(400))
    train = np.random.default_rng(1).integers(12, 60, size=(40, 16)).astype(np.int32)
    rep = long_train(build_model(CFG), train, train[:8], steps=24, out_dir=tmp_path, meta={"tag": "t"}, batch_size=8, warmup=1,
                     eval_every=12, save_every=100, bulk=bulk, bulk_frac=0.5, carry=4, carry_share=0.5,
                     extra_val=running_text(64, seed=0), progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None)
    assert rep["steps_done"] == 24
    asked = bulk.asked
    assert len(asked) == 24 and all(len(a) == 4 for a in asked)      # every step: half the batch is running text, the rest is the dataset
    runs = [asked[i:i + 4] for i in range(0, 24, 4)]
    assert all((run[k] == run[0] + k).all() for run in runs for k in range(4))
    assert any((runs[0][0] != r[0]).any() for r in runs[1:])           # a new stream starts somewhere else
    rows = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert np.isfinite(rows[-1]["web_carried"]) and np.isfinite(rows[-1]["train_loss"])


def test_mixed_batches_teach_the_carried_state_and_keep_the_cold_start(tmp_path):
    torch.manual_seed(0)
    bulk = documents(400, seed=1)
    model = build_model({**CFG, "pattern": "NS"})
    long_train(model, bulk[:40], bulk[:8], steps=400, out_dir=tmp_path, meta={"tag": "t"}, batch_size=16, lr=3e-3, warmup=10,
               eval_every=1000, save_every=1000, bulk=bulk, bulk_frac=1.0, carry=4, carry_share=0.5,
               progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None)
    test = documents(30, seed=2)
    with torch.no_grad():
        plain, carried = _own_token(model, test, carried=False), _own_token(model, test, carried=True)
    assert plain > 2.0 and carried < 0.5 * plain


def test_carried_evaluation_counts_the_same_tokens_and_uses_the_state():
    torch.manual_seed(0)
    m = build_model(CFG).eval()
    with torch.no_grad():
        for blk in m.blocks:
            blk.fc_out.weight.mul_(20)
            blk.mixer.out.weight.mul_(20)
    rows = running_text(64)
    plain = evaluate(m, rows, "cpu", batch_size=8)
    carried = evaluate_carried(m, rows, "cpu", streams=4)
    assert np.isfinite(carried) and abs(carried - plain) > 1e-6       # the state from the row before changes the predictions
    # by hand for one stream: rows 0..15 in order, the state handed on
    total, n, states = 0.0, 0, None
    with torch.no_grad():
        for k in range(16):
            b = torch.from_numpy(rows[k][None]).long()
            logits, states = m(b, states)
            total += float(torch.nn.functional.cross_entropy(logits[0, :-1], b[0, 1:], reduction="sum"))
            n += 15
    one = evaluate_carried(m, rows[:16], "cpu", streams=1)
    assert abs(one - total / n) < 1e-4


def documents(n_docs, seed, rows_per_doc=4, seq=16):
    """Texts of four rows: every row starts with three filler tokens, the rest is the text's own token.
    The first own token of a row cannot be guessed from the row itself - only from the rows before it."""
    ids = np.random.default_rng(seed).integers(0, 20, size=n_docs)
    rows = np.full((n_docs * rows_per_doc, seq), 5, dtype=np.int32)
    rows[:, 3:] = np.repeat(20 + ids, rows_per_doc)[:, None]
    return rows


def test_a_core_trained_on_running_text_learns_to_use_what_it_carries(tmp_path):
    torch.manual_seed(0)
    bulk = documents(400, seed=1)
    model = build_model({**CFG, "pattern": "LL"})
    long_train(model, bulk[:40], bulk[:8], steps=300, out_dir=tmp_path, meta={"tag": "t"}, batch_size=16, lr=3e-3, warmup=10,
               eval_every=1000, save_every=1000, bulk=bulk, bulk_frac=1.0, carry=4,
               progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None)
    test = documents(30, seed=2)
    with torch.no_grad():
        plain, carried = _own_token(model, test, carried=False), _own_token(model, test, carried=True)
    assert plain > 2.0 and carried < 0.5 * plain          # without the state it is a guess among 20; with it, it is known


def _own_token(model, rows, carried):
    """Mean loss on the first own token of rows 2-4 of every text."""
    model.eval()
    total, n, states = 0.0, 0, None
    for k in range(len(rows)):
        if k % 4 == 0:
            states = None                                  # a new text
        b = torch.from_numpy(rows[k][None]).long()
        logits, new = model(b, states if carried else None)
        if k % 4:
            total += float(torch.nn.functional.cross_entropy(logits[0, 2:3], b[0, 3:4], reduction="sum"))
            n += 1
        states = detach_states(new)
    return total / n

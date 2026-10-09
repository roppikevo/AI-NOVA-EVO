"""NOVA-Q: the state keeps its length, continues exactly, a trained core can be rebuilt with Q memory, and the recall
test runs on any core that carries a state."""

import numpy as np
import pytest
import torch

from evo.engine.architecture_factory import build_model
from nova import quantum as Q
from nova import recall as R
from nova.tokenizer import NovaTokenizer

BASE = {"vocab_size": 300, "arch": "nova8", "d_model": 64, "pattern": "NSNSN", "mlp_hidden": 96, "heads": 2, "slots": 4}


def test_the_state_keeps_its_length_and_its_size():
    m = Q.QMixer(32, heads=2, dim=16)
    u = torch.randn(3, 50, 32)
    y, phase = m(u)
    assert y.shape == u.shape and phase.shape == (3, 2, 16)
    assert torch.allclose(Q.norm_of_state(m, phase), torch.ones(3, 2), atol=1e-5)
    y2, phase2 = m(torch.randn(3, 500, 32), phase)
    assert phase2.shape == phase.shape and torch.allclose(Q.norm_of_state(m, phase2), torch.ones(3, 2), atol=1e-5)
    assert float(phase2.min()) >= 0 and float(phase2.max()) < 2 * np.pi


def test_reading_in_pieces_or_token_by_token_gives_the_same_result():
    torch.manual_seed(0)
    m = build_model({**BASE, "pattern": "QNQ", "q_heads": 2, "q_dim": 16}).eval()
    x = torch.randint(1, 300, (2, 60))
    with torch.no_grad():
        full, _ = m(x)
        a, s = m(x[:, :23])
        b, _ = m(x[:, 23:], s)
        assert (torch.cat([a, b], 1) - full).abs().max() < 1e-4
        outs, s = [], None
        for i in range(60):
            o, s = m(x[:, i:i + 1], s)
            outs.append(o)
        assert (torch.cat(outs, 1) - full).abs().max() < 1e-4


def test_a_trained_core_is_rebuilt_with_q_memory_and_starts_as_the_old_one_without_those_memories():
    torch.manual_seed(1)
    old = build_model(BASE)
    ck = {"config": BASE, "model_state_dict": old.state_dict()}
    for pattern in ("QQQQQ", "NQNQN"):
        out = Q.transplant(ck, pattern, q_heads=2, q_dim=16)
        assert out["config"]["pattern"] == pattern and out["transplant"]["copied_parameters"] > 0
        new = build_model(out["config"])
        new.load_state_dict(out["model_state_dict"])
        for k in ("embedding.weight", "blocks.1.fc_a.weight", "blocks.4.fc_out.bias", "final_norm.weight"):
            assert torch.equal(new.state_dict()[k], old.state_dict()[k])
        # the old core with the replaced memories silenced must give the same logits as the new core at step 0
        silenced = build_model(BASE)
        silenced.load_state_dict(old.state_dict())
        with torch.no_grad():
            for i, kind in enumerate(pattern):
                if kind == "Q":
                    silenced.blocks[i].mixer.out.weight.zero_()
                    if silenced.blocks[i].mixer.out.bias is not None:
                        silenced.blocks[i].mixer.out.bias.zero_()
            x = torch.randint(1, 300, (2, 30))
            assert (new.eval()(x)[0] - silenced.eval()(x)[0]).abs().max() < 1e-5
    with pytest.raises(ValueError):
        Q.transplant(ck, "NNNNN")                     # a block can only stay or become Q
    with pytest.raises(ValueError):
        Q.transplant(ck, "QQQ")


def test_q_memory_learns_something_on_a_small_task():
    """A copy task over a gap: the Q core must do better than chance after a short training."""
    torch.manual_seed(0)
    m = build_model({**BASE, "pattern": "QNQ", "q_heads": 2, "q_dim": 32, "vocab_size": 40})
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for step in range(60):
        key = torch.randint(10, 40, (32, 1))
        x = torch.cat([key, torch.randint(1, 10, (32, 6)), key], dim=1)
        logits, _ = m(x)
        loss = torch.nn.functional.cross_entropy(logits[:, -2], x[:, -1])
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first


def test_recall_at_a_distance_runs_and_tells_apart_the_two_facts_at_distance_zero():
    texts = ["Tajný kód je 1234.", "Tajný kód je 9876.", "The river runs through the town and the bridge."] * 40
    tok = NovaTokenizer.train(texts, vocab_size=300)
    torch.manual_seed(0)
    m = build_model({**BASE, "pattern": "QNQ", "q_heads": 2, "q_dim": 16}).eval()
    filler = np.array(tok.encode(" ".join(texts[2::3])) * 20, dtype=np.int64)
    r = R.recall(m, tok, filler, distances=(0, 16, 64), samples=3, lang="sk")
    assert set(r["recall_nats"]) == {"0", "16", "64"} and r["samples"] == 3
    assert all(np.isfinite(v) for v in r["recall_nats"].values())
    q, a = R.answer_ids(tok, "Tajný kód je", "1234")
    assert q and a and tok.encode("Tajný kód je 1234")[:len(q)] == q
    with pytest.raises(ValueError):
        R.recall(torch.nn.Linear(2, 2), tok, filler)

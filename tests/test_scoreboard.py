"""Scoreboard: per-language loss, weakest language, autopilot tuning (offline)."""

import json

import numpy as np
import torch
import torch.nn as nn

from evo.engine import autopilot as ap
from evo.engine import scoreboard as sb


class Tiny(nn.Module):
    def __init__(self, v=40):
        super().__init__()
        self.e, self.h = nn.Embedding(v, 8), nn.Linear(8, v)

    def forward(self, x, states=None):
        return self.h(self.e(x)), None


def test_language_of_tokens_follows_tags_and_eos():
    seq = np.array([[20, 21, 3, 4, 22, 23, 3, 8, 24, 25], [26, 27, 3, 20, 20, 5, 21, 21, 21, 21]])
    assert sb.language_of_tokens(seq).tolist() == [[0, 0, 0, 4, 4, 4, 4, 8, 8, 8],
                                                   [8, 8, 8, 0, 0, 5, 5, 5, 5, 5]]  # carries over rows


def test_per_language_loss_splits_tokens():
    rng = np.random.default_rng(0)
    val = rng.integers(10, 40, size=(12, 16)).astype(np.int32)
    val[:6, 0], val[6:, 0] = 4, 8
    r = sb.per_language_loss(Tiny(), val, "cpu", batch_size=4)
    assert set(r["lang"]) == {"sk", "py"} and r["lang_tokens"]["sk"] == 6 * 15
    assert abs(r["val"] - (r["lang"]["sk"] + r["lang"]["py"]) / 2) < 0.05


def test_weakest_language_is_the_one_that_fell_behind():
    hist = [{"lang": {"sk": 3.0, "en": 3.0}}]
    assert sb.weakest_language({"lang": {"sk": 3.2, "en": 2.9}}, hist) == "sk"
    assert sb.weakest_language({"lang": {"sk": 2.9, "en": 2.9}}, hist) is None


def test_autopilot_tunes_training_from_scoreboard(tmp_path):
    board = tmp_path / "b.jsonl"
    cmd = ["python", "-m", "evo.engine.long_train", "--code-frac", "0.05"]
    assert ap.tune("long_train", cmd, board) == cmd
    board.write_text(json.dumps({"code": {"pass1": 0.1}, "weakest": "pl"}) + "\n")
    tuned = ap.tune("long_train", cmd, board)
    assert tuned[tuned.index("--code-frac") + 1] == "0.10" and tuned[-2:] == ["--boost-lang", "pl"]
    board.write_text(json.dumps({"code": {"pass1": 0.9}, "weakest": "py"}) + "\n")
    tuned = ap.tune("long_train", cmd, board)
    assert tuned[tuned.index("--code-frac") + 1] == "0.03" and "--boost-lang" not in tuned
    assert ap.tune("teachers", cmd, board) == cmd


def test_join_sequences():
    from evo.engine.long_train import join_sequences

    a = np.arange(5 * 4).reshape(5, 4)
    j = join_sequences(a, 2)
    assert j.shape == (2, 8) and j[0].tolist() == list(range(8))


def test_ab_test_decision_and_override(tmp_path, monkeypatch):
    from evo.engine import ab_test

    assert ab_test.decide({"best_val": 3.50}, {"best_val": 3.49}, 0.004) == "B"
    assert ab_test.decide({"best_val": 3.50}, {"best_val": 3.498}, 0.004) == "A"
    rep = ab_test.parse_report('noise\n=== LONG TRAIN REPORT ===\n{"best_val": 3.4, "val_start": 3.5}')
    assert rep["best_val"] == 3.4
    ov = tmp_path / "ov.json"
    ab_test.apply_override("long_train", {"--seq-mult": "2", "--batch-size": "32"}, ov)
    monkeypatch.setattr(ap, "OVERRIDES", ov)
    cmd = ap.tune("long_train", ["x", "--batch-size", "64"], tmp_path / "none.jsonl")
    assert cmd == ["x", "--batch-size", "32", "--seq-mult", "2"]


def test_factorized_embedding_keeps_size_and_shapes():
    from evo.engine.architecture_factory import build_model

    base = {"vocab_size": 16384, "d_model": 384, "d_state": 384, "num_layers": 6, "conv_kernel": 5,
            "forget_bias": 1.125, "learnable_initial_state": False}
    a = build_model(base)
    b = build_model({**base, "d_embed": 128, "num_layers": 12})
    na, nb = (sum(p.numel() for p in m.parameters()) for m in (a, b))
    assert nb <= na and a.embed_in is None and b.embed_in is not None
    x = torch.randint(4, 16384, (2, 12))
    assert b(x)[0].shape == (2, 12, 16384)
    assert set(a.state_dict()) == {k for k in a.state_dict() if "embed_in" not in k}  # old checkpoints still load


def test_tournament_ranking_and_variants_fit_the_budget():
    from evo.engine import arch_tournament as at
    from evo.engine.architecture_factory import build_model

    base = {"vocab_size": 16384, "d_model": 384, "d_state": 384, "num_layers": 6, "conv_kernel": 5,
            "forget_bias": 1.125, "learnable_initial_state": False}
    size_now = sum(p.numel() for p in build_model(base).parameters())
    for name, ov in at.VARIANTS.items():
        assert sum(p.numel() for p in build_model({**base, **ov}).parameters()) <= size_now, name
    r = at.ranking([{"variant": "a", "best_val": 3.6}, {"variant": "b", "best_val": 3.5}, {"variant": "c", "best_val": None}])
    assert [x["variant"] for x in r] == ["b", "a"]


def test_weakest_ignores_rows_of_an_older_method():
    hist = [{"lang": {"en": 2.1, "sk": 2.7}}]                       # old method, no "method" field
    row = {"method": 2, "lang": {"en": 4.19, "sk": 3.62}}
    assert sb.weakest_language(row, hist) is None


def test_val_floor_caps_cumulative_forgetting(tmp_path):
    from evo.learning.self_correction import val_floor

    f = tmp_path / "floor.json"
    assert val_floor(3.45, f) == 3.45
    assert val_floor(3.48, f) == 3.45      # later, worse start: floor stays at the best
    assert val_floor(3.40, f) == 3.40


def test_release_is_complete_and_never_overwritten(tmp_path):
    import pytest
    from evo.engine import release

    assert release.pick([{"val": 3.4, "creator": -3.0, "checkpoint": "a"},
                         {"val": 3.5, "creator": -0.01, "checkpoint": "b"}])["checkpoint"] == "b"
    tokp = tmp_path / "tokenizer.json"
    tokp.write_text("{}")
    ck = {"config": {"d_model": 8}, "candidate": "X", "optimizer": {"big": 1},
          "model_state_dict": {"w": torch.randn(4, 4), "n": torch.tensor([1, 2])}}
    out = tmp_path / "rel" / "NOVA-test"
    release.write_release(out, ck, "src.pt", {"val": 3.4}, tokp, {"dataset": "d"})
    a = torch.load(out / "nova_model.pt", weights_only=False)
    b = torch.load(out / "nova_model_fp32.pt", weights_only=False)
    assert a["model_state_dict"]["w"].dtype == torch.float16 and b["model_state_dict"]["w"].dtype == torch.float32
    assert "optimizer" not in a and json.loads((out / "MODEL.json").read_text())["parameters"] == 18
    sums = (out / "SHA256SUMS").read_text()
    assert "nova_model.pt" in sums and "tokenizer.json" in sums
    with pytest.raises(FileExistsError):
        release.write_release(out, ck, "src.pt", {"val": 3.4}, tokp, {})


def test_release_names_a_generation8_core_by_its_pattern_and_counts_the_token_table_once(tmp_path):
    from evo.engine import release
    from evo.engine.architecture_factory import build_model

    cfg = {"arch": "nova8", "vocab_size": 64, "d_model": 16, "pattern": "NS", "heads": 2, "mlp_hidden": 32, "slots": 4, "max_seq_len": 16}
    model = build_model(cfg)
    ck = {"config": cfg, "candidate": "GEN7-CORE-001", "model_state_dict": model.state_dict()}
    tokp = tmp_path / "tokenizer.json"
    tokp.write_text("{}")
    out = tmp_path / "rel" / "NOVA8-test"
    release.write_release(out, ck, "src.pt", {"val": 3.0}, tokp, {})
    info = json.loads((out / "MODEL.json").read_text())
    assert info["core"] == "nova8 NS"
    assert info["parameters"] == sum(p.numel() for p in model.parameters()) < sum(v.numel() for v in model.state_dict().values())

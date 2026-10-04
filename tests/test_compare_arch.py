"""NOVA core vs transformer: the yardstick model and the comparison logic (no training, no server)."""

import json

import numpy as np
import torch

from evo.engine import compare_arch as ca
from evo.engine.architecture_factory import build_model

BASE = {"vocab_size": 16384, "d_model": 384, "d_state": 384, "num_layers": 6, "conv_kernel": 5, "forget_bias": 1.125,
        "learnable_initial_state": False}


def n_params(override):
    return sum(p.numel() for p in build_model({**BASE, **override}).parameters())


def test_transformer_sizes_match_the_nova_cores():
    for g in ("10M", "24M"):
        nova = n_params(ca.NOVA[g])
        for shape in ca.TRANSFORMER[g].values():
            tf = n_params({"arch": "transformer", **shape})
            assert abs(tf - nova) / nova < 0.03, (g, shape, tf, nova)


def small():
    torch.manual_seed(0)
    return build_model({**BASE, "arch": "transformer", "d_model": 64, "num_layers": 2, "n_heads": 4}).eval()


def test_transformer_cache_gives_the_same_logits_as_a_full_pass():
    m, x = small(), torch.randint(12, 16384, (2, 18))
    full, _ = m(x)
    a, st = m(x[:, :6])
    b, st = m(x[:, 6:10], st)            # several new tokens at once
    outs = [a, b]
    for t in range(10, 18):
        o, st = m(x[:, t:t + 1], st)
        outs.append(o)
    assert float((full - torch.cat(outs, dim=1)).abs().max()) < 1e-4


def test_transformer_is_causal_and_uses_word_order():
    m, x = small(), torch.randint(12, 16384, (1, 16))
    full, _ = m(x)
    y = x.clone()
    y[0, 12] = 13
    assert float((m(y)[0][:, :12] - full[:, :12]).abs().max()) == 0.0        # the future does not leak back
    z = x.clone()
    z[0, [2, 3]] = z[0, [3, 2]]
    assert float((m(z)[0][:, -1] - full[:, -1]).abs().max()) > 1e-4         # positions matter


def test_cpu_speed_works_for_both_architectures():
    for cfg in ({"d_model": 32, "d_state": 32, "num_layers": 2}, {"arch": "transformer", "d_model": 32, "num_layers": 2, "n_heads": 2}):
        s = ca.cpu_speed(build_model({**BASE, **cfg}), 16384, prompt_len=16, gen=4, threads=1, repeats=1)
        assert s["read_tokens_per_s"] > 0 and s["write_tokens_per_s"] > 0 and s["state_kb_after_writing"] > 0


def test_plan_three_transformer_attempts_per_size():
    first = ca.first_round(["10M", "24M"])
    assert [v["name"] for v in first] == ["nova-10M", "tf-10M-a", "tf-10M-b", "nova-24M", "tf-24M-a", "tf-24M-b"]
    assert all(v["lr"] == ca.LR for v in first) and first[0]["override"] == {}
    results = {v["name"]: {**v, "loss": {"dataset": l}} for v, l in zip(first, [3.7, 3.9, 3.8, 3.4, 3.5, 3.6])}
    second = ca.second_round(results, ["10M", "24M"])
    assert [v["name"] for v in second] == ["tf-10M-lr1e-3", "tf-24M-lr1e-3"]
    assert second[0]["override"]["d_model"] == 288 and second[1]["override"]["d_model"] == 512 and second[0]["lr"] == "1e-3"


def test_train_builds_the_command_and_reads_the_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "OUT", tmp_path)
    seen = {}

    class P:
        returncode, stderr = 0, ""
        stdout = "step 3000 val 4.1 best 4.1\n=== LONG TRAIN REPORT ===\n" + json.dumps(
            {"params": 9_800_000, "hours": 0.5, "steps_done": 18000, "best_val": 3.8, "tokens_seen": 146_304_000})

    def runner(cmd, **kw):
        seen["cmd"] = cmd
        (tmp_path / "tf-10M-a.pt").write_bytes(b"x")
        return P()

    v = ca.first_round(["10M"])[1]
    r = ca.train(v, 18000, 3.0, runner=runner)
    cmd = seen["cmd"]
    assert "--from-scratch" in cmd and "--no-activate" in cmd and cmd[cmd.index("--lr") + 1] == "3e-4"
    assert json.loads(cmd[cmd.index("--config-override") + 1])["arch"] == "transformer"
    assert r["train_tokens_per_s"] == 81280 and r["curve"] == ["step 3000 val 4.1"]


def test_verdict_names_the_winner_per_size_and_text_is_complete():
    rng = np.random.default_rng(0)
    hard = rng.normal(0, 0.4, size=500)
    seq, results = {}, {}
    for name, group, arch, loss in [("nova-10M", "10M", "nova", 3.70), ("tf-10M-a", "10M", "transformer", 3.80),
                                    ("tf-10M-b", "10M", "transformer", 3.75), ("nova-24M", "24M", "nova", 3.50),
                                    ("tf-24M-a", "24M", "transformer", 3.44)]:
        for s in ("dataset", "web"):
            seq[f"{s}/{name}"] = (loss + hard + rng.normal(0, 0.03, size=500)).astype(np.float32)
        results[name] = {"name": name, "group": group, "arch": arch, "override": {}, "lr": "3e-4", "params": 9_800_000,
                         "loss": {s: round(float(seq[f"{s}/{name}"].mean()), 4) for s in ("dataset", "web")},
                         "code_solved": 30, "code_tasks": 79, "train_tokens_per_s": 50000,
                         "cpu": {"read_tokens_per_s": 5000, "write_tokens_per_s": 400.0 if arch == "nova" else 200.0,
                                 "state_kb_after_writing": 10.0},
                         "cpu_by_context": {"1024": {"write_tokens_per_s": 80.0, "state_kb_after_writing": 20.0}}}
    results["tf-10M-x"] = {"name": "tf-10M-x", "group": "10M", "arch": "transformer", "error": "CUDA out of memory"}
    v = ca.verdicts(results, seq, ["10M", "24M"])
    assert v["10M"]["transformer"] == "tf-10M-b" and v["10M"]["transformer_attempts"] == 2
    assert v["10M"]["sets"]["dataset"]["winner"] == "NOVA" and v["24M"]["sets"]["web"]["winner"] == "transformer"
    assert v["10M"]["cpu_write_speedup"] == 2.0
    t = ca.text({"date": "2026-10-05", "steps": 18000, "results": results, "verdicts": v})
    assert "NOVA CORE vs TRANSFORMER" in t and "FAILED" in t and "-> NOVA" in t and "-> TRANSFORMER" in t
    assert "CPU after 1024 tokens" in t

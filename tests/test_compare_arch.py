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


def test_extra_learning_rates_for_both_sides_and_the_best_nova_attempt_is_compared():
    rng = np.random.default_rng(1)
    hard = rng.normal(0, 0.4, size=400)
    seq, results = {}, {}

    def add(name, arch, loss, lr, override):
        for s in ("dataset", "web"):
            seq[f"{s}/{name}"] = (loss + hard + rng.normal(0, 0.02, size=400)).astype(np.float32)
        results[name] = {"name": name, "group": "24M", "arch": arch, "override": override, "lr": lr, "params": 24_000_000,
                         "loss": {s: round(float(seq[f"{s}/{name}"].mean()), 4) for s in ("dataset", "web")},
                         "code_solved": 50, "code_tasks": 79, "train_tokens_per_s": 50000,
                         "cpu": {"read_tokens_per_s": 3000, "write_tokens_per_s": 200.0, "state_kb_after_writing": 100.0}}

    add("nova-24M", "nova", 3.33, "3e-4", ca.NOVA["24M"])
    add("tf-24M-a", "transformer", 3.06, "3e-4", {"arch": "transformer", "d_model": 512})
    add("tf-24M-b", "transformer", 3.05, "3e-4", {"arch": "transformer", "d_model": 448})
    add("tf-24M-lr1e-3", "transformer", 2.98, "1e-3", {"arch": "transformer", "d_model": 448})
    plan = ca.extra_round(results, ["24M"], ["3e-4", "1e-3", "2e-3"], ["2e-3"])
    assert [(v["name"], v["lr"]) for v in plan] == [("nova-24M-lr1e-3", "1e-3"), ("nova-24M-lr2e-3", "2e-3"), ("tf-24M-lr2e-3", "2e-3")]
    assert plan[0]["override"] == ca.NOVA["24M"] and plan[2]["override"]["d_model"] == 448      # the better shape at the common rate
    add("nova-24M-lr1e-3", "nova", 3.20, "1e-3", ca.NOVA["24M"])
    v = ca.verdicts(results, seq, ["24M"])["24M"]
    assert v["nova"] == "nova-24M-lr1e-3" and v["nova_attempts"] == 2 and v["transformer_attempts"] == 3
    t = ca.text({"date": "2026-10-04", "steps": 18000, "results": results, "verdicts": {"24M": v}})
    assert "best NOVA nova-24M-lr1e-3 (of 2 attempts) vs best transformer tf-24M-lr1e-3 (of 3 attempts)" in t


def test_generation_8_candidates_plan_sizes_and_speed_measurement():
    plan = ca.candidate_round(["24M"], ["mlp", "lru", "nothing"], "1e-3")
    assert [v["name"] for v in plan] == ["n8-mlp-24M", "n8-lru-24M"] and plan[0]["arch"] == "nova8"
    assert ca.candidate_round(["24M"], ["lru"], "2e-3")[0]["name"] == "n8-lru-24M-lr2e-3"
    target = n_params(ca.NOVA["24M"])
    for name, shape in ca.CANDIDATES["24M"].items():
        assert abs(n_params(shape) / target - 1) < 0.03, name
    m = build_model({**BASE, "arch": "nova8", "d_model": 32, "pattern": "LSW", "mlp_hidden": 48, "heads": 2, "window": 8}).eval()
    short = ca.cpu_speed(m, BASE["vocab_size"], prompt_len=20, gen=4, threads=1, repeats=1)
    long = ca.cpu_speed(m, BASE["vocab_size"], prompt_len=300, gen=4, threads=1, repeats=1)
    assert short["write_tokens_per_s"] > 0 and short["state_kb_after_writing"] == long["state_kb_after_writing"] > 0


def test_candidates_are_compared_with_the_transformer_and_with_generation_7():
    rng = np.random.default_rng(2)
    hard = rng.normal(0, 0.4, size=400)
    seq, results = {}, {}
    for name, arch, loss in [("nova-24M", "nova", 3.33), ("tf-24M-b", "transformer", 3.05), ("n8-lru-24M", "nova8", 3.02),
                             ("n8-mlp-24M", "nova8", 3.20)]:
        for s in ("dataset", "web"):
            seq[f"{s}/{name}"] = (loss + hard + rng.normal(0, 0.02, size=400)).astype(np.float32)
        results[name] = {"name": name, "group": "24M", "arch": arch, "override": {}, "lr": "1e-3", "params": 24_000_000,
                         "loss": {s: round(float(seq[f"{s}/{name}"].mean()), 4) for s in ("dataset", "web")},
                         "code_solved": 50, "code_tasks": 79, "train_tokens_per_s": 60000,
                         "cpu": {"read_tokens_per_s": 3000, "write_tokens_per_s": 200.0, "state_kb_after_writing": 100.0},
                         "cpu_by_context": {"4096": {"write_tokens_per_s": 180.0, "state_kb_after_writing": 100.0}}}
    v = ca.verdicts(results, seq, ["24M"])["24M"]
    assert list(v["candidates"]) == ["n8-lru-24M", "n8-mlp-24M"]                      # best first
    assert v["candidates"]["n8-lru-24M"]["vs_transformer"]["dataset"]["result"] == "better"
    assert v["candidates"]["n8-mlp-24M"]["vs_transformer"]["dataset"]["result"] == "worse"
    assert v["candidates"]["n8-mlp-24M"]["vs_gen7"]["web"]["result"] == "better"
    t = ca.text({"date": "2026-10-05", "steps": 18000, "results": results, "verdicts": {"24M": v}})
    assert "candidate n8-lru-24M" in t and "vs generation 7: better" in t and "after 4096 tokens 180.0 tok/s with 100.0 kB" in t


def test_candidates_can_train_compiled(tmp_path, monkeypatch):
    v = ca.candidate_round(["24M"], ["slot"], "1e-3", compiled=True)[0]
    assert v["compile"] is True and "compile" not in ca.candidate_round(["24M"], ["slot"], "1e-3")[0]
    monkeypatch.setattr(ca, "OUT", tmp_path)
    seen = {}

    def runner(cmd, **kw):
        seen["cmd"] = cmd
        (tmp_path / f"{v['name']}.pt").write_bytes(b"x")
        report = {"params": 1, "hours": 0.5, "steps_done": 10, "best_val": 3.0, "tokens_seen": 1000}
        return type("P", (), {"returncode": 0, "stdout": "=== LONG TRAIN REPORT ===\n" + json.dumps(report), "stderr": ""})()

    ca.train(v, 10, 1.0, runner=runner)
    assert "--compile" in seen["cmd"]


def test_candidates_on_running_text_and_the_carried_loss():
    v = ca.candidate_round(["24M"], ["slot"], "1e-3", carry=8)[0]
    assert v["name"] == "n8-slot-24M-carry8" and v["carry"] == 8
    m = ca.candidate_round(["24M"], ["slot"], "1e-3", carry=8, share=0.35)[0]
    assert m["name"] == "n8-slot-24M-carry8mix" and m["carry"] == 8 and m["carry_share"] == 0.35
    rows = np.random.default_rng(0).integers(12, BASE["vocab_size"], size=(64, 16)).astype(np.int32)
    n8 = build_model({**BASE, "arch": "nova8", "d_model": 32, "pattern": "LS", "mlp_hidden": 48, "heads": 2}).eval()
    gen7 = build_model({**BASE, "d_model": 32, "d_state": 32, "num_layers": 2}).eval()
    assert ca.carried_loss(n8, rows, "cpu") > 0 and ca.carried_loss(gen7, rows, "cpu") > 0 and not n8.training
    assert ca.carried_loss(small(), rows, "cpu") is None


def test_the_best_candidates_are_picked_for_the_run_on_running_text():
    def row(name, d, w, **kw):
        return {"name": name, "group": "24M", "arch": "nova8", "loss": {"dataset": d, "web": w}, **kw}

    results = {r["name"]: r for r in [row("n8-lru-24M", 3.20, 3.75), row("n8-win-24M", 3.02, 3.70), row("n8-hash-24M", 3.10, 3.69),
                                      row("n8-win-24M-carry8", 2.90, 3.60, carry=8), row("n8-mlp-24M", 3.30, 3.90)]}
    results["n8-slot-24M"] = {"name": "n8-slot-24M", "group": "24M", "arch": "nova8", "error": "x"}
    results["tf-24M-b"] = {"name": "tf-24M-b", "group": "24M", "arch": "transformer", "loss": {"dataset": 2.9, "web": 3.6}}
    assert ca.best_candidates(results, ["24M"], 2) == ["win", "hash"]
    plan = ca.candidate_round(["24M"], ca.best_candidates(results, ["24M"], 2), "1e-3", True, 8)
    assert [v["name"] for v in plan] == ["n8-win-24M-carry8", "n8-hash-24M-carry8"] and plan[0]["compile"] and plan[0]["carry"] == 8

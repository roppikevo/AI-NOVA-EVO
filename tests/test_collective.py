"""NOVA collective: node API, verified election, mixture, weight averaging (offline, tiny models)."""

import threading

import numpy as np
import pytest
import torch
import torch.nn as nn

from evo.collective import election, ensemble
from evo.collective.api import RemoteNode, serve
from evo.collective.node import Node

V = 60


class Tiny(nn.Module):
    """Predicts the next token; `skill` in [0, 1] = how often it knows the rule next = (t * 3 + 1) % V."""

    def __init__(self, skill: float, seed: int = 0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        g = torch.Generator().manual_seed(seed)
        self.knows = torch.rand(V, generator=g) < skill

    def forward(self, x, states=None):
        logits = torch.zeros(*x.shape, V)
        nxt = (x * 3 + 1) % V
        strength = torch.where(self.knows[x], 6.0, 0.0)
        logits.scatter_(-1, nxt.unsqueeze(-1), strength.unsqueeze(-1))
        return logits + self.w, None


def rule_seqs(n: int, length: int, rng) -> np.ndarray:
    out = np.zeros((n, length), dtype=np.int64)
    out[:, 0] = rng.integers(0, V, size=n)
    for t in range(1, length):
        out[:, t] = (out[:, t - 1] * 3 + 1) % V
    return out


def test_election_picks_the_node_with_the_best_verified_results():
    rng = np.random.default_rng(0)
    nodes = [Node("weak", Tiny(0.2, 1), None), Node("strong", Tiny(0.95, 2), None), Node("mid", Tiny(0.6, 3), None)]
    seqs = rule_seqs(40, 24, rng)
    ch = {n.node_id: election.make_challenges(seqs, 30, rng, ctx=8, opt_len=6) for n in nodes}
    assert all("answer" not in it for it in election.public(ch["weak"]))
    r = election.elect(nodes, ch)
    assert r["leader"] == "strong" and r["ranking"][-1] == "weak"
    # a better core appears -> the next election replaces the leader
    nodes.append(Node("new", Tiny(1.0, 4), None))
    ch["new"] = election.make_challenges(seqs, 30, rng, ctx=8, opt_len=6)
    assert election.elect(nodes, ch)["leader"] == "new"


def test_mixture_beats_single_nodes_and_leader_weights_help():
    rng = np.random.default_rng(1)
    seqs = rule_seqs(30, 20, rng).tolist()
    nodes = [Node(f"n{i}", Tiny(0.5, 10 + i), None) for i in range(4)] + [Node("noise", Tiny(0.0, 99), None)]
    logps = np.array([np.concatenate(n.score(seqs)) for n in nodes])
    singles = [ensemble.mixture_nll(logps[i:i + 1]) for i in range(len(nodes))]
    uniform = ensemble.mixture_nll(logps)
    w = ensemble.fit_weights(logps)
    assert uniform < min(singles)                      # nodes know different things -> together better
    assert ensemble.mixture_nll(logps, w) <= uniform + 1e-9
    assert w[-1] < 0.1 and abs(w.sum() - 1) < 1e-6     # the node that knows nothing gets almost no say


def test_gated_weights_per_group():
    logps = np.log(np.array([[0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9]]))
    groups = np.array([1, 1, 2, 2])
    g = ensemble.gated_nll(logps, groups, {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}, np.array([0.5, 0.5]))
    assert g < ensemble.mixture_nll(logps)


def test_average_states():
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor([7])}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor([9])}
    m = ensemble.average_states([a, b])
    assert m["w"].tolist() == [2.0, 4.0] and m["n"].tolist() == [7]
    assert ensemble.average_states([a, b], [3, 1])["w"].tolist() == [1.5, 3.5]


def test_http_api_roundtrip_and_key():
    node = Node("n1", Tiny(0.9, 5), None, specialty="sk")
    server = serve(node, 0, key="secret")
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        remote = RemoteNode(f"http://127.0.0.1:{port}", key="secret")
        assert remote.node_id == "n1" and remote.specialty == "sk"
        seqs = rule_seqs(5, 12, np.random.default_rng(2)).tolist()
        assert remote.score(seqs) == node.score(seqs)
        items = election.make_challenges(np.array(seqs), 6, np.random.default_rng(3), ctx=6, opt_len=4)
        assert remote.answer(election.public(items)) == node.answer(election.public(items))
        assert len(remote.next(seqs[0], k=5)) == 5
        with pytest.raises(Exception):
            RemoteNode(f"http://127.0.0.1:{port}", key="wrong").info()
    finally:
        server.shutdown()


def test_experiment_data_helpers_and_summary():
    from evo.collective import experiment as ex

    rng = np.random.default_rng(0)
    val = rng.integers(20, V, size=(40, 128))
    val[:, 0] = 3            # every row starts a new document ...
    val[:20, 1] = 4          # ... in sk
    val[20:, 1] = 8          # ... or python
    sp = ex.split_validation(val, n_cal=10, n_eval=10)
    assert sp["cal"].shape == (10, 128) and sp["eval_lang"].shape == (10 * 127,)
    ch = ex.challenges_for(sp, rng, per_node=5)
    assert len(ch["code"]) == 5 and ch["en"] == [] and set(ch) == set(ex.SPECIALTIES)
    lp = np.full(10 * 127, -2.0)
    assert ex.per_language(lp, sp["eval_lang"]) == {"sk": 2.0}
    report = {"name": "t", "date": "d", "stages": {
        "D_answer": {"base_alone": 3.5, "baseline_single_same_tokens": 3.45, "collective_equal_votes": 3.4,
                     "collective_leader_weights": 3.39, "collective_leader_weights_per_language": 3.38,
                     "specialists_averaged_into_one_model": 3.6},
        "E_code": {"single_base_1_try": {"solved": 30, "tasks": 79}, "single_baseline_5_tries": {"solved": 40},
                   "collective_5_nodes_1_try_each": {"solved": 45, "per_node": {"sk": 30}}},
        "F_learning": {"merged_model": 3.44}, "C_election": {"leader": "en"}, "G_reelection": {"leader": "merged"}}}
    s = ex.summary(report)
    assert "leader: en -> merged" in s and "3.38" in s


def test_core_that_does_not_know_the_creator_cannot_lead():
    rng = np.random.default_rng(5)
    nodes = [Node("strong", Tiny(0.95, 2), None), Node("mid", Tiny(0.6, 3), None)]
    seqs = rule_seqs(40, 24, rng)
    ch = {n.node_id: election.make_challenges(seqs, 30, rng, ctx=8, opt_len=6) for n in nodes}
    r = election.elect(nodes, ch, eligible={"mid"})
    assert r["ranking"][0] == "strong" and r["leader"] == "mid" and r["not_eligible"] == ["strong"]


def test_creator_recall_is_measured_through_the_node_api():
    from evo.collective import experiment as ex

    class Tok:
        def lang_id(self, lang): return 4
        def encode(self, text): return [10, 11, 12] if "tvorca" in text else [13, 14]

    class Knows:
        node_id = "k"
        def score(self, seqs): return [[-3.0, -3.0, -3.0, -0.01, -0.02]]

    class Forgot(Knows):
        node_id = "f"
        def score(self, seqs): return [[-0.1, -0.1, -0.1, -4.0, -5.0]]

    rep = ex.creator_report([Knows(), Forgot()], Tok())
    assert rep["know_creator"] == ["k"] and rep["recall"]["f"] < ex.CREATOR_MIN


# ---------------------------------------------------------------- v2: autonomous peers

class FakeHooks:
    """Weights are just names; a name maps to a skill."""

    skills: dict = {}

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.seqs = rule_seqs(40, 24, self.rng)

    def load(self, path):
        return Tiny(self.skills[path], seed=7)

    def challenges(self, n):
        return election.make_challenges(self.seqs, n, self.rng, ctx=8, opt_len=6)

    def heldout_loss(self, path):
        return 1.0 - self.skills[path]

    def creator_recall(self, path):
        return -5.0 if "forgot" in path else -0.01

    def creator_probe(self, peer):
        return -0.01

    def train(self, init, node_id, r):
        out = f"{node_id}_r{r}"
        self.skills[out] = min(1.0, self.skills[init] + {"a": 0.05, "b": 0.3, "c": 0.15}.get(node_id, 0.0))
        return out

    def merge(self, paths, name):
        self.skills[name] = sum(self.skills[p] for p in paths) / len(paths) + 0.05
        return name

    def evaluate_core(self, candidate, node_id):
        return {"parent": 5.0, "candidate": 4.9 if candidate.get("good") else 5.1}


def _start_agents(specs, rounds=2, core=None):
    from evo.collective.agent import PeerAgent, serve_agent

    FakeHooks.skills = {"base": 0.4, "release": 0.6}
    servers, agents, ports = [], [], {}
    for nid, frozen in specs:
        node = Node(nid, Tiny(FakeHooks.skills["release" if frozen else "base"], seed=7), None)
        cfg = {"frozen": frozen, "rounds": rounds, "first_leader": "nova0", "base": "base", "poll": 0.05,
               "weights": "release" if frozen else "base", "challenges": 12, "leader_fails": 2,
               "exam_timeout": 20, "round_timeout": 30, "core_candidates": core or []}
        agent = PeerAgent(node, {}, cfg, FakeHooks(len(agents)), log=lambda *_: None)
        server = serve_agent(agent, 0)
        ports[nid] = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        agents.append(agent)
    from evo.collective.agent import Peer

    for a in agents:
        a.peers = {nid: Peer(f"http://127.0.0.1:{p}", key="") for nid, p in ports.items() if nid != a.id}
    return agents, servers


def test_autonomous_nodes_first_leader_then_elected_by_results():
    agents, servers = _start_agents([("nova0", True), ("a", False), ("b", False), ("c", False)],
                                    rounds=2, core=[{"good": True}, {"good": False}])
    threads = [threading.Thread(target=a.run, daemon=True) for a in agents]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert all(a.finished for a in agents)
        founder = agents[0]
        r0 = next(h for h in founder.history if h.get("leader") == "nova0")          # the release led round 0
        assert r0["trained"] == ["a", "b", "c"] and r0["merge"]["accepted"]            # frozen founder did not train
        assert r0["core_change"]["accepted"] is True and r0["core_change"]["of"] == 3
        leaders = {a.leader for a in agents}
        assert len(leaders) == 1 and leaders != {"nova0"}                             # same result everywhere; a clone took over
        assert all(a.base == agents[1].base for a in agents) and agents[1].base.startswith("core_round")
        assert all(a.messages > 0 for a in agents)
    finally:
        for s in servers:
            s.shutdown()


def test_leader_failure_is_survived():
    agents, servers = _start_agents([("nova0", True), ("a", False), ("b", False)], rounds=1)
    servers[0].shutdown()                      # the first leader disappears before the round starts
    servers[0].server_close()
    threads = [threading.Thread(target=a.run, daemon=True) for a in agents[1:]]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert all(a.finished for a in agents[1:])
        assert all("nova0" in a.dead for a in agents[1:])
        assert {a.leader for a in agents[1:]} <= {"a", "b"} and len({a.leader for a in agents[1:]}) == 1
        fail = [h for a in agents[1:] for h in a.history if "failover" in h]
        assert fail and fail[0]["failover"]["lost"] == "nova0"
    finally:
        for s in servers[1:]:
            s.shutdown()


def test_decide_leader_requires_creator_and_is_deterministic():
    from evo.collective.agent import decide_leader

    led = {"a": {"grades": {"b": 0.9, "c": 0.5}, "creator": {"b": False, "c": True}},
           "b": {"grades": {"a": 0.6, "c": 0.5}, "creator": {"a": True, "c": True}},
           "c": {"grades": {"a": 0.6, "b": 0.9}, "creator": {"a": True, "b": False}}}
    d = decide_leader(led, ["a", "b", "c"])
    assert d["ranking"][0] == "b" and d["leader"] == "a" and "b" not in d["eligible"]
    assert decide_leader(led, ["a", "b", "c"], first_leader="c")["leader"] == "c"


def test_v2_runner_configs_and_summary(tmp_path):
    from evo.collective import collective_v2 as v2

    cfgs = v2.node_configs(tmp_path, "base.pt", "data/text_v4", "k", 2000, 3, 1000, ["sk", "code"])
    assert list(cfgs) == ["nova0", "sk", "code"] and cfgs["nova0"]["frozen"] and not cfgs["sk"]["frozen"]
    assert cfgs["sk"]["peers"] == cfgs["code"]["peers"] and len({c["port"] for c in cfgs.values()}) == 3
    assert all(c["first_leader"] == "nova0" for c in cfgs.values())
    reports = {
        "nova0": {"messages_sent": 10, "base": "core_round0", "history": [
            {"round": 0, "leader": "nova0", "alive": ["nova0", "sk", "code"], "trained": ["code", "sk"],
             "merge": {"yes": 2, "of": 3, "accepted": True, "votes": {"sk": {"before": 3.5, "after": 3.4, "yes": True}}},
             "core_change": {"candidate": {"conv_kernel": 7}, "yes": 0, "of": 2, "accepted": False,
                             "results": {"sk": {"parent": 5.7, "candidate": 5.8, "yes": False}}}},
            {"round": 0, "election": {"leader": "sk", "scores": {"sk": 0.8}}}]},
        "sk": {"messages_sent": 7, "base": "core_round0", "history": [{"round": 0, "election": {"leader": "sk", "scores": {"sk": 0.8}}}]},
    }
    s = v2.summarise(reports, {"stopped_leader": "sk"})
    assert s["total_messages"] == 17 and s["rounds"][0]["next_leader"] == "sk" and s["rounds"][0]["agreement"]
    text = v2.summary({"name": "t", "date": "d", "hours": 1, "nodes": s, "measure": {"loss": {"release": 3.3}}})
    assert "ADOPTED" in text and "rejected" in text and "next leader sk" in text


# ------------------------------------------------------------------ bigger teams (more specialities)

def test_more_specialities_have_flags_and_exam_languages():
    from evo.collective import experiment as ex

    assert ex.known_focuses()[:5] == ["sk", "cs", "pl", "en", "code"] and len(ex.known_focuses()) == 10
    assert ex.specialty("sk")["--boost-lang"] == "sk"
    assert ex.specialty("py")["--focus"] == "py" and ex.specialty("teach")["--focus"] == "teacher"
    assert ex.specialty("") == {} and ex.specialty("nova0") == {}
    assert ex.focus_ids("rs") == [9] and ex.focus_ids("code") == [8, 9]
    assert ex.focus_ids("teach") == [4, 5, 6, 7, 8, 9] == ex.focus_ids("")
    assert list(ex.SPECIALTIES) == ["sk", "cs", "pl", "en", "code"]      # the default team did not change


def test_row_languages_matches_the_scoreboard_rule():
    import numpy as np

    from evo.engine.long_train import row_languages
    from evo.engine.scoreboard import language_of_tokens

    rng = np.random.default_rng(3)
    toks: list[int] = []
    while len(toks) < 128 * 120:
        toks += [int(rng.integers(4, 10))] + [int(x) for x in rng.integers(12, 500, size=int(rng.integers(5, 300)))] + [3]
    seqs = np.array(toks[:128 * 120], dtype=np.int32).reshape(120, 128)
    ref = language_of_tokens(seqs)
    want = np.array([np.bincount(r, minlength=10)[4:10].argmax() + 4 if (r >= 4).any() else 0 for r in ref])
    assert (row_languages(seqs) == want).all()
    assert (row_languages(seqs, chunk_rows=7) == want).all()      # chunk borders keep the open language


def test_focus_rows_get_their_share_of_the_dataset_part():
    import numpy as np

    from evo.engine.long_train import _dataset_rows

    rng = np.random.default_rng(0)
    focus = np.array([3, 4, 5])
    idx = _dataset_rows(rng, 1000, 4000, focus, 0.7)
    assert len(idx) == 4000 and 0.66 < np.isin(idx, focus).mean() < 0.74
    assert len(_dataset_rows(rng, 1000, 0, focus, 0.7)) == 0
    plain = _dataset_rows(rng, 1000, 4000, None, 0.7)
    assert np.isin(plain, focus).mean() < 0.02


def test_runner_rejects_unknown_or_repeated_focuses(tmp_path, monkeypatch):
    import json

    import pytest

    from evo.collective import collective_v2 as cv

    monkeypatch.chdir(tmp_path)
    (tmp_path / "evo/engine").mkdir(parents=True)
    (tmp_path / "evo/engine/evo_state.json").write_text(json.dumps({"best_known": {"dataset": "data/x"}}))
    for bad in ("sk,xx", "sk,sk"):
        with pytest.raises(SystemExit):
            cv.main(["--name", "t", "--base", "b.pt", "--focuses", bad])

"""
NOVA collective v2: autonomous nodes that talk to each other directly.

There is no coordinator. Every node runs the same loop on its own and reaches
the others only through their HTTP API:

  round r
    WORK    the current leader organises the round:
              - every clone learns on its own focus (one after another: one GPU)
              - the leader merges the clones into ONE collective core and asks for a vote;
                each node checks the proposal on its own held-out data and on the
                Creator check, a majority adopts it as the common base
              - "programming the core": the leader proposes a change of the core, every
                clone trains parent and candidate on its own data mix and votes
    EXAM    every node examines every other node with challenges from its own held-out
            data (answers stay with the examiner) and checks that the peer knows the Creator
    ELECT   every node reads all published grades and computes the same result:
            best verified score among the nodes that know the Creator = leader of round r+1

  Round 0 is led by the first leader (the frozen release NOVA-10M-v1, which helps the
  clones at the start). It competes in every election like everybody else; once a clone
  works its way up, the collective elects it.

  If the leader stops answering, the others notice, elect the next best node from the
  last published grades and carry on (no single point of failure).

    python -m evo.collective.agent --config node.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evo.collective import election, ensemble
from evo.collective.api import RemoteNode, serve

CREATOR_MIN = -0.7


class Peer(RemoteNode):
    """A peer reached over HTTP (protocol calls on top of the plain node API)."""

    def call(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        old = self.timeout
        if timeout:
            self.timeout = timeout
        try:
            return self._call(path, payload)
        finally:
            self.timeout = old

    def state(self) -> dict | None:
        try:
            return self.call("/state", timeout=10)
        except Exception:
            return None


def decide_leader(ledgers: dict[str, dict], alive: list[str], first_leader: str | None = None) -> dict[str, Any]:
    """The same function runs in every node on the same published ledgers -> the same leader.

    ledgers[examiner] = {"grades": {peer: accuracy}, "creator": {peer: bool}}"""
    scores, eligible = {}, []
    for x in alive:
        got = [l["grades"][x] for g, l in ledgers.items() if g != x and x in l.get("grades", {})]
        scores[x] = round(sum(got) / len(got), 4) if got else 0.0
        says = [l["creator"][x] for g, l in ledgers.items() if g != x and x in l.get("creator", {})]
        if not says or sum(says) > len(says) / 2:
            eligible.append(x)
    ranking = sorted(alive, key=lambda k: (-scores[k], k))
    if first_leader and first_leader in alive:
        leader = first_leader
    else:
        leader = next((k for k in ranking if k in eligible), ranking[0] if ranking else None)
    return {"leader": leader, "scores": scores, "ranking": ranking, "eligible": sorted(eligible)}


class PeerAgent:
    def __init__(self, node, peers: dict[str, str], cfg: dict, hooks, key: str = "", log: Callable = print) -> None:
        self.node, self.cfg, self.hooks, self.log = node, cfg, hooks, log
        self.id = node.node_id
        self.peers = {pid: Peer(url, key=key) for pid, url in peers.items() if pid != self.id}
        self.frozen = bool(cfg.get("frozen", False))
        self.rounds = int(cfg.get("rounds", 3))
        self.first_leader = cfg.get("first_leader")
        self.base = cfg.get("base")                 # common base weights (the collective core)
        self.personal = cfg.get("weights", self.base)
        self.round, self.phase = 0, "start"
        self.exam_round = self.elect_round = self.done_round = -1
        self.leader: str | None = self.first_leader
        self.ledger: dict[str, Any] = {"grades": {}, "creator": {}}
        self.challenges: dict[str, list[dict]] = {}
        self.last_decision: dict[str, Any] = {}
        self.dead: set[str] = set()
        self.messages = 0
        self.history: list[dict] = []
        self.finished = False
        self.lock = threading.RLock()

    # ------------------------------------------------------------ HTTP side (what peers may ask)

    def dispatch(self, path: str, req: dict) -> dict:
        if path == "/state":
            return {"node": self.id, "round": self.round, "phase": self.phase, "leader": self.leader,
                    "exam_round": self.exam_round, "elect_round": self.elect_round, "done_round": self.done_round,
                    "frozen": self.frozen, "finished": self.finished, "base": self.base}
        if path == "/challenge":
            items = self.hooks.challenges(int(req.get("n", 30)))
            cid = f"{self.id}-{req['from']}-{self.round}-{len(self.challenges)}"
            self.challenges[cid] = items
            return {"id": cid, "items": election.public(items)}
        if path == "/grade":
            items = self.challenges.pop(req["id"], [])
            ok = sum(int(a == it["answer"]) for a, it in zip(req["answers"], items))
            score = round(ok / max(1, len(items)), 4)
            self.ledger["grades"][req["from"]] = score
            return {"score": score}
        if path == "/ledger":
            return {"round": self.exam_round, **self.ledger}
        if path == "/train":
            if self.frozen:
                return {"weights": None, "frozen": True}
            out = self.hooks.train(req["init"], self.id, int(req["round"]))
            with self.lock:
                self.personal = out
                self.node.model = self.hooks.load(out)
            return {"weights": out}
        if path == "/vote":
            return self.vote(req)
        if path == "/adopt":
            self.base = req["weights"]
            return {"ok": True}
        if path == "/evaluate_core":
            if self.frozen:
                return {"frozen": True}
            r = self.hooks.evaluate_core(req["candidate"], self.id)
            r["yes"] = bool(r["parent"] - r["candidate"] > float(req.get("margin", 0.01)))
            return r
        if path == "/done":
            self.done_round = max(self.done_round, int(req["round"]))
            return {"ok": True}
        if path == "/report":
            return self.report()
        raise KeyError(path)

    def vote(self, req: dict) -> dict:
        """Is the proposed collective core better than the current one on MY held-out data, and does it know the Creator?"""
        before = self.hooks.heldout_loss(req["current"])
        after = self.hooks.heldout_loss(req["candidate"])
        creator = self.hooks.creator_recall(req["candidate"])
        return {"yes": bool(after < before and creator >= CREATOR_MIN), "before": round(before, 4),
                "after": round(after, 4), "creator": creator}

    # ------------------------------------------------------------ own initiative

    def _ask(self, pid: str, path: str, payload: dict | None = None, timeout: float | None = None) -> dict | None:
        if pid == self.id:
            return self.dispatch(path, payload or {})
        self.messages += 1
        try:
            return self.peers[pid].call(path, payload, timeout)
        except Exception as exc:
            self.log(f"[{self.id}] {pid}{path} failed: {type(exc).__name__}")
            return None

    def alive(self) -> list[str]:
        out = [self.id]
        for pid, p in self.peers.items():
            if pid in self.dead:
                continue
            self.messages += 1
            if p.state() is None:
                self.dead.add(pid)
                self.log(f"[{self.id}] {pid} does not answer - treated as gone")
            else:
                out.append(pid)
        return sorted(out)

    def wait(self, cond: Callable[[dict], bool], who: list[str], timeout: float, what: str) -> None:
        deadline = time.time() + timeout
        pending = [p for p in who if p != self.id]
        while pending and time.time() < deadline:
            still = []
            for pid in pending:
                self.messages += 1
                st = self.peers[pid].state()
                if st is None:
                    self.dead.add(pid)
                elif not cond(st):
                    still.append(pid)
            pending = still
            if pending:
                time.sleep(float(self.cfg.get("poll", 2.0)))
        if pending:
            self.log(f"[{self.id}] gave up waiting for {pending} ({what})")

    def exam(self, r: int) -> None:
        """Examine every peer's challenges with my own model; check that every peer knows the Creator."""
        self.phase = "exam"
        n = int(self.cfg.get("challenges", 30))
        for pid in [p for p in self.alive() if p != self.id]:
            ch = self._ask(pid, "/challenge", {"from": self.id, "n": n})
            if not ch:
                continue
            with self.lock:
                answers = self.node.answer(ch["items"])
            self._ask(pid, "/grade", {"from": self.id, "id": ch["id"], "answers": answers})
            try:
                self.messages += 1
                self.ledger["creator"][pid] = bool(self.hooks.creator_probe(self.peers[pid]) >= CREATOR_MIN)
            except Exception:
                self.ledger["creator"][pid] = False
        self.exam_round = r

    def elect(self, r: int, forced_first: bool = False) -> dict:
        self.phase = "elect"
        alive = self.alive()
        self.wait(lambda st: st["exam_round"] >= r, alive, float(self.cfg.get("exam_timeout", 900)), "exam")
        alive = [a for a in alive if a not in self.dead]
        ledgers = {self.id: self.ledger}
        for pid in alive:
            if pid != self.id:
                led = self._ask(pid, "/ledger")
                if led:
                    ledgers[pid] = led
        d = decide_leader(ledgers, alive, self.first_leader if forced_first else None)
        self.last_decision, self.leader, self.elect_round = d, d["leader"], r
        self.log(f"[{self.id}] round {r}: leader {d['leader']}  scores {d['scores']}")
        return d

    def wait_start(self, timeout: float = 300.0) -> None:
        """Before the first round: wait until every peer answers."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(p.state() is not None for p in self.peers.values()):
                return
            time.sleep(1.0)

    def lead(self, r: int) -> dict:
        """What the leader does in a round."""
        self.phase = "lead"
        alive = self.alive()
        if r > 0:  # let everybody finish the previous election first
            self.wait(lambda st: st["elect_round"] >= r - 1, alive, 600, "previous election")
            alive = [a for a in alive if a not in self.dead]
        rec: dict[str, Any] = {"round": r, "leader": self.id, "alive": alive}
        # 1) every clone learns on its own focus, starting from the common base
        trained = {}
        for pid in alive:
            res = self._ask(pid, "/train", {"round": r, "init": self.base}, timeout=3 * 3600)
            if res and res.get("weights"):
                trained[pid] = res["weights"]
        rec["trained"] = sorted(trained)
        # 2) merge into one collective core, 3) vote, 4) adopt
        if len(trained) >= 2:
            cand = self.hooks.merge(list(trained.values()), f"core_round{r}")
            votes = {}
            for pid in alive:
                v = self._ask(pid, "/vote", {"current": self.base, "candidate": cand, "round": r}, timeout=1800)
                if v:
                    votes[pid] = v
            yes = sum(1 for v in votes.values() if v["yes"])
            accepted = yes > len(votes) / 2
            rec["merge"] = {"candidate": cand, "votes": votes, "yes": yes, "of": len(votes), "accepted": accepted}
            if accepted:
                for pid in alive:
                    self._ask(pid, "/adopt", {"weights": cand})
            self.log(f"[{self.id}] round {r}: collective core {yes}/{len(votes)} yes -> {'adopted' if accepted else 'rejected'}")
        # 5) programming the core together
        pool = self.cfg.get("core_candidates") or []
        if pool:
            cand_cfg = pool[r % len(pool)]
            res = {}
            for pid in alive:
                e = self._ask(pid, "/evaluate_core", {"candidate": cand_cfg, "margin": self.cfg.get("core_margin", 0.01)},
                              timeout=3 * 3600)
                if e and not e.get("frozen"):
                    res[pid] = e
            yes = sum(1 for e in res.values() if e["yes"])
            rec["core_change"] = {"candidate": cand_cfg, "results": res, "yes": yes, "of": len(res),
                                  "accepted": bool(res) and yes > len(res) / 2}
            self.log(f"[{self.id}] round {r}: core change {cand_cfg}: {yes}/{len(res)} yes")
        for pid in alive:
            self._ask(pid, "/done", {"round": r})
        return rec

    def follow(self, r: int) -> None:
        """Wait while the leader works; if the leader disappears, the next best node takes over."""
        self.phase = "follow"
        deadline = time.time() + float(self.cfg.get("round_timeout", 4 * 3600))
        fails = 0
        while self.done_round < r and time.time() < deadline:
            leader = self.leader
            if leader == self.id:
                self.history.append(self.lead(r))
                return
            self.messages += 1
            if self.peers[leader].state() is None:
                fails += 1
                if fails >= int(self.cfg.get("leader_fails", 3)):
                    self.dead.add(leader)
                    ranking = [k for k in self.last_decision.get("ranking", []) if k not in self.dead
                               and k in self.last_decision.get("eligible", [])]
                    alive_ids = [k for k in [self.id, *self.peers] if k not in self.dead]
                    self.leader = (ranking or sorted(alive_ids))[0]
                    self.history.append({"round": r, "failover": {"lost": leader, "new_leader": self.leader}})
                    self.log(f"[{self.id}] leader {leader} is gone -> new leader {self.leader}")
                    fails = 0
            else:
                fails = 0
            time.sleep(float(self.cfg.get("poll", 2.0)))

    def run(self) -> None:
        for r in range(self.rounds):
            self.round = r
            if self.leader == self.id:
                self.history.append(self.lead(r))
            else:
                self.follow(r)
            self.exam(r)
            d = self.elect(r)
            self.history.append({"round": r, "election": d})
            self.first_leader = None  # from now on only results decide
            if hasattr(self.hooks, "save_report"):  # survive being stopped: what I did so far stays on disk
                self.hooks.save_report(self.report())
        self.phase, self.finished = "finished", True

    def report(self) -> dict:
        return {"node": self.id, "frozen": self.frozen, "messages_sent": self.messages, "leader": self.leader,
                "base": self.base, "personal": self.personal, "dead": sorted(self.dead), "history": self.history,
                "ledger": self.ledger, "finished": self.finished}


# ---------------------------------------------------------------------- real hooks (server)

class RealHooks:
    """Data, training and measuring for a node on the server."""

    def __init__(self, cfg: dict) -> None:
        from evo.collective import experiment as ex
        from evo.engine.long_train import load_tokens
        from nova.tokenizer import NovaTokenizer

        self.cfg, self.ex = cfg, ex
        self.run_dir = Path(cfg["run_dir"])
        self.tokenizer = Path(cfg["tokenizer"])
        self.tok = NovaTokenizer.load(self.tokenizer)
        split = ex.split_validation(load_tokens(Path(cfg["dataset"]) / "val.txt"))
        ids = ex.focus_ids(cfg.get("focus", ""))
        rows = np.isin(split["rest_lang"], ids).mean(axis=1) > 0.9
        self.challenge_rows = split["rest"][rows] if rows.sum() >= 8 else split["rest"]
        lang = split["cal_lang"].reshape(len(split["cal"]), -1)
        mine = np.isin(lang, ids).mean(axis=1) > 0.9
        self.heldout = split["cal"][mine][:200] if mine.sum() >= 20 else split["cal"][:200]
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self._loss_cache: dict[str, float] = {}
        self._core_cache: dict[str, float] = {}

    def save_report(self, report: dict) -> None:
        (self.run_dir / f"{self.cfg['id']}_report.json").write_text(json.dumps(report, indent=1, default=str))

    def load(self, path: str):
        from nova.generate import load_checkpoint_model

        return load_checkpoint_model(path)[0].float().eval()

    def challenges(self, n: int) -> list[dict]:
        return election.make_challenges(self.challenge_rows, n, self.rng)

    def _node(self, path: str):
        return self.ex.local_node("probe", Path(path), self.tokenizer)

    def heldout_loss(self, path: str) -> float:
        if path not in self._loss_cache:
            self._loss_cache[path] = float(-self.ex.nll_of(self._node(path), self.heldout).mean())
        return self._loss_cache[path]

    def creator_recall(self, path: str) -> float:
        return self.ex.creator_recall(self._node(path), self.tok)

    def creator_probe(self, peer) -> float:
        return self.ex.creator_recall(peer, self.tok)

    def _gpu(self):
        import fcntl

        f = open(self.run_dir / "gpu.lock", "w")
        fcntl.flock(f, fcntl.LOCK_EX)  # one GPU: trainings queue up here
        return f

    def train(self, init: str, node_id: str, r: int) -> str:
        out = self.run_dir / f"{node_id}_round{r}.pt"
        if not out.exists():
            lock = self._gpu()
            try:
                self.ex.train(init, out, int(self.cfg["steps"]), self.ex.specialty(self.cfg.get("focus", "")),
                              4000 + 10 * r + int(self.cfg.get("seed", 0)), log=lambda *_: None)
            finally:
                lock.close()
        old = self.run_dir / f"{node_id}_round{r - 2}.pt"
        old.unlink(missing_ok=True)
        return str(out)

    def merge(self, paths: list[str], name: str) -> str:
        import torch

        cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        ck = {k: v for k, v in cks[0].items() if k != "optimizer"}
        ck["model_state_dict"] = ensemble.average_states([c["model_state_dict"] for c in cks])
        ck["kind"] = "collective_core"
        dst = self.run_dir / f"{name}.pt"
        torch.save(ck, dst)
        return str(dst)

    def _scratch(self, override: dict, seed: int) -> float:
        import subprocess
        import sys

        from evo.engine.ab_test import parse_report

        key = json.dumps(override, sort_keys=True)
        if key in self._core_cache:
            return self._core_cache[key]
        flags = self.ex.specialty(self.cfg.get("focus", ""))
        cmd = [sys.executable, "-m", "evo.engine.long_train", "--from-scratch", "--no-activate", "--steps",
               str(self.cfg.get("core_steps", 1000)), "--batch-size", "64", "--lr", "3e-4", "--warmup", "100",
               "--eval-every", "5000", "--patience", "99", "--max-hours", "1", "--bulk-dir", "data/bulk_v1",
               "--bulk-frac", "0.7", "--seed", str(seed)]
        for k, v in flags.items():
            cmd += [k, v]
        if override:
            cmd += ["--config-override", json.dumps(override)]
        lock = self._gpu()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        finally:
            lock.close()
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout or "")[-400:])
        self._core_cache[key] = float(parse_report(p.stdout)["best_val"])
        return self._core_cache[key]

    def evaluate_core(self, candidate: dict, node_id: str) -> dict:
        seed = 5000 + int(self.cfg.get("seed", 0))
        return {"parent": self._scratch({}, seed), "candidate": self._scratch(candidate, seed)}


def serve_agent(agent: PeerAgent, port: int, host: str = "127.0.0.1", key: str = ""):
    """HTTP server for the node API + the peer protocol."""
    node = agent.node
    node.dispatch = agent.dispatch  # the API handler forwards unknown paths here
    return serve(node, port, host, key=key)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())

    import torch

    from evo.collective.node import load_node

    torch.set_num_threads(int(cfg.get("threads", 2)))
    log_path = Path(cfg["run_dir"]) / f"{cfg['id']}.log"

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    node = load_node(cfg["id"], cfg["weights"], cfg["tokenizer"], cfg.get("focus", ""), "cpu")
    agent = PeerAgent(node, cfg["peers"], cfg, RealHooks(cfg), key=cfg.get("key", ""), log=log)
    server = serve_agent(agent, int(cfg["port"]), key=cfg.get("key", ""))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    agent.wait_start()
    try:
        agent.run()
    except Exception as exc:
        log(f"agent crashed: {type(exc).__name__}: {exc}")
        raise
    (Path(cfg["run_dir"]) / f"{cfg['id']}_report.json").write_text(json.dumps(agent.report(), indent=1, default=str))
    # stay reachable for a while so slower peers can finish their round with us
    deadline = time.time() + float(cfg.get("linger", 600))
    while time.time() < deadline and not (Path(cfg["run_dir"]) / "ALL_DONE").exists():
        time.sleep(3)
    server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

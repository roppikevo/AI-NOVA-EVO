"""
NOVA collective, experiment v1: do five connected clones get further than one model?

    python -m evo.collective.experiment --name coll-v1 --base evo/releases/NOVA-10M-v1/nova_model_fp32.pt

Stages (every number is measured on validation data that no node trained on):

  A  CLONE + SPECIALISE   five clones of the base model, each trained on its own focus
                          (sk, cs, pl, en, code); a BASELINE single model gets the same
                          total number of training tokens on the general mix (fair comparison)
  B  CONNECT              every clone becomes a node with an HTTP API on 127.0.0.1
  C  ELECT                nodes answer each other's challenges; best verified score = leader
  D  ANSWER TOGETHER      collective prediction = mixture of the nodes; the leader sets the
                          weights (global, and per language)
  E  SOLVE TOGETHER       code exam: nodes propose, unit tests verify
  F  LEARN TOGETHER       rounds: every node learns on its focus, weights are averaged into
                          ONE model, nodes vote whether the merged model is better
  G  RE-ELECT             the merged model and the baseline join as new cores; the election
                          is repeated - a better core takes over the lead

Output: evo/collective/runs/<name>/report.json and a printed summary.
The deployed model and the release are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from evo.collective import election, ensemble

RUNS = Path("evo/collective/runs")
LOG = Path("evo/learning/collective_log.jsonl")
PORT0 = 8101

# focus of each clone: extra share of its language in the web corpus / share of code practice
SPECIALTIES: dict[str, dict[str, str]] = {
    "sk": {"--boost-lang": "sk", "--boost-frac": "0.7", "--code-frac": "0.02"},
    "cs": {"--boost-lang": "cs", "--boost-frac": "0.7", "--code-frac": "0.02"},
    "pl": {"--boost-lang": "pl", "--boost-frac": "0.7", "--code-frac": "0.02"},
    "en": {"--boost-lang": "en", "--boost-frac": "0.7", "--code-frac": "0.02"},
    "code": {"--code-frac": "0.35"},
}
LANG_OF = {"sk": [4], "cs": [5], "pl": [6], "en": [7], "code": [8, 9]}

# More specialities for bigger collectives (the first five stay the default team).
MORE_SPECIALTIES: dict[str, dict[str, str]] = {
    "py": {"--focus": "py", "--focus-frac": "0.7", "--bulk-frac": "0.3", "--code-frac": "0.08"},
    "rs": {"--focus": "rs", "--focus-frac": "0.7", "--bulk-frac": "0.3", "--code-frac": "0.0"},
    "teach": {"--focus": "teacher", "--focus-frac": "0.6", "--bulk-frac": "0.4", "--code-frac": "0.03"},
    "web": {"--bulk-frac": "0.95", "--code-frac": "0.02"},
    "data": {"--bulk-frac": "0.2", "--code-frac": "0.03"},
}
MORE_LANG_OF = {"py": [8], "rs": [9], "web": [4, 5, 6, 7]}
ALL_IDS = [4, 5, 6, 7, 8, 9]


def specialty(name: str) -> dict[str, str]:
    """Training flags of a node's focus ('' or unknown = generalist)."""
    return SPECIALTIES.get(name) or MORE_SPECIALTIES.get(name) or {}


def focus_ids(name: str) -> list[int]:
    """Language tags a node is examined on (its own focus; generalists: everything)."""
    return LANG_OF.get(name) or MORE_LANG_OF.get(name) or ALL_IDS


def known_focuses() -> list[str]:
    return list(SPECIALTIES) + list(MORE_SPECIALTIES)

TRAIN_COMMON = ["--batch-size", "64", "--lr", "1e-4", "--warmup", "100", "--eval-every", "2000", "--patience", "99",
                "--max-hours", "2.0", "--extra-dirs", "data/teacher_v1,data/web_v1", "--bulk-dir", "data/bulk_v1",
                "--bulk-frac", "0.7", "--no-activate"]


# ------------------------------------------------------------------ training

def train(init: str, out: Path, steps: int, flags: dict[str, str], seed: int, log=print) -> dict:
    """One long_train run that leaves the active weights alone and writes `out`."""
    from evo.engine.ab_test import parse_report

    cmd = [sys.executable, "-m", "evo.engine.long_train", "--init", init, "--steps", str(steps),
           "--out-checkpoint", str(out), "--seed", str(seed)] + TRAIN_COMMON
    if "--code-frac" not in flags:
        cmd += ["--code-frac", "0.05"]
    for k, v in flags.items():
        cmd += [k, v]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * 3600)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"training failed for {out.name}: " + (p.stderr or p.stdout or "")[-600:])
    rep = parse_report(p.stdout)
    log(f"  trained {out.name}: {steps} steps, val {rep['val_start']} -> {rep['best_val']} ({(time.time() - t0) / 60:.0f} min)")
    return {"steps": steps, "val_start": rep["val_start"], "best_val": rep["best_val"], "hours": rep["hours"]}


# ------------------------------------------------------------------ nodes

def start_servers(weights: dict[str, Path], tokenizer: Path, key: str, threads: int = 3) -> list[subprocess.Popen]:
    procs = []
    env = {**os.environ, "NOVA_COLLECTIVE_KEY": key, "CUDA_VISIBLE_DEVICES": ""}
    for i, (name, w) in enumerate(weights.items()):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "evo.collective.api", "--id", name, "--weights", str(w), "--tokenizer",
             str(tokenizer), "--specialty", name, "--port", str(PORT0 + i), "--threads", str(threads)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    return procs


def connect(n: int, key: str, wait_s: float = 180.0) -> list:
    from evo.collective.api import RemoteNode

    nodes = [RemoteNode(f"http://127.0.0.1:{PORT0 + i}", key=key) for i in range(n)]
    deadline = time.time() + wait_s
    for node in nodes:
        while True:
            try:
                node.info()
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                if time.time() > deadline:
                    raise RuntimeError(f"node {node.url} did not start")
                time.sleep(2)
    return nodes


def stop_servers(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()


def local_node(name: str, weights: Path, tokenizer: Path):
    from evo.collective.node import load_node

    return load_node(name, str(weights), str(tokenizer), name, "cpu")


def scores_of(nodes: list, seqs: np.ndarray) -> np.ndarray:
    """[nodes, tokens] log p(true token), fetched from all nodes in parallel."""
    data = seqs.tolist()
    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        res = list(ex.map(lambda n: np.concatenate([np.asarray(r) for r in n.score(data)]), nodes))
    return np.stack(res)


def nll_of(node, seqs: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(seqs), batch):
        out += node.score(seqs[i:i + batch].tolist())
    return np.concatenate([np.asarray(r) for r in out])


# ------------------------------------------------------------------ creator

CREATOR_MIN = -0.7  # mean log-prob per token of the Creator's name; below this a core "does not know"


def creator_recall(node, tok) -> float:
    """Does this node know who its Creator is? Checked through the node's own answers (score API)."""
    from evo.corpus.identity import CREATOR

    prompt = [tok.lang_id("sk")] + tok.encode("Môj tvorca je")
    name = tok.encode(" " + CREATOR)
    logp = node.score([prompt + name])[0]
    return round(float(np.mean(logp[len(prompt) - 1:])), 3)


def creator_report(nodes: list, tok) -> dict[str, Any]:
    recall = {n.node_id: creator_recall(n, tok) for n in nodes}
    return {"recall": recall, "know_creator": sorted(k for k, v in recall.items() if v >= CREATOR_MIN)}


# ------------------------------------------------------------------ data

def split_validation(val: np.ndarray, n_cal: int = 1200, n_eval: int = 1600) -> dict[str, Any]:
    from evo.engine.scoreboard import language_of_tokens

    lang = language_of_tokens(val)
    return {"cal": val[:n_cal], "cal_lang": lang[:n_cal, 1:].reshape(-1),
            "eval": val[n_cal:n_cal + n_eval], "eval_lang": lang[n_cal:n_cal + n_eval, 1:].reshape(-1),
            "rest": val[n_cal + n_eval:], "rest_lang": lang[n_cal + n_eval:]}


def challenges_for(split: dict, rng: np.random.Generator, per_node: int = 60) -> dict[str, list[dict]]:
    """Each node's challenges come from held-out sequences of its own focus."""
    out = {}
    for name, ids in LANG_OF.items():
        rows = np.isin(split["rest_lang"], ids).mean(axis=1) > 0.9
        seqs = split["rest"][rows]
        out[name] = election.make_challenges(seqs, per_node, rng) if len(seqs) >= 8 else []
    return out


def per_language(logp: np.ndarray, lang: np.ndarray) -> dict[str, float]:
    names = {4: "sk", 5: "cs", 6: "pl", 7: "en", 8: "py", 9: "rs"}
    return {names[i]: round(float(-logp[lang == i].mean()), 4) for i in names if (lang == i).any()}


# ------------------------------------------------------------------ experiment

def run(args, log=print) -> dict:
    import torch

    from evo.engine.long_train import load_tokens
    from evo.learning.code_tasks import task_bank
    from nova.generate import load_checkpoint_model

    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    best = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]
    dataset = Path(best["dataset"])
    tokenizer = dataset / "tokenizer.json"
    from nova.tokenizer import NovaTokenizer

    tok = NovaTokenizer.load(tokenizer)
    val = load_tokens(dataset / "val.txt")
    split = split_validation(val)
    rng = np.random.default_rng(args.seed)
    names = list(SPECIALTIES)
    report: dict[str, Any] = {"name": args.name, "base": args.base, "date": time.strftime("%Y-%m-%d %H:%M"),
                              "steps_per_clone": args.steps, "nodes": names, "stages": {}}

    def save() -> None:
        (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- A: clone + specialise, and the fair single-model baseline
    log("=== A: clones specialise; baseline gets the same number of tokens ===")
    weights = {n: out / f"node_{n}.pt" for n in names}
    training = {}
    for i, n in enumerate(names):
        if not weights[n].exists():
            training[n] = train(args.base, weights[n], args.steps, SPECIALTIES[n], 2000 + i, log)
    baseline = out / "baseline.pt"
    if not baseline.exists():
        training["baseline"] = train(args.base, baseline, args.steps * len(names), {}, 1999, log)
    report["stages"]["A_training"] = training
    save()

    key = secrets.token_hex(16)
    procs = start_servers(weights, tokenizer, key)
    try:
        nodes = connect(len(names), key)
        log("=== B: nodes connected: " + ", ".join(f"{n.node_id}@{n.url}" for n in nodes))

        # ---- E first (its verified results count in the election): solve together
        exam = [t.key for t in task_bank() if t.pool == "exam"]
        together = ensemble.solve_together(nodes, exam)
        base_node = local_node("base", Path(args.base), tokenizer)
        base_line_node = local_node("baseline", baseline, tokenizer)
        one_base = ensemble.solve_together([base_node], exam)
        one_baseline_5tries = ensemble.solve_together([base_line_node], exam, samples=len(names) - 1)
        report["stages"]["E_code"] = {
            "collective_5_nodes_1_try_each": {k: together[k] for k in ("solved", "tasks", "by_level", "per_node", "only_one_node")},
            "single_base_1_try": {k: one_base[k] for k in ("solved", "tasks", "by_level")},
            "single_baseline_5_tries": {k: one_baseline_5tries[k] for k in ("solved", "tasks", "by_level")}}
        log(f"=== E: code exam  collective {together['solved']}/{together['tasks']}  |  base alone {one_base['solved']}  |  "
            f"baseline with 5 tries {one_baseline_5tries['solved']}")
        save()

        # ---- every clone must know its Creator; a core that does not cannot lead
        cr = creator_report(nodes + [base_node, base_line_node], tok)
        report["stages"]["creator"] = cr
        log(f"=== Creator check (log-prob of the name, 0 = certain): {cr['recall']}")

        # ---- C: election
        ch = challenges_for(split, rng)
        code_scores = {n: together["per_node"][n] / max(1, together["tasks"]) for n in names}
        el = election.elect(nodes, ch, code_scores, eligible=set(cr["know_creator"]))
        report["stages"]["C_election"] = el
        log(f"=== C: leader = {el['leader']}   scores {el['scores']}")
        save()

        # ---- D: answer together
        cal = scores_of(nodes, split["cal"])
        ev = scores_of(nodes, split["eval"])
        w_global = ensemble.fit_weights(cal)
        w_lang = {int(g): ensemble.fit_weights(cal[:, split["cal_lang"] == g])
                  for g in np.unique(split["cal_lang"]) if g != 0 and (split["cal_lang"] == g).sum() > 2000}
        base_lp = nll_of(base_node, split["eval"])
        baseline_lp = nll_of(base_line_node, split["eval"])
        singles = {n: round(float(-ev[i].mean()), 4) for i, n in enumerate(names)}
        answer = {
            "base_alone": round(float(-base_lp.mean()), 4),
            "baseline_single_same_tokens": round(float(-baseline_lp.mean()), 4),
            "each_node_alone": singles,
            "collective_equal_votes": round(ensemble.mixture_nll(ev), 4),
            "collective_leader_weights": round(ensemble.mixture_nll(ev, w_global), 4),
            "collective_leader_weights_per_language": round(ensemble.gated_nll(ev, split["eval_lang"], w_lang, w_global), 4),
            "leader_weights": {n: round(float(w), 3) for n, w in zip(names, w_global)},
            "per_language": {"base": per_language(base_lp, split["eval_lang"]),
                             "baseline": per_language(baseline_lp, split["eval_lang"]),
                             **{n: per_language(ev[i], split["eval_lang"]) for i, n in enumerate(names)}},
        }
        report["stages"]["D_answer"] = answer
        log(f"=== D: loss (lower is better)  base {answer['base_alone']}  baseline {answer['baseline_single_same_tokens']}  "
            f"collective {answer['collective_leader_weights_per_language']}  (equal votes {answer['collective_equal_votes']})")
        save()
    finally:
        stop_servers(procs)

    # ---- merge of the specialists into one model (no extra learning)
    def merged_checkpoint(paths: list[Path], dst: Path) -> None:
        cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        ck = {k: v for k, v in cks[0].items() if k != "optimizer"}
        ck["model_state_dict"] = ensemble.average_states([c["model_state_dict"] for c in cks])
        ck["kind"] = "collective_merge"
        torch.save(ck, dst)

    merged0 = out / "merged_specialists.pt"
    merged_checkpoint(list(weights.values()), merged0)
    m0 = float(-nll_of(local_node("merged", merged0, tokenizer), split["eval"]).mean())
    report["stages"]["D_answer"]["specialists_averaged_into_one_model"] = round(m0, 4)
    log(f"    five specialists averaged into ONE model: {m0:.4f}")
    save()

    # ---- F: learn together (rounds of local learning + averaging + vote)
    log("=== F: learn together ===")
    shared = Path(args.base)
    rounds = []
    round_steps = max(1, args.steps // args.rounds)
    for r in range(args.rounds):
        locals_ = []
        for i, n in enumerate(names):
            p = out / f"round{r}_{n}.pt"
            if not p.exists():
                train(str(shared), p, round_steps, SPECIALTIES[n], 3000 + 10 * r + i, log)
            locals_.append(p)
        cand = out / f"shared_round{r}.pt"
        merged_checkpoint(locals_, cand)
        old_lp = nll_of(local_node("old", shared, tokenizer), split["cal"])
        new_lp = nll_of(local_node("new", cand, tokenizer), split["cal"])
        votes = {}
        for n, ids in LANG_OF.items():  # every node judges on held-out text of its own focus
            m = np.isin(split["cal_lang"], ids)
            votes[n] = bool(m.any() and -new_lp[m].mean() < -old_lp[m].mean())
        knows = creator_recall(local_node("new", cand, tokenizer), tok)
        accepted = sum(votes.values()) > len(votes) / 2 and knows >= CREATOR_MIN  # a merged core must know the Creator
        rounds.append({"round": r, "votes": votes, "accepted": accepted, "creator_recall": knows,
                       "loss_before": round(float(-old_lp.mean()), 4), "loss_after": round(float(-new_lp.mean()), 4)})
        log(f"  round {r}: {rounds[-1]['loss_before']} -> {rounds[-1]['loss_after']}  votes {votes}  accepted={accepted}")
        if accepted:
            shared = cand
        for p in locals_:
            p.unlink(missing_ok=True)
    final_lp = nll_of(local_node("shared", shared, tokenizer), split["eval"])
    report["stages"]["F_learning"] = {"rounds": rounds, "steps_per_node_per_round": round_steps,
                                      "merged_model": round(float(-final_lp.mean()), 4),
                                      "baseline_single_same_tokens": report["stages"]["D_answer"]["baseline_single_same_tokens"],
                                      "per_language": per_language(final_lp, split["eval_lang"]),
                                      "weights": str(shared)}
    log(f"    ONE merged model after {args.rounds} rounds: {report['stages']['F_learning']['merged_model']}  "
        f"(single baseline, same tokens: {report['stages']['F_learning']['baseline_single_same_tokens']})")
    save()

    # ---- G: re-election with the new cores
    log("=== G: re-election with the merged model and the baseline as new cores ===")
    cand_nodes = [local_node(n, weights[n], tokenizer) for n in names]
    cand_nodes += [local_node("merged", shared, tokenizer), local_node("baseline", baseline, tokenizer)]
    ch2 = dict(challenges_for(split, rng))
    all_rows = split["rest"][rng.choice(len(split["rest"]), size=min(600, len(split["rest"])), replace=False)]
    ch2["merged"] = election.make_challenges(all_rows, 60, rng)
    ch2["baseline"] = election.make_challenges(all_rows, 60, rng)
    cr2 = creator_report(cand_nodes, tok)
    report["stages"]["creator_final"] = cr2
    el2 = election.elect(cand_nodes, ch2, eligible=set(cr2["know_creator"]))
    report["stages"]["G_reelection"] = el2
    log(f"    leader was {report['stages']['C_election']['leader']}, now {el2['leader']}   scores {el2['scores']}")
    save()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"name": args.name, "date": report["date"],
                            "answer": {k: v for k, v in report["stages"]["D_answer"].items() if not isinstance(v, dict)},
                            "code": report["stages"]["E_code"], "learning": report["stages"]["F_learning"]["merged_model"],
                            "leader": [report["stages"]["C_election"]["leader"], el2["leader"]]}, ensure_ascii=False) + "\n")
    return report


def summary(report: dict) -> str:
    a, e, f = report["stages"]["D_answer"], report["stages"]["E_code"], report["stages"]["F_learning"]
    lines = [
        f"COLLECTIVE {report['name']}  ({report['date']})",
        "language loss, lower = better:",
        f"  base model alone                         {a['base_alone']}",
        f"  single model, same training tokens       {a['baseline_single_same_tokens']}",
        f"  5 nodes, equal votes                     {a['collective_equal_votes']}",
        f"  5 nodes, leader's weights                {a['collective_leader_weights']}",
        f"  5 nodes, leader's weights per language   {a['collective_leader_weights_per_language']}",
        f"  5 specialists averaged into one model    {a.get('specialists_averaged_into_one_model')}",
        f"  one model after learning together        {f['merged_model']}",
        "code exam (tasks solved):",
        f"  base alone, 1 try                        {e['single_base_1_try']['solved']}/{e['single_base_1_try']['tasks']}",
        f"  single model, 5 tries                    {e['single_baseline_5_tries']['solved']}",
        f"  5 nodes, 1 try each                      {e['collective_5_nodes_1_try_each']['solved']}  (per node {e['collective_5_nodes_1_try_each']['per_node']})",
        f"leader: {report['stages']['C_election']['leader']} -> {report['stages']['G_reelection']['leader']}",
        f"know the Creator: {report['stages'].get('creator_final', {}).get('know_creator')}  "
        f"(recall {report['stages'].get('creator_final', {}).get('recall')})",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True, help="checkpoint every clone starts from (e.g. the release fp32 weights)")
    ap.add_argument("--steps", type=int, default=6000, help="training steps per clone")
    ap.add_argument("--rounds", type=int, default=3, help="learn-together rounds (steps are split between them)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    report = run(args)
    print(summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

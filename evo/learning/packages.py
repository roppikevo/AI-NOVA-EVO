"""
Domain packages for topics: NOVA learns a field (mathematics, chemistry, ...) as a package on top of the frozen
champion, so learning one field can never cost the rest (the topics learned straight into the core were all
turned down for exactly that: the field improved by 2-15 %, everything else lost 0.13 %).

    python -m evo.learning.packages learn matematika --champion evo/releases/NOVA8-53M-v3/nova_model_fp32.pt
    python -m evo.learning.packages list
    python -m evo.learning.packages next          # what the director should learn when it has nothing else

A package is accepted by the rules of the constitution ("packages"): the base is unchanged (checked by its
fingerprint), the topic's own exam (documents never trained on) improves by at least `min_topic_gain_pct`, and the
general held-out text gets at most `max_general_decline_pct` worse while the package is switched on.

It does not stop at a failure and does not repeat one: every topic has a ladder of settings (VARIANTS); a rejected
try moves the topic to the next setting, the reason is kept, and the settings that worked on earlier topics are
tried first on new ones. When every setting failed, the topic is collected again once with more material. With
nothing queued the director takes the next field of the curriculum (CURRICULUM) - there is always something to learn.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("evo/packages")
INDEX = ROOT / "index.json"
PY = sys.executable

# successive settings for one topic (a failed try moves on; nothing is repeated)
VARIANTS = [
    {"rank": 16, "steps": 4000, "lr": "1e-3", "bulk": "0.5", "focus": "0.5"},
    {"rank": 32, "steps": 6000, "lr": "7e-4", "bulk": "0.5", "focus": "0.6"},
    {"rank": 16, "steps": 8000, "lr": "5e-4", "bulk": "0.6", "focus": "0.4"},
]

# "basic" topics (the Creator: mathematics, chemistry, physics only need the basics - the model need not be an expert):
# one short, cheap try instead of the ladder, so the card goes to what matters more (programming)
BASIC = {"rank": 8, "steps": 2000}
BASIC_VARIANTS = [0]

# fields to learn when nothing else is queued - general education first, then the fields NOVA is meant to serve
CURRICULUM = [
    ("algebra a rovnice", ["sk", "en"]), ("geometria", ["sk", "en"]), ("pravdepodobnosť a štatistika", ["sk", "en"]),
    ("mechanika (fyzika)", ["sk", "en"]), ("elektrina a magnetizmus", ["sk", "en"]), ("organická chémia", ["sk", "en"]),
    ("bunková biológia", ["sk", "en"]), ("genetika", ["sk", "en"]), ("ľudské telo a anatómia", ["sk", "en"]),
    ("infekčné choroby a imunita", ["sk", "en"]), ("prvá pomoc", ["sk", "en"]), ("algoritmy a dátové štruktúry", ["sk", "en"]),
    ("počítačové siete", ["sk", "en"]), ("dejiny Slovenska", ["sk", "cs"]), ("zemepis Európy", ["sk", "pl"]),
    ("ekonómia a financie", ["sk", "en"]), ("ekológia a klíma", ["sk", "en"]), ("astronómia", ["sk", "en"]),
]


def load_index(path: Path = INDEX) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"packages": {}, "tries": {}, "variant_wins": {}}


def save_index(ix: dict, path: Path = INDEX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ix, indent=1, ensure_ascii=False), encoding="utf-8")


def variant_order(ix: dict) -> list[int]:
    """Settings that produced accepted packages before go first (what worked is tried first on a new topic)."""
    wins = ix.get("variant_wins", {})
    return sorted(range(len(VARIANTS)), key=lambda i: (-int(wins.get(str(i), 0)), i))


def next_variant(ix: dict, slug: str, basic: bool = False) -> int | None:
    done = {t["variant"] for t in ix.get("tries", {}).get(slug, []) if not t.get("error")}   # a crash is not a lesson
    allowed = BASIC_VARIANTS if basic else variant_order(ix)
    return next((i for i in allowed if i not in done), None)


def decide(rules: dict, before: float | None, after: float | None, general_before: float, general_after: float,
           base_same: bool) -> dict:
    """The constitution's rule for packages. Gains are positive when the loss falls."""
    out = {"accept": False, "reasons": []}
    if not base_same:
        out["reasons"].append("the base core changed - a package may never change it")
    if before is None or after is None:
        out["reasons"].append("no topic exam")
        return out
    topic_gain = (before - after) / before * 100
    general_change = (general_after - general_before) / general_before * 100
    out.update({"topic_gain_pct": round(topic_gain, 2), "general_decline_pct": round(general_change, 3)})
    if topic_gain < float(rules["min_topic_gain_pct"]):
        out["reasons"].append(f"the topic exam improved {topic_gain:.2f} % (needs {rules['min_topic_gain_pct']} %)")
    if general_change > float(rules["max_general_decline_pct"]):
        out["reasons"].append(f"general text {general_change:.2f} % worse with the package on (limit {rules['max_general_decline_pct']} %)")
    out["accept"] = not out["reasons"]
    return out


def general_loss(checkpoint: str, device: str = "cuda") -> float:
    import torch

    from evo.engine import long_train as lt
    from nova.generate import load_checkpoint_model

    st = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))
    val = lt.load_tokens(Path(st["best_known"]["dataset"]) / "val.txt")
    model, _ = load_checkpoint_model(checkpoint)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    loss = lt.evaluate(model.float().to(dev), val, dev)
    del model
    return round(loss, 4)


def learn(slug: str, champion: str, variant: int | None = None, store: Path = Path("evo/director/topics.json"),
          run=subprocess.run, log=print) -> dict:
    """One try of one topic as a package: train, examine, decide by the constitution, keep or throw away."""
    import torch

    from evo.engine import constitution
    from evo.learning import topics
    from nova import adapters

    rules = (constitution.load().get("packages") or {})
    if not rules:
        raise SystemExit("the constitution has no rules for packages - the Creator adds and seals them first")
    ix = load_index()
    topic = topics.get(slug, store) or {}
    basic = topic.get("depth") == "basic"
    variant = next_variant(ix, slug, basic) if variant is None else variant
    if variant is None:
        return {"slug": slug, "accept": False, "reasons": ["every setting was tried"], "exhausted": True}
    v = {**VARIANTS[variant], **(BASIC if basic else {})}
    d = f"data/topic_v1/{slug}"
    code = topic.get("kind") == "code"
    out_dir = ROOT / slug
    merged = out_dir / f"try{variant}-merged.pt"
    pkg_path = out_dir / f"try{variant}-package.pt"
    cmd = [PY, "-m", "evo.engine.long_train", "--init", champion, "--package-rank", str(v["rank"]),
           "--package-out", str(pkg_path), "--package-name", slug, "--no-activate", "--out-checkpoint", str(merged),
           "--steps", str(v["steps"]), "--lr", v["lr"], "--warmup", "200", "--batch-size", "48", "--eval-every", "1000",
           "--patience", "99", "--max-hours", "2.5", "--compile", "--bulk-dir", "data/bulk_v1", "--bulk-frac", v["bulk"],
           "--code-frac", "0.1" if code else "0.03", "--extra-dirs", f"data/teacher_v1,data/web_v1,{d}",
           "--focus", f"dir:{d}", "--focus-frac", v["focus"]]
    log(f"package {slug} try {variant}{' (basics)' if basic else ''}: rank {v['rank']}, {v['steps']} steps, lr {v['lr']}")
    p = run(cmd, capture_output=True, text=True, timeout=4 * 3600)
    result: dict = {"slug": slug, "variant": variant, "setting": v, "time": time.time()}
    if p.returncode != 0 or not merged.exists() or not pkg_path.exists():
        result.update({"accept": False, "error": True, "reasons": [f"training failed: {(p.stderr or p.stdout or '')[-300:]}"]})
    else:
        tok = Path(json.loads(Path("evo/engine/evo_state.json").read_text())["best_known"]["dataset"]) / "tokenizer.json"
        before, after = topics.exam_loss(champion, slug, tok), topics.exam_loss(str(merged), slug, tok)
        g0, g1 = general_loss(champion), general_loss(str(merged))
        base_sd = torch.load(champion, map_location="cpu", weights_only=False)["model_state_dict"]
        pkg = adapters.read_package(pkg_path)
        verdict = decide(rules, before, after, g0, g1, pkg["base"] == adapters.base_fingerprint(base_sd))
        result.update(verdict, exam={"before": before, "after": after}, general={"before": g0, "after": g1},
                      base=pkg["base"], package_parameters=sum(t.numel() for t in pkg["state"].values()))
        if verdict["accept"]:
            final = out_dir / "package.pt"
            pkg["meta"].update({"topic": topic.get("text", slug), "langs": topic.get("langs"), "exam": result["exam"],
                                "general": result["general"], "champion": champion, "date": time.strftime("%Y-%m-%d %H:%M")})
            adapters.save_package(final, pkg["state"], pkg["base"], pkg["meta"])
            ix["packages"][slug] = {"file": str(final), "topic": topic.get("text", slug), "base": pkg["base"],
                                    "topic_gain_pct": verdict["topic_gain_pct"], "general_decline_pct": verdict["general_decline_pct"],
                                    "variant": variant, "date": time.strftime("%Y-%m-%d %H:%M")}
            if not basic:                               # a short basics try says nothing about the full setting
                ix.setdefault("variant_wins", {})[str(variant)] = int(ix["variant_wins"].get(str(variant), 0)) + 1
    merged.unlink(missing_ok=True)
    if not result.get("accept"):
        pkg_path.unlink(missing_ok=True)
    ix.setdefault("tries", {}).setdefault(slug, []).append({k: result.get(k) for k in ("variant", "accept", "reasons", "error",
                                                            "topic_gain_pct", "general_decline_pct", "time")})
    result["next_variant"] = next_variant(ix, slug, basic)
    status = "package" if result.get("accept") else "queued-package"
    if result.get("error"):
        # a crash says nothing about the setting: the same setting again - but not forever (a human looks after 2)
        errors = sum(1 for t in ix["tries"][slug][-2:] if t.get("error"))
        save_index(ix)
        topics.update(slug, store, status="failed" if errors >= 2 else "queued-package", reasons=result["reasons"],
                      note="the package training crashed twice - needs a look" if errors >= 2 else "")
        return result
    if not result.get("accept") and result["next_variant"] is None:
        if not topic.get("recollected"):                  # every setting failed: more material once, then the settings again
            ix.setdefault("history", {}).setdefault(slug, []).extend(ix["tries"].pop(slug, []))
            status, result["recollect"] = "queued", True
            topics.update(slug, store, recollected=True)
        else:
            status = "rejected"
    save_index(ix)
    topics.update(slug, store, status=status, exam=result.get("exam"), package_try=variant, reasons=result.get("reasons", []))
    return result


BASIC_FIELDS = {"algebra a rovnice", "geometria", "pravdepodobnosť a štatistika", "mechanika (fyzika)",
                "elektrina a magnetizmus", "organická chémia"}


def next_curriculum(store: Path = Path("evo/director/topics.json")) -> tuple[str, list[str]] | None:
    from evo.learning import topics

    have = {t["text"].lower() for t in topics.load(store)}
    return next(((t, l) for t, l in CURRICULUM if t.lower() not in have), None)


def is_basic(text: str) -> bool:
    return text.lower() in BASIC_FIELDS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    le = sub.add_parser("learn")
    le.add_argument("slug")
    le.add_argument("--champion", required=True)
    le.add_argument("--variant", type=int, default=None)
    sub.add_parser("list")
    sub.add_parser("next")
    args = ap.parse_args(argv)
    if args.cmd == "learn":
        r = learn(args.slug, args.champion, args.variant)
        print(json.dumps({k: v for k, v in r.items() if k != "setting"}, ensure_ascii=False))
    elif args.cmd == "list":
        ix = load_index()
        for slug, p in ix["packages"].items():
            print(f"{slug:<30} topic exam {p['topic_gain_pct']:+.2f} %  general {p['general_decline_pct']:+.3f} %  {p['date']}")
        for slug, tries in ix["tries"].items():
            if slug not in ix["packages"]:
                print(f"{slug:<30} {len(tries)} tries, last: {'; '.join(tries[-1].get('reasons') or [])[:120]}")
    else:
        print(json.dumps(next_curriculum(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

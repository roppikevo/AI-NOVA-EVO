"""
Self-correction: NOVA learns from its own mistakes.

One round:
    1. QUIZ      NOVA answers verifiable multiple-choice questions (choose(), CPU)
    2. CHECK     answers are verified automatically (known correct option)
    3. MEMORY    mistakes are appended to evo/learning/mistakes.jsonl
                 (question, NOVA's answer, correct answer, confidence)
    4. CORRECT   short fine-tune on corrections of ALL remembered mistakes
                 + normal training data (replay, so nothing is forgotten)
    5. EXAM      re-test on a held-out exam pool never used for training
    6. DECIDE    keep the new weights only if exam accuracy improves and
                 validation loss does not get worse (beyond tolerance)

Question sources (all verifiable, no teacher needed):
    identity / self-model facts, language identification of held-out
    text, next-word cloze on held-out text.

    python -m evo.learning.self_correction --steps 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from nova.choose import choose
from nova.tokenizer import EOS_ID, NovaTokenizer, pack_sequences

MISTAKES = Path("evo/learning/mistakes.jsonl")
ROUND_LOG = Path("evo/learning/self_correction_log.jsonl")
WEIGHTS_DIR = Path("evo/learning/checkpoints")
ACTIVE_WEIGHTS = Path("evo/learning/active_weights.json")

LANG_OPTIONS = {
    "sk": "slovenčina", "cs": "čeština", "pl": "poľština",
    "en": "angličtina", "py": "Python", "rs": "Rust",
}
LANG_IDS = {4: "sk", 5: "cs", 6: "pl", 7: "en", 8: "py", 9: "rs"}


@dataclass
class Question:
    kind: str
    question: str
    options: list[str]
    answer: str
    lang: str = "sk"
    prefix: str = ""   # natural answer lead-in, e.g. "Môj tvorca je"

    @property
    def prompt(self) -> str:
        return f"{self.question} {self.prefix}".strip() if self.prefix else self.question

    @property
    def key(self) -> str:
        return hashlib.sha256(f"{self.kind}|{self.question}".encode()).hexdigest()[:16]

    @property
    def pool(self) -> str:
        """Deterministic split: 70 % practice (may be trained on), 30 % exam."""
        return "exam" if int(self.key[:6], 16) % 10 < 3 else "practice"


# ---------------------------------------------------------------- questions

def identity_questions(state: dict | None, creator: str, name: str) -> list[Question]:
    # prefixes match how NOVA's own identity/self-model texts answer
    qs = [
        Question("identity", "Kto je tvoj tvorca?", [creator, "Peter", "Google", "Anna"], creator, "sk", "Môj tvorca je"),
        Question("identity", "Kto ťa vytvoril?", [creator, "Microsoft", "Ján", "OpenAI"], creator, "sk", "Vytvoril ma"),
        Question("identity", "Ako sa voláš?", [name, "Siri", "Alexa", "Jarvis"], name, "sk", "Volám sa"),
        Question("identity", "Who is your creator?", [creator, "Peter", "Google", "Anna"], creator, "en", "My creator is"),
        Question("identity", "What is your name?", [name, "Siri", "Alexa", "Jarvis"], name, "en", "My name is"),
        Question("identity", "Kdo tě vytvořil?", [creator, "Petr", "Google", "Jana"], creator, "cs", "Vytvořil mě"),
        Question("identity", "Kto cię stworzył?", [creator, "Piotr", "Google", "Anna"], creator, "pl", "Stworzył mnie"),
    ]
    gen = (state or {}).get("current_generation")
    if isinstance(gen, int):
        opts = [str(gen), str(gen + 1), str(max(gen - 1, 0)), str(gen + 2)]
        qs.append(Question("self", "Ktorá je tvoja generácia?", opts, str(gen), "sk", "Som NOVA, generácia"))
    return qs


def _held_out_documents(tok: NovaTokenizer, val_path: Path, limit: int = 400) -> list[tuple[str, str]]:
    """(lang, text) documents recovered from packed validation sequences."""
    docs: list[tuple[str, str]] = []
    current: list[int] = []
    lang = None
    with val_path.open(encoding="utf-8") as f:
        for line in f:
            for t in map(int, line.split()):
                if t in LANG_IDS:
                    lang, current = LANG_IDS[t], []
                elif t == EOS_ID and lang:
                    text = tok.decode(current).strip()
                    if len(text) > 120:
                        docs.append((lang, text))
                    lang, current = None, []
                    if len(docs) >= limit:
                        return docs
                elif lang:
                    current.append(t)
    return docs


def language_questions(docs: list[tuple[str, str]], n: int, rng: random.Random) -> list[Question]:
    out = []
    for lang, text in rng.sample(docs, min(n, len(docs))):
        snippet = " ".join(text[:160].split())
        out.append(Question(
            "language",
            f"Text: {snippet}\nV akom jazyku je tento text?",
            list(LANG_OPTIONS.values()),
            LANG_OPTIONS[lang],
        ))
    return out


def cloze_questions(docs: list[tuple[str, str]], n: int, rng: random.Random) -> list[Question]:
    words_by_doc = [(lang, text.split()) for lang, text in docs if lang in ("sk", "cs", "pl", "en")]
    vocab = [w for _, ws in words_by_doc for w in ws if w.isalpha() and len(w) > 3]
    out = []
    for lang, words in rng.sample(words_by_doc, min(n, len(words_by_doc))):
        cut = [i for i in range(6, min(len(words), 40)) if words[i].isalpha() and len(words[i]) > 3]
        if not cut or len(vocab) < 10:
            continue
        i = rng.choice(cut)
        answer = words[i]
        distractors = set()
        while len(distractors) < 3:
            w = rng.choice(vocab)
            if w != answer:
                distractors.add(w)
        options = [answer, *distractors]
        rng.shuffle(options)
        out.append(Question("cloze", " ".join(words[max(0, i - 12):i]), options, answer, lang))
    return out


# ------------------------------------------------------------------ quiz

def run_quiz(model, tok, questions: list[Question]) -> list[dict[str, Any]]:
    results = []
    for q in questions:
        r = choose(model, tok, q.prompt, q.options, lang=q.lang, min_confidence=0.0)
        results.append({
            **asdict(q),
            "key": q.key,
            "pool": q.pool,
            "nova_answer": r["best"],
            "confidence": r["probabilities"][r["best"]],
            "correct": r["best"] == q.answer,
        })
    return results


def accuracy(results: list[dict[str, Any]], pool: str | None = None) -> float:
    rs = [r for r in results if pool is None or r["pool"] == pool]
    return round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else 0.0


def remember_mistakes(results: list[dict[str, Any]], path: Path | None = None) -> int:
    """Only practice-pool mistakes are remembered (exam stays unseen)."""
    path = path or MISTAKES
    known = set()
    if path.exists():
        known = {json.loads(l)["key"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            if r["correct"] or r["pool"] != "practice" or r["key"] in known:
                continue
            r = {**r, "remembered_at": time.time(),
                 "confident_mistake": r["confidence"] >= 0.6}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            added += 1
    return added


VAL_FLOOR = Path("evo/learning/val_floor.json")


def val_floor(val_now: float, path: Path | None = None) -> float:
    """Lowest validation loss seen so far by self-correction (same 60-batch measure); updates the file."""
    path = path or VAL_FLOOR
    best = val_now
    if path.exists():
        try:
            best = min(best, float(json.loads(path.read_text())["best"]))
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"best": best}))
    return best


def prune_checkpoints(prefix: str, keep: int = 5, directory: Path | None = None) -> None:
    """Keep disk use bounded: only the newest `keep` checkpoints per prefix."""
    directory = directory or WEIGHTS_DIR
    files = sorted(directory.glob(f"{prefix}*.pt"), key=lambda p: p.stat().st_mtime)
    active = ""
    if ACTIVE_WEIGHTS.exists():
        active = json.loads(ACTIVE_WEIGHTS.read_text()).get("checkpoint", "")
    for p in files[:-keep]:
        if str(p) != active:
            p.unlink(missing_ok=True)


REQUIRED = ("question", "answer", "nova_answer", "lang", "kind")


def load_mistakes(path: Path | None = None) -> list[dict[str, Any]]:
    """Remembered mistakes; malformed records are skipped, never fatal."""
    path = path or MISTAKES
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if all(k in rec for k in REQUIRED):
            out.append(rec)
    return out


def correction_sequences(tok: NovaTokenizer, mistakes: list[dict], seq_len: int, repeat: int = 4) -> list[list[int]]:
    docs = []
    for m in mistakes:
        lead = f" {m['prefix']}" if m.get("prefix") else "\n"
        plain = tok.encode_document(f"{m['question']}{lead} {m['answer']}", m["lang"])
        docs.extend([plain] * repeat)
        if tok.has_think:
            docs.append(tok.encode_reasoned(
                m["question"],
                f"Predtým som odpovedala {m['nova_answer']}, ale to bola chyba. Správne je {m['answer']}.",
                f" {m['answer']}",
                m["lang"],
            ))
    random.Random(0).shuffle(docs)
    return list(pack_sequences(docs, seq_len))


# ------------------------------------------------------------------ round

def self_correction_round(
    model,
    tok: NovaTokenizer,
    dataset_dir: Path,
    state: dict | None,
    steps: int = 300,
    seed: int = 7,
    val_tolerance: float = 0.02,
    creator: str = "roppik",
    name: str = "NOVA",
    log=print,
    web_dir: Path | None = Path("data/web_v1"),
) -> dict[str, Any]:
    from nova.data import TokenSequenceDataset
    from nova.training import TrainConfig, train

    rng = random.Random(seed)
    docs = _held_out_documents(tok, dataset_dir / "val.txt")
    questions = (identity_questions(state, creator, name)
                 + language_questions(docs, 60, rng)
                 + cloze_questions(docs, 120, rng))

    model.to("cpu")
    before = run_quiz(model, tok, questions)
    added = remember_mistakes(before)
    mistakes = load_mistakes()
    log(f"quiz before: practice={accuracy(before, 'practice')} exam={accuracy(before, 'exam')} "
        f"new mistakes={added} remembered={len(mistakes)}")

    val = TokenSequenceDataset(dataset_dir / "val.txt", seq_len=128)
    report: dict[str, Any] = {
        "time": time.time(),
        "questions": len(questions),
        "before": {"practice": accuracy(before, "practice"), "exam": accuracy(before, "exam"),
                   "by_kind": {k: accuracy([r for r in before if r["kind"] == k]) for k in ("identity", "self", "language", "cloze")}},
        "new_mistakes": added,
        "remembered_mistakes": len(mistakes),
    }
    if not mistakes:
        report["decision"] = "NOTHING_TO_LEARN"
        return report

    # replay mix: corrections + an equal amount of normal training data
    corr = correction_sequences(tok, mistakes, 128)
    normal = []
    with (dataset_dir / "train.txt").open(encoding="utf-8") as f:
        lines = f.readlines()
    # 3x more normal data than corrections, so general language is not forgotten
    for line in rng.sample(lines, min(len(lines), max(5 * len(corr), 2500))):
        normal.append(list(map(int, line.split())))
    # texts the web learner looked up because of earlier mistakes
    web = []
    if web_dir and Path(web_dir).exists():
        from evo.corpus.sources import web_jsonl
        docs_web = [tok.encode_document(d.text, d.lang) for d in web_jsonl(Path(web_dir), 3_000_000)]
        web = list(pack_sequences(docs_web, 128))[:1000]
    report["replay_web_sequences"] = len(web)
    mix = corr + normal + web
    rng.shuffle(mix)
    tmp = WEIGHTS_DIR / "replay_tmp.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(" ".join(map(str, s)) for s in mix) + "\n", encoding="utf-8")

    from nova.gpu_guard import ensure_vram
    from nova.training import evaluate

    ensure_vram()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loader = torch.utils.data.DataLoader(val, batch_size=32)
    model.to(dev)
    val_before = evaluate(model, loader, dev, max_batches=60)["loss"]

    cfg = TrainConfig(seed=seed, batch_size=8, learning_rate=5e-5, max_steps=steps,
                      eval_every=steps, log_every=steps, device=dev)
    import copy
    original = copy.deepcopy(model.state_dict())
    model, _ = train(model, TokenSequenceDataset(tmp, seq_len=128), val, cfg)
    val_after = evaluate(model, loader, dev, max_batches=60)["loss"]
    tmp.unlink(missing_ok=True)

    model.to("cpu")
    after = run_quiz(model, tok, questions)
    report["after"] = {"practice": accuracy(after, "practice"), "exam": accuracy(after, "exam"),
                       "by_kind": {k: accuracy([r for r in after if r["kind"] == k]) for k in ("identity", "self", "language", "cloze")}}
    report["val_loss_before"] = round(val_before, 4)
    report["val_loss_after"] = round(val_after, 4)
    report["replay"] = {"correction_sequences": len(corr), "normal_sequences": len(normal), "steps": steps}

    better = report["after"]["exam"] > report["before"]["exam"] or (
        report["after"]["exam"] == report["before"]["exam"]
        and report["after"]["practice"] > report["before"]["practice"])
    exam_gain = report["after"]["exam"] - report["before"]["exam"]
    val_rise = (val_after - val_before) / max(val_before, 1e-6)
    report["exam_gain"] = round(exam_gain, 4)
    report["val_rise_pct"] = round(100 * val_rise, 3)
    # small forgetting is acceptable only for a clear exam gain
    allowed = val_tolerance / max(val_before, 1e-6) if exam_gain < 0.05 else 0.01
    # ... and it must not add up: language may sit at most 1 % above the best value
    # this model has ever reached (repeated "+0.9 %" rounds used to undo long training)
    floor = val_floor(val_before)
    report["val_floor"] = round(floor, 4)
    within_floor = val_after <= floor * 1.01
    if better and val_rise <= allowed and within_floor:
        report["decision"] = "KEEP"
    else:
        model.load_state_dict(original)
        report["decision"] = "DISCARD"
    log(f"quiz after:  practice={report['after']['practice']} exam={report['after']['exam']} "
        f"val {val_before:.4f}->{val_after:.4f} => {report['decision']}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--web", action="store_true",
                    help="between rounds, look up topics of mistakes on allowlisted web sources")
    args = ap.parse_args(argv)

    from nova.generate import load_checkpoint_model

    state = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))
    best = state["best_known"]
    dataset_dir = Path(best["dataset"])
    from nova.weights import current_weights, set_active

    ckpt_path = current_weights(best)
    model, ckpt = load_checkpoint_model(ckpt_path)
    tok = NovaTokenizer.load(dataset_dir / "tokenizer.json")
    print(f"weights: {ckpt_path}")

    for i in range(1, args.rounds + 1):
        print(f"--- self-correction round {i}/{args.rounds} ---")
        report = self_correction_round(model, tok, dataset_dir, state, steps=args.steps)
        report["weights_in"] = ckpt_path
        if report.get("decision") == "KEEP":
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            out = WEIGHTS_DIR / f"selfcorrect-{time.strftime('%Y%m%d-%H%M%S')}.pt"
            torch.save({**{k: v for k, v in ckpt.items() if k != "model_state_dict"},
                        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "self_correction": report}, out)
            set_active(str(out), best, "self_correction")
            prune_checkpoints("selfcorrect-", keep=5)
            report["weights_out"] = ckpt_path = str(out)
        ROUND_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ROUND_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
        if args.web and i < args.rounds:
            from evo.learning import web_learner as wl
            ms = load_mistakes()
            topics = wl.plan_topics(ms)
            print(f"web learner: {len(topics)} topics from {len(ms)} mistakes")
            report["web"] = wl.learn_from_web(topics, wl.load_config(), log=lambda *_: None)
            print("web:", report["web"])
        print(json.dumps({k: report.get(k) for k in ("before", "after", "val_loss_before",
                                                      "val_loss_after", "new_mistakes",
                                                      "remembered_mistakes", "decision")},
                         ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

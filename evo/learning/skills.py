"""
Skills: NOVA finds its own weak spots in thinking, and gets material it can learn them from that is right by
construction.

Every task here is made by a program that also knows the answer, so nothing has to be believed: not a teacher, not
a web page, not NOVA itself. That gives two things at once:

  * an exam per kind of thinking - arithmetic, comparing, continuing a sequence, deduction with made-up words (logic,
    not memory), "if A then B" reasoning, ordering by a relation, recalling a fact from the context, reading code -
    always with fresh examples NOVA never trained on. The core is asked to choose among options; the option it finds
    most likely is its answer. Chance level is known, so the score says how far above guessing it is.
  * unlimited correct training material for exactly the weak kind, from other random seeds than the exam (the exam
    items are never in the material).

    python -m evo.learning.skills exam evo/releases/NOVA8-53M-v3/nova_model_fp32.pt      # scores, weakest first
    python -m evo.learning.skills material logic-if --n 20000                            # data/skills_v1/<kind>/

The weak spot (the lowest score above chance) is what the self-education loop works on next.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

EXAM_SEED = 1_000_000          # exam items come from seeds EXAM_SEED + i; material from seeds below it
MATERIAL = Path("data/skills_v1")

NONCE = ["zubor", "kelin", "fapor", "drim", "lobak", "tesuk", "mirva", "gonol", "plesk", "vadur", "kopin", "sevar"]
NAMES = ["Peter", "Jana", "Eva", "Marek", "Zuzana", "Tomáš", "Lucia", "Adam", "Mária", "Filip"]
NAMES_EN = ["Peter", "Anna", "Eve", "Mark", "Susan", "Tom", "Lucy", "Adam", "Mary", "Phil"]
THINGS = ["lopta", "kľúč", "kniha", "jablko", "čiapka", "pero", "šál", "zvonček", "mapa", "lampa"]
THINGS_EN = ["a ball", "a key", "a book", "an apple", "a cap", "a pen", "a scarf", "a bell", "a map", "a lamp"]


def _opts(rng: random.Random, right, wrong: list) -> tuple[list[str], int]:
    seen, out = {str(right)}, [str(right)]
    for w in wrong:
        if str(w) not in seen:
            seen.add(str(w))
            out.append(str(w))
    out = out[:4]
    rng.shuffle(out)
    return out, out.index(str(right))


def arith(rng: random.Random, lang: str) -> dict:
    op = rng.choice("+-*")
    a, b = (rng.randint(2, 60), rng.randint(2, 40)) if op != "*" else (rng.randint(2, 12), rng.randint(2, 12))
    r = a + b if op == "+" else a - b if op == "-" else a * b
    q = f"Koľko je {a} {op} {b}?" if lang == "sk" else f"What is {a} {op} {b}?"
    o, i = _opts(rng, r, [r + 1, r - 1, r + 10, r - 10, a + b if op != "+" else a * b])
    return {"prompt": q, "options": o, "answer": i, "feat": {"op": op, "negative": r < 0, "big": abs(r) >= 50}}


def compare(rng: random.Random, lang: str) -> dict:
    a, b = rng.sample(range(10, 999), 2)
    q = f"Ktoré číslo je väčšie: {a} alebo {b}?" if lang == "sk" else f"Which number is larger: {a} or {b}?"
    o, i = _opts(rng, max(a, b), [min(a, b)])
    return {"prompt": q, "options": o, "answer": i,
            "feat": {"first_larger": a > b, "same_digits": len(str(a)) == len(str(b))}}


def sequence(rng: random.Random, lang: str) -> dict:
    s, d = rng.randint(1, 30), rng.choice([2, 3, 4, 5, 7, 10, -2, -3])
    seq = [s + d * k for k in range(4)]
    nxt = s + d * 4
    q = (f"Doplň ďalšie číslo: {', '.join(map(str, seq))}, ?" if lang == "sk"
         else f"Continue the sequence: {', '.join(map(str, seq))}, ?")
    o, i = _opts(rng, nxt, [nxt + 1, nxt - 1, nxt + d, seq[-1]])
    return {"prompt": q, "options": o, "answer": i, "feat": {"falling": d < 0, "step": d, "crosses_zero": nxt < 0 <= seq[0]}}


def deduction(rng: random.Random, lang: str) -> dict:
    """A syllogism with made-up words: only the logic can answer it, no memory of the world."""
    a, b, c = rng.sample(NONCE, 3)
    x = rng.choice(NAMES if lang == "sk" else NAMES_EN)
    kind = rng.randrange(3)
    if lang == "sk":
        if kind == 0:
            q, r, w = f"Každý {a} je {b}. {x} je {a}. Čo je {x}?", f"{x} je {b}", [f"{x} je {c}", f"{x} nie je {b}"]
        elif kind == 1:
            q, r, w = f"Žiadny {a} nie je {b}. {x} je {a}. Je {x} {b}?", "nie", ["áno", "nedá sa určiť"]
        else:
            q, r, w = f"Každý {a} je {b}. {x} je {b}. Je {x} určite {a}?", "nedá sa určiť", ["áno", "nie"]
    else:
        if kind == 0:
            q, r, w = f"Every {a} is a {b}. {x} is a {a}. What is {x}?", f"{x} is a {b}", [f"{x} is a {c}", f"{x} is not a {b}"]
        elif kind == 1:
            q, r, w = f"No {a} is a {b}. {x} is a {a}. Is {x} a {b}?", "no", ["yes", "cannot tell"]
        else:
            q, r, w = f"Every {a} is a {b}. {x} is a {b}. Is {x} surely a {a}?", "cannot tell", ["yes", "no"]
    o, i = _opts(rng, r, w)
    return {"prompt": q, "options": o, "answer": i, "feat": {"rule": ["all-then", "none", "converse-trap"][kind]}}


def logic_if(rng: random.Random, lang: str) -> dict:
    """If A then B - with lamps, so that nothing but the rule decides (modus ponens, modus tollens, and the trap)."""
    p, q_ = rng.sample("ABCDEFGH", 2)
    kind = rng.randrange(3)
    if lang == "sk":
        rule = f"Ak svieti lampa {p}, svieti aj lampa {q_}."
        if kind == 0:
            q, r, w = f"{rule} Lampa {p} svieti. Svieti lampa {q_}?", "áno", ["nie", "nedá sa určiť"]
        elif kind == 1:
            q, r, w = f"{rule} Lampa {q_} nesvieti. Svieti lampa {p}?", "nie", ["áno", "nedá sa určiť"]
        else:
            q, r, w = f"{rule} Lampa {p} nesvieti. Svieti lampa {q_}?", "nedá sa určiť", ["nie", "áno"]
    else:
        rule = f"If lamp {p} is on, lamp {q_} is on too."
        if kind == 0:
            q, r, w = f"{rule} Lamp {p} is on. Is lamp {q_} on?", "yes", ["no", "cannot tell"]
        elif kind == 1:
            q, r, w = f"{rule} Lamp {q_} is off. Is lamp {p} on?", "no", ["yes", "cannot tell"]
        else:
            q, r, w = f"{rule} Lamp {p} is off. Is lamp {q_} on?", "cannot tell", ["no", "yes"]
    o, i = _opts(rng, r, w)
    return {"prompt": q, "options": o, "answer": i, "feat": {"rule": ["ponens", "tollens", "denying-antecedent"][kind]}}


def ordering(rng: random.Random, lang: str) -> dict:
    """Transitive order: who has the most, from two or three comparisons given in a shuffled order."""
    names = rng.sample(NAMES if lang == "sk" else NAMES_EN, rng.choice([3, 4]))      # names[0] has the most
    facts = [(names[k], names[k + 1]) for k in range(len(names) - 1)]
    rng.shuffle(facts)
    most = rng.random() < 0.5
    if lang == "sk":
        text = " ".join(f"{a} má viac jabĺk ako {b}." for a, b in facts)
        q = f"{text} Kto má {'najviac' if most else 'najmenej'} jabĺk?"
    else:
        text = " ".join(f"{a} has more apples than {b}." for a, b in facts)
        q = f"{text} Who has the {'most' if most else 'fewest'} apples?"
    right = names[0] if most else names[-1]
    o, i = _opts(rng, right, [n for n in names if n != right])
    return {"prompt": q, "options": o, "answer": i, "feat": {"people": len(names), "asks": "most" if most else "least"}}


def recall(rng: random.Random, lang: str) -> dict:
    """A fact given earlier in the context, among distractors, must be found again."""
    n = rng.choice([3, 4, 5, 6])
    boxes = rng.sample(range(1, 40), n)
    things = rng.sample(THINGS if lang == "sk" else THINGS_EN, n)
    k = rng.randrange(n)
    if lang == "sk":
        text = " ".join(f"V skrinke {b} je {t}." for b, t in zip(boxes, things))
        q = f"{text} Čo je v skrinke {boxes[k]}?"
    else:
        text = " ".join(f"Box {b} holds {t}." for b, t in zip(boxes, things))
        q = f"{text} What is in box {boxes[k]}?"
    o, i = _opts(rng, things[k], [t for j, t in enumerate(things) if j != k])
    return {"prompt": q, "options": o, "answer": i, "feat": {"facts": n, "position": "first" if k == 0 else "last" if k == n - 1 else "middle"}}


def code(rng: random.Random, lang: str) -> dict:
    """What a few lines of Python print - answered by running them."""
    a, b, c = rng.randint(1, 9), rng.randint(2, 6), rng.randint(1, 9)
    form = rng.randrange(4)
    if form == 0:
        src = f"x = {a}\ny = x * {b}\nprint(y + {c})"
    elif form == 1:
        src = f"s = 0\nfor i in range({b}):\n    s += {a}\nprint(s)"
    elif form == 2:
        src = f"x = {a}\nif x > {c}:\n    x = x - {c}\nelse:\n    x = x + {b}\nprint(x)"
    else:
        src = f"v = [{a}, {b}, {c}]\nprint(max(v) - min(v))"
    out: list[str] = []
    exec(compile(src, "<skill>", "exec"), {"print": lambda *x: out.append(" ".join(map(str, x))), "range": range, "max": max, "min": min})
    r = int(out[0])
    o, i = _opts(rng, r, [r + 1, r - 1, r + b, a])
    q = src + "\n# " + ("Čo vypíše tento program?" if lang == "sk" else "What does this program print?")
    return {"prompt": q, "options": o, "answer": i, "lang": "py", "feat": {"program": ["arithmetic", "loop", "branch", "list"][form]}}


KINDS: dict[str, Callable[[random.Random, str], dict]] = {
    "arith": arith, "compare": compare, "sequence": sequence, "deduction": deduction, "logic-if": logic_if,
    "ordering": ordering, "recall": recall, "code": code,
}
NAMES_SK = {"arith": "počítanie", "compare": "porovnávanie čísel", "sequence": "postupnosti", "deduction": "logické vyvodzovanie",
            "logic-if": "ak–potom (príčina a následok)", "ordering": "poradie a vzťahy", "recall": "vybavenie z kontextu",
            "code": "čítanie kódu"}


# Forms of the question. 0: the question, the answer on the next line (the first material). 1: "Otázka: ... Odpoveď:"
# - NEVER used in any material, it is the exam's unknown form. 2-4: further forms the material may use, so that what is
# learned is the task and not one way of asking it.
FORMS = {
    1: ("Otázka: {p} Odpoveď:", "Question: {p} Answer:"),
    2: ("Úloha: {p}\nRiešenie:", "Task: {p}\nSolution:"),
    3: ("Na otázku „{p}“ je správna odpoveď", "The correct answer to “{p}” is"),
    4: ("Zadanie – {p} | výsledok:", "Problem – {p} | result:"),
}
TRAIN_FORMS = (0, 2, 3, 4)


def item(kind: str, seed: int, fmt: int = 0) -> dict:
    """One task. fmt 0: the question, the answer on the next line; fmt 1-4: the forms in FORMS (answer after a space)."""
    rng = random.Random(f"{kind}:{seed}")
    lang = "sk" if rng.random() < 0.6 else "en"
    it = KINDS[kind](rng, lang)
    it.setdefault("lang", lang)
    it.update(kind=kind, seed=seed, fmt=fmt)
    if fmt:
        sk = lang == "sk" or (lang == "py" and "Čo vypíše" in it["prompt"])
        it["base_prompt"] = it["prompt"]
        it["prompt"] = FORMS[fmt][0 if sk else 1].format(p=it["prompt"])
        it["sep"] = " "
    return it


def exam_items(kind: str, n: int = 120, fmt: int = 0, start: int = 0) -> list[dict]:
    """The reserved exam items (seeds EXAM_SEED + start ...). The material never contains the first 400; the first
    200 are the selection sample the judge uses, 200-399 the confirmation sample no decision ever looks at."""
    return [item(kind, EXAM_SEED + start + i, fmt) for i in range(n)]


def trained_items(kind: str, n: int = 200) -> list[dict]:
    """The first items of the material itself (seeds 0..) - did it learn what it practised?"""
    exam = {e["prompt"] for e in exam_items(kind, 400)}
    out, seed = [], 0
    while len(out) < n:
        it = item(kind, seed)
        seed += 1
        if it["prompt"] not in exam:
            out.append(it)
    return out


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % interval of an accuracy measured on n items."""
    if n == 0:
        return 0.0, 1.0
    p = right / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(c - h, 3), round(c + h, 3)


def material(kind: str, n: int, out_root: Path = MATERIAL, start: int = 0, forms: tuple[int, ...] = (0,),
             accept=None) -> Path:
    """n correct examples of one kind, never one of the exam's (other seeds, and checked by the question itself).
    With several forms the items take them in turn; form 1 (the exam's unknown form) is never allowed."""
    if 1 in forms:
        raise ValueError("form 1 is the exam's unknown form - it never goes into material")
    exam = {e["prompt"] for e in exam_items(kind, 400)}
    path = Path(out_root) / kind / "material.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    written, seed = 0, start
    with path.open("w", encoding="utf-8") as f:
        while written < n and seed < EXAM_SEED:
            fmt = forms[seed % len(forms)]
            it = item(kind, seed, fmt)
            seed += 1
            if it.get("base_prompt", it["prompt"]) in exam:
                continue
            if accept is not None and not accept(it, seed):
                continue
            answer = it["options"][it["answer"]]
            row = ({"prompt": it["prompt"], "text": answer} if fmt == 0
                   else {"text": it["prompt"] + it["sep"] + answer})          # the whole line exactly as asked
            f.write(json.dumps({"lang": it["lang"], **row, "source": "skills", "topic": kind, "form": fmt},
                               ensure_ascii=False) + "\n")
            written += 1
    return path


def _option_logprobs(model, tok, contexts: list[tuple[str, str]], options: list[list[str]], device: str, batch: int = 64):
    """Per context (text, lang) and each of its options: (sum, mean) log-probability of the option after the text."""
    import torch
    import torch.nn.functional as F

    seqs, owner = [], []
    for n, ((text, lang), opts) in enumerate(zip(contexts, options)):
        p = [tok.lang_id(lang)] + tok.encode(text)
        for j, o in enumerate(opts):
            c = tok.encode(o)
            seqs.append((p + c, len(p), len(c)))
            owner.append((n, j))
    out_s: list[list[tuple[float, float]]] = [[(0.0, 0.0)] * len(o) for o in options]
    model = model.to(device).eval()
    with torch.no_grad():
        for s in range(0, len(seqs), batch):
            chunk = seqs[s:s + batch]
            T = max(len(x[0]) for x in chunk)
            x = torch.zeros(len(chunk), T, dtype=torch.long)
            for r, (ids, _, _) in enumerate(chunk):
                x[r, :len(ids)] = torch.tensor(ids)
            out = model(x.to(device))
            logp = F.log_softmax((out[0] if isinstance(out, (tuple, list)) else out).float(), dim=-1)
            for r, (ids, lp, lc) in enumerate(chunk):
                tgt = torch.tensor(ids[lp:lp + lc], device=device)
                v = logp[r, lp - 1:lp - 1 + lc].gather(1, tgt[:, None])
                n, j = owner[s + r]
                out_s[n][j] = (float(v.sum()), float(v.mean()))
    return out_s


def neutral_context(it: dict) -> str:
    """The answer slot without the question: what the option costs anyway in this form (for calibrated scoring)."""
    sk = it["lang"] == "sk" or (it["lang"] == "py" and "Čo vypíše" in it.get("base_prompt", it["prompt"]))
    if it.get("fmt") == 1:
        return ("Odpoveď:" if sk else "Answer:") + it.get("sep", " ")
    return ("Odpoveď:" if sk else "Answer:") + "\n"


def score_options(model, tok, items: list[dict], device: str = "cpu", batch: int = 64, calibrate: bool = False,
                  margins: list | None = None) -> list[int]:
    """The option the core finds most likely after the prompt - its answer.
    Plain: mean log-probability per token (the exam's fixed rule). calibrate=True: log P(option | question) minus
    log P(option | the answer slot alone) - an option that is likely in any case (a long, predictable phrase) does
    not win just for that; the question has to make it likelier."""
    ctx = [(it["prompt"] + it.get("sep", "\n"), it["lang"]) for it in items]
    opts = [it["options"] for it in items]
    lp = _option_logprobs(model, tok, ctx, opts, device, batch)
    if calibrate:
        base = _option_logprobs(model, tok, [(neutral_context(it), it["lang"]) for it in items], opts, device, batch)
        sc = [[lp[n][j][0] - base[n][j][0] for j in range(len(o))] for n, o in enumerate(opts)]
    else:
        sc = [[lp[n][j][1] for j in range(len(o))] for n, o in enumerate(opts)]
    picks = [max(range(len(x)), key=lambda j: x[j]) for x in sc]
    if margins is not None:                   # how sure the core is of its own answer: best minus second best
        margins.extend(round(sorted(x)[-1] - sorted(x)[-2], 3) if len(x) > 1 else 0.0 for x in sc)
    return picks


def exam(checkpoint: str, tokenizer: str, n: int = 120, kinds: list[str] | None = None, device: str = "auto",
         parts: tuple[str, ...] = ("new",), start: int = 0) -> dict:
    """parts: "new" (fresh items, the material's form), "format" (fresh items, a form never trained),
    "trained" (items of the material itself)."""
    import torch

    from nova.generate import load_checkpoint_model
    from nova.tokenizer import NovaTokenizer

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, _ = load_checkpoint_model(checkpoint)
    model = model.float()
    tok = NovaTokenizer.load(tokenizer)
    out: dict = {"checkpoint": checkpoint, "n": n, "kinds": {}}
    for k in kinds or list(KINDS):
        row: dict = {"name": NAMES_SK[k]}
        for part in parts:
            cal = part.endswith("_cal")
            base_part = part[:-4] if cal else part
            items = (exam_items(k, n, start=start) if base_part == "new" else exam_items(k, n, fmt=1, start=start)
                     if base_part == "format" else trained_items(k, n))
            marg: list = []
            got = score_options(model, tok, items, device, calibrate=cal, margins=marg)
            right = sum(g == it["answer"] for g, it in zip(got, items))
            chance = sum(1 / len(it["options"]) for it in items) / len(items)
            acc = right / len(items)
            picks = [it["options"][g] for g, it in zip(got, items)]
            r = {"accuracy": round(acc, 3), "ci95": wilson(right, len(items)), "chance": round(chance, 3),
                 "most_picked": round(max(picks.count(x) for x in set(picks)) / len(picks), 3),
                 "above_chance": round((acc - chance) / (1 - chance), 3),
                 "correct": "".join("1" if g == it["answer"] else "0" for g, it in zip(got, items)),   # per item, for paired tests
                 "feat": [{"lang": it["lang"], **it.get("feat", {})} for it in items], "margin": marg}
            if part == "new":
                row.update(r)
            else:
                row[part] = r
        out["kinds"][k] = row
    out["weakest"] = sorted(out["kinds"], key=lambda k: out["kinds"][k].get("above_chance", 0))
    return out


def mcnemar(a: str, b: str) -> dict:
    """Paired comparison of two cores on the SAME items (strings of 0/1): exact two-sided McNemar test.
    Only the items where they differ count: b10 = a right, b wrong; b01 = a wrong, b right."""
    from math import comb

    b10 = sum(x == "1" and y == "0" for x, y in zip(a, b))
    b01 = sum(x == "0" and y == "1" for x, y in zip(a, b))
    n, k = b10 + b01, min(b10, b01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_a": b10, "only_b": b01, "diff": round((b01 - b10) / max(1, len(a)), 3), "p": round(p, 5)}


def holm(ps: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm correction: which of several comparisons stay significant when all are tested at once."""
    out, m = {}, len(ps)
    for i, (k, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        if p > alpha / (m - i):
            out.update({k2: False for k2, _ in sorted(ps.items(), key=lambda kv: kv[1])[i:]})
            break
        out[k] = True
    return out


def compare(base_file: str, other_file: str, part: str = "new") -> dict:
    """Two exam results on the same items: per kind the paired difference, McNemar p and Holm over the kinds."""
    a = json.loads(Path(base_file).read_text(encoding="utf-8"))["kinds"]
    b = json.loads(Path(other_file).read_text(encoding="utf-8"))["kinds"]
    rows = {}
    for k in a:
        ra = a[k] if part == "new" else a[k].get(part, {})
        rb = b.get(k, {}) if part == "new" else b.get(k, {}).get(part, {})
        if "correct" in ra and "correct" in rb:
            rows[k] = {"before": ra["accuracy"], "after": rb["accuracy"], **mcnemar(ra["correct"], rb["correct"])}
    sig = holm({k: r["p"] for k, r in rows.items()})
    for k in rows:
        rows[k]["significant_holm"] = sig.get(k, False)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("exam")
    ex.add_argument("checkpoint")
    ex.add_argument("--tokenizer", default=None)
    ex.add_argument("--n", type=int, default=120)
    ex.add_argument("--device", default="auto")
    ex.add_argument("--parts", default="new", help="comma list of new, format, trained")
    ma = sub.add_parser("material")
    ma.add_argument("kind", choices=list(KINDS))
    ma.add_argument("--n", type=int, default=20000)
    ma.add_argument("--forms", default="0", help="comma list of forms for the material (never 1)")
    co = sub.add_parser("compare", help="paired test of two exam files (same items)")
    co.add_argument("base")
    co.add_argument("other")
    co.add_argument("--part", default="new", choices=["new", "format", "trained", "new_cal", "format_cal", "trained_cal"])
    sh = sub.add_parser("show")
    sh.add_argument("kind", choices=list(KINDS))
    sh.add_argument("--n", type=int, default=3)
    args = ap.parse_args(argv)
    if args.cmd == "exam":
        tok = args.tokenizer
        if tok is None:
            st = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))
            tok = str(Path(st["best_known"]["dataset"]) / "tokenizer.json")
        print(json.dumps(exam(args.checkpoint, tok, args.n, device=args.device, parts=tuple(args.parts.split(","))), ensure_ascii=False))
    elif args.cmd == "compare":
        for k, r in compare(args.base, args.other, args.part).items():
            print(f"{k:10} {r['before']:.3f} -> {r['after']:.3f}  only before {r['only_a']:3}  only after {r['only_b']:3}  "
                  f"p {r['p']:.4f}  {'SIGNIFICANT (Holm)' if r['significant_holm'] else ''}")
    elif args.cmd == "material":
        print(material(args.kind, args.n, forms=tuple(int(x) for x in args.forms.split(","))))
    else:
        for it in exam_items(args.kind, args.n):
            print(json.dumps(it, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

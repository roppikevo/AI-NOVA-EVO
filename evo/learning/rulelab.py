"""
RuleLab 01 (the Creator's proposal 15.10): does NOVA learn a RULE better when it is shown the rule and a contrast,
than from questions and answers alone?

Domain: relations and order ("A is taller than B"), with uncertainty. A program builds every problem from a hidden
order of people, so every answer - including "cannot tell" - is right by construction.

Two ways of learning with the same training budget (same steps, same share of the batch):
  A  "qa"     the facts, the question, the answer
  B  "rule"   the facts, the derived order written out ("Poradie: Lea > Eva > Nina"), the answer, and a CONTRAST:
              the same problem with one fact removed or the question turned, and its (different) answer -
              often "cannot tell". The order is the rule the answer follows from; the contrast shows when it stops
              holding.

The exam is fixed in advance and never trained; at the exam the core answers directly (no rule written out), so it
measures what was learned, not what is copied:
  same      new problems like the material
  phrasing  a way of stating a fact never used in the material (a relative clause: "A je vyššia ako B, ktorá je ...")
  words     relation words never used in the material (heavier/lighter, richer/poorer)
  deeper    chains of 5 people (the material has 2-4)
Every part has a share of "cannot tell" problems. Answers are chosen among the options (the names + cannot tell).

    python -m evo.learning.rulelab material qa  --n 60000        # data/rulelab_v1/qa/material.jsonl
    python -m evo.learning.rulelab material rule --n 60000
    python -m evo.learning.rulelab exam <checkpoint> --n 200     # parts as above, per item for paired tests
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path("data/rulelab_v1")
EXAM_SEED = 2_000_000

PEOPLE_SK = [("Peter", "m"), ("Jana", "f"), ("Eva", "f"), ("Marek", "m"), ("Zuzana", "f"), ("Tomáš", "m"), ("Lucia", "f"),
             ("Adam", "m"), ("Mária", "f"), ("Filip", "m"), ("Lea", "f"), ("Nina", "f"), ("Juraj", "m"), ("Klára", "f")]
PEOPLE_EN = ["Peter", "Anna", "Eve", "Mark", "Susan", "Tom", "Lucy", "Adam", "Mary", "Phil", "Lea", "Nina", "George", "Clara"]

# relation words: (more m, more f, less m, less f, most, least) for sk; (more, less, most, least) for en
DIMS_SK = {
    "height": ("vyšší", "vyššia", "nižší", "nižšia", "najvyšší", "najnižší"),
    "age": ("starší", "staršia", "mladší", "mladšia", "najstarší", "najmladší"),
    "speed": ("rýchlejší", "rýchlejšia", "pomalší", "pomalšia", "najrýchlejší", "najpomalší"),
    "weight": ("ťažší", "ťažšia", "ľahší", "ľahšia", "najťažší", "najľahší"),            # exam only
    "wealth": ("bohatší", "bohatšia", "chudobnejší", "chudobnejšia", "najbohatší", "najchudobnejší"),   # exam only
}
DIMS_EN = {
    "height": ("taller", "shorter", "tallest", "shortest"),
    "age": ("older", "younger", "oldest", "youngest"),
    "speed": ("faster", "slower", "fastest", "slowest"),
    "weight": ("heavier", "lighter", "heaviest", "lightest"),
    "wealth": ("richer", "poorer", "richest", "poorest"),
}
TRAIN_DIMS = ("height", "age", "speed")
EXAM_DIMS = ("weight", "wealth")
CANT = {"sk": "nedá sa určiť", "en": "cannot tell"}


def _person(lang: str, p):
    return p[0] if lang == "sk" else p


def fact_text(lang: str, dim: str, a, b, how: str) -> str:
    """'a' is above 'b' on dim. how: 'more' (A je vyšší ako B) or 'less' (B je nižší ako A)."""
    if lang == "sk":
        w = DIMS_SK[dim]
        if how == "more":
            return f"{a[0]} je {w[0] if a[1] == 'm' else w[1]} ako {b[0]}."
        return f"{b[0]} je {w[2] if b[1] == 'm' else w[3]} ako {a[0]}."
    w = DIMS_EN[dim]
    return f"{a} is {w[0]} than {b}." if how == "more" else f"{b} is {w[1]} than {a}."


def chain_text(lang: str, dim: str, a, b, c) -> str:
    """The held-out phrasing: two facts in one sentence with a relative clause."""
    if lang == "sk":
        w = DIMS_SK[dim]
        rel = "ktorý" if b[1] == "m" else "ktorá"
        return f"{a[0]} je {w[0] if a[1] == 'm' else w[1]} ako {b[0]}, {rel} je {w[0] if b[1] == 'm' else w[1]} ako {c[0]}."
    w = DIMS_EN[dim]
    return f"{a} is {w[0]} than {b}, who is {w[0]} than {c}."


def problem(seed: int, part: str = "same", rng: random.Random | None = None, lengths: tuple[int, ...] | None = None) -> dict:
    """A hidden order of k people, facts between neighbours (shuffled, each stated 'more' or 'less'), a question.
    With some chance one fact is left out, so the order splits into two pieces and some questions cannot be told."""
    rng = rng or random.Random(f"rulelab:{part}:{seed}")
    lang = "sk" if rng.random() < 0.6 else "en"
    dims = EXAM_DIMS if part == "words" else TRAIN_DIMS
    dim = rng.choice(dims)
    k = 5 if part == "deeper" else 7 if part == "seven" else rng.choice(list(lengths) if lengths else [3, 3, 4])
    pool = PEOPLE_SK if lang == "sk" else PEOPLE_EN
    order = rng.sample(pool, k)                                  # order[0] is the most
    links = list(range(k - 1))                                   # link i: order[i] > order[i+1]
    # a missing link splits the order into two pieces; each piece keeps at least two people, so everyone still
    # appears in some fact (a person never mentioned would make "cannot tell" unfair)
    gap = rng.randrange(1, k - 2) if k >= 4 and rng.random() < 0.35 else None
    if gap is not None:
        links.remove(gap)
    sentences = []
    if part == "phrasing":
        i = 0
        while i < len(links):
            l = links[i]
            if i + 1 < len(links) and links[i + 1] == l + 1:
                sentences.append(chain_text(lang, dim, order[l], order[l + 1], order[l + 2]))
                i += 2
            else:
                sentences.append(fact_text(lang, dim, order[l], order[l + 1], rng.choice(["more", "less"])))
                i += 1
    else:
        sentences = [fact_text(lang, dim, order[l], order[l + 1], rng.choice(["more", "less"])) for l in links]
    rng.shuffle(sentences)
    names = [_person(lang, p) for p in order]
    q = rng.choice(["most", "least", "pair"])
    if q in ("most", "least"):
        if lang == "sk":
            word = DIMS_SK[dim][4 if q == "most" else 5]
            question = f"Kto je {word}?"
        else:
            question = f"Who is the {DIMS_EN[dim][2 if q == 'most' else 3]}?"
        if gap is None:
            answer = names[0] if q == "most" else names[-1]
        else:
            answer = CANT[lang]                                   # two pieces: the top (or bottom) is not known
    else:
        i, j = sorted(rng.sample(range(k), 2))                    # names[i] is above names[j]
        x, y = (i, j) if rng.random() < 0.5 else (j, i)
        if lang == "sk":
            w = DIMS_SK[dim]
            question = f"Kto je {w[0]}: {names[x]} alebo {names[y]}?"
        else:
            question = f"Who is {DIMS_EN[dim][0]}: {names[x]} or {names[y]}?"
        connected = gap is None or not (i <= gap < j)
        answer = names[i] if connected else CANT[lang]
    options = sorted(set(names), key=lambda n: rng.random()) + [CANT[lang]]
    return {"lang": lang, "dim": dim, "facts": sentences, "question": question, "answer": answer, "options": options,
            "order": names, "gap": gap, "q": q, "prompt": " ".join(sentences) + " " + question,
            "cannot": answer == CANT[lang], "part": part, "seed": seed}


def order_line(p: dict) -> str:
    """The rule the answer follows from, written out: the order (or its pieces when a link is missing)."""
    names, gap = p["order"], p["gap"]
    pieces = [names] if gap is None else [names[:gap + 1], names[gap + 1:]]
    head = "Poradie" if p["lang"] == "sk" else "Order"
    joined = (" | ".join(" > ".join(x) for x in pieces))
    note = (" (dve oddelené časti)" if p["lang"] == "sk" else " (two separate pieces)") if gap is not None else ""
    return f"{head}: {joined}{note}"


def contrast(p: dict, rng: random.Random) -> dict:
    """The same people and facts with one change that changes the answer: a link removed (often -> cannot tell), or,
    if a link is already missing, the question turned to the other end."""
    lang = p["lang"]
    names = p["order"]
    asked = [n for n in names if n in p["question"]]
    if p["gap"] is None:
        options = []
        for drop, sentence in enumerate(p["facts"]):
            g = next(i for i in range(len(names) - 1) if names[i] in sentence and names[i + 1] in sentence)
            left, right = names[:g + 1], names[g + 1:]
            fair = all(len(x) >= 2 or (p["q"] == "pair" and x[0] in asked) for x in (left, right))
            if fair:
                options.append((drop, g))
        if options:
            drop, g = rng.choice(options)
            q = dict(p)
            q["facts"] = [f for i, f in enumerate(p["facts"]) if i != drop]
            q["gap"] = g
            if p["q"] in ("most", "least"):
                q["answer"] = CANT[lang]
            else:
                i, j = sorted(names.index(n) for n in asked)
                q["answer"] = names[i] if not (i <= g < j) else CANT[lang]
            q["prompt"] = " ".join(q["facts"]) + " " + p["question"]
            q["cannot"] = q["answer"] == CANT[lang]
            return q
    if p["q"] in ("most", "least"):                                     # the question turned to the other end
        q = dict(p)
        other = "least" if p["q"] == "most" else "most"
        if lang == "sk":
            q["question"] = f"Kto je {DIMS_SK[p['dim']][4 if other == 'most' else 5]}?"
        else:
            q["question"] = f"Who is the {DIMS_EN[p['dim']][2 if other == 'most' else 3]}?"
        q["q"] = other
        if p["gap"] is None:
            q["answer"] = names[0] if other == "most" else names[-1]
        q["prompt"] = " ".join(p["facts"]) + " " + q["question"]
        q["cannot"] = q["answer"] == CANT[lang]
        return q
    return problem(p["seed"] + 7_777_777, p["part"], rng)              # a fresh sibling (rare)


def material(method: str, n: int, root: Path = ROOT, accept=None, lengths: tuple[int, ...] | None = None,
             name: str | None = None) -> Path:
    exam = {problem(EXAM_SEED + i, part)["prompt"] for part in ("same", "phrasing", "words", "deeper") for i in range(400)}
    path = Path(root) / (name or method) / "material.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    written, seed = 0, 0
    with path.open("w", encoding="utf-8") as f:
        while written < n:
            rng = random.Random(f"rulelab-mat:{seed}")
            p = problem(seed, "same", rng, lengths)
            seed += 1
            if p["prompt"] in exam:
                continue
            if accept is not None and not accept({"lang": p["lang"], "question": p["q"], "cannot_tell": p["cannot"],
                                                  "people": len(p["order"]), "dim": p["dim"]}, seed):
                continue
            if method == "qa":
                rows = [{"prompt": p["prompt"], "text": p["answer"]}]
            else:
                c = contrast(p, rng)
                lab = "Ak sa zmení jedna vec:" if p["lang"] == "sk" else "If one thing changes:"
                text = (f"{order_line(p)}\n{p['answer']}\n{lab} {c['prompt']}\n{order_line(c)}\n{c['answer']}")
                rows = [{"prompt": p["prompt"], "text": text}]
            for r in rows:
                f.write(json.dumps({"lang": p["lang"], **r, "source": "rulelab", "topic": f"rulelab-{method}"},
                                   ensure_ascii=False) + "\n")
            written += 1
    return path


PARTS = ("same", "phrasing", "words", "deeper")


def exam_items(part: str, n: int = 200, start: int = 0) -> list[dict]:
    """Reserved items: the material never contains the first 400; 0-199 selection, 200-399 confirmation."""
    return [problem(EXAM_SEED + start + i, part) for i in range(n)]


def exam(checkpoint: str, tokenizer: str, n: int = 200, device: str = "auto", start: int = 0) -> dict:
    import torch

    from evo.learning.skills import score_options, wilson
    from nova.generate import load_checkpoint_model
    from nova.tokenizer import NovaTokenizer

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, _ = load_checkpoint_model(checkpoint)
    model = model.float()
    tok = NovaTokenizer.load(tokenizer)
    out: dict = {"checkpoint": checkpoint, "n": n, "kinds": {}}
    for part in PARTS:
        items = exam_items(part, n, start)
        for it in items:
            it["answer"] = it["options"].index(it["answer"])
        marg: list = []
        got = score_options(model, tok, items, device, margins=marg)
        ok = [g == it["answer"] for g, it in zip(got, items)]
        cant = [o for o, it in zip(ok, items) if it["cannot"]]
        sure = [o for o, it in zip(ok, items) if not it["cannot"]]
        chance = sum(1 / len(it["options"]) for it in items) / len(items)
        out["kinds"][part] = {"accuracy": round(sum(ok) / len(ok), 3), "ci95": wilson(sum(ok), len(ok)),
                              "chance": round(chance, 3), "cannot_tell": round(sum(cant) / max(1, len(cant)), 3),
                              "decidable": round(sum(sure) / max(1, len(sure)), 3), "n_cannot": len(cant),
                              "said_cannot": round(sum(it["options"][g] == CANT[it["lang"]] for g, it in zip(got, items)) / len(items), 3),
                              "correct": "".join("1" if o else "0" for o in ok),
                              "feat": [{"lang": it["lang"], "question": it["q"], "cannot_tell": it["cannot"], "people": len(it["order"]),
                                        "dim": it["dim"]} for it in items], "margin": marg}
    return out


def parse_order(line: str) -> list[list[str]] | None:
    """'Poradie: A > B | C > D (…)' -> [[A, B], [C, D]]; None if the line is not an order."""
    for head in ("Poradie:", "Order:"):
        if line.strip().startswith(head):
            body = line.strip()[len(head):].split("(")[0]
            return [[n.strip() for n in piece.split(">") if n.strip()] for piece in body.split("|") if piece.strip()]
    return None


def exam_self(checkpoint: str, tokenizer: str, n: int = 200, device: str = "auto",
              parts: tuple[str, ...] = PARTS) -> dict:
    """The fair exam for the rule-trained core: it first WRITES the order itself (greedy, one line), an independent
    checker compares it with the true order, and only then the answer is chosen - after its own line, the way it
    learned. Measures the discoverer (is its order right?) and the use of it (is the answer right, given its order?)."""
    import torch

    from evo.learning.skills import score_options, wilson
    from nova.generate import generate, load_checkpoint_model
    from nova.tokenizer import NovaTokenizer

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, _ = load_checkpoint_model(checkpoint)
    model = model.float().to(device).eval()
    tok = NovaTokenizer.load(tokenizer)
    out: dict = {"checkpoint": checkpoint, "n": n, "kinds": {}}
    for part in parts:
        items = exam_items(part, n)
        lines, right_order, wellformed = [], [], []
        for it in items:
            gen = generate(model, tok, it["prompt"] + "\n", it["lang"], max_new_tokens=48, device=str(device))
            line = gen[len(it["prompt"]) + 1:].split("\n")[0]
            lines.append(line)
            got = parse_order(line)
            truth = parse_order(order_line(it))
            wellformed.append(got is not None)
            right_order.append(got is not None and sorted(map(tuple, got)) == sorted(map(tuple, truth)))
            it["prompt"] = it["prompt"] + "\n" + line
            it["answer"] = it["options"].index(it["answer"])
        picks = score_options(model, tok, items, device)
        ok = [g == it["answer"] for g, it in zip(picks, items)]
        with_right = [o for o, r in zip(ok, right_order) if r]
        with_wrong = [o for o, r in zip(ok, right_order) if not r]
        out["kinds"][part] = {"accuracy": round(sum(ok) / len(ok), 3), "ci95": wilson(sum(ok), len(ok)),
                              "order_wellformed": round(sum(wellformed) / len(items), 3),
                              "order_right": round(sum(right_order) / len(items), 3),
                              "answer_if_order_right": round(sum(with_right) / max(1, len(with_right)), 3),
                              "answer_if_order_wrong": round(sum(with_wrong) / max(1, len(with_wrong)), 3),
                              "examples": lines[:3], "correct": "".join("1" if o else "0" for o in ok)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ma = sub.add_parser("material")
    ma.add_argument("method", choices=["qa", "rule"])
    ma.add_argument("--n", type=int, default=60000)
    ma.add_argument("--lengths", default=None, help="comma list of chain lengths for the material (default 3,3,4)")
    ma.add_argument("--name", default=None)
    ex = sub.add_parser("exam")
    ex.add_argument("checkpoint")
    ex.add_argument("--tokenizer", default=None)
    ex.add_argument("--n", type=int, default=200)
    ex.add_argument("--device", default="auto")
    es = sub.add_parser("exam-self", help="the core writes the order itself, then answers")
    es.add_argument("checkpoint")
    es.add_argument("--tokenizer", default=None)
    es.add_argument("--n", type=int, default=200)
    es.add_argument("--device", default="auto")
    es.add_argument("--parts", default=",".join(PARTS), help="comma list; 'seven' = chains of 7 (never in any material)")
    sh = sub.add_parser("show")
    sh.add_argument("part", choices=list(PARTS) + ["rule"])
    sh.add_argument("--n", type=int, default=3)
    args = ap.parse_args(argv)
    if args.cmd == "material":
        print(material(args.method, args.n, lengths=tuple(int(x) for x in args.lengths.split(",")) if args.lengths else None,
                       name=args.name))
    elif args.cmd in ("exam", "exam-self"):
        tok = args.tokenizer
        if tok is None:
            st = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))
            tok = str(Path(st["best_known"]["dataset"]) / "tokenizer.json")
        if args.cmd == "exam":
            print(json.dumps(exam(args.checkpoint, tok, args.n, args.device), ensure_ascii=False))
        else:
            print(json.dumps(exam_self(args.checkpoint, tok, args.n, args.device, tuple(args.parts.split(","))), ensure_ascii=False))
    elif args.part == "rule":
        for s in range(args.n):
            rng = random.Random(f"rulelab-mat:{s}")
            p = problem(s, "same", rng)
            c = contrast(p, rng)
            print(p["prompt"], "\n", order_line(p), "\n", p["answer"], "\n  CONTRAST:", c["prompt"], "|", order_line(c), "|", c["answer"], "\n")
    else:
        for p in exam_items(args.part, args.n):
            print(json.dumps({k: p[k] for k in ("prompt", "options", "answer")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

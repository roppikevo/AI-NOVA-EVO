"""
"Uč sa <téma>": topics to learn - from the Creator or found by the director in its own weak spots.

    python -m evo.learning.topics add "fotosyntéza" --lang sk,en        # the Creator asks for a topic
    python -m evo.learning.topics add "spracovanie textu" --kind code    # programming (Python)
    python -m evo.learning.topics list
    python -m evo.learning.topics gaps                                   # what the champion is weakest at now

The loop (run by the director, evo/engine/director.py, one step at a time):

  1. topic      queued in evo/director/topics.json - by the Creator (first) or by the director when it has nothing
                else to try (its weakest language from the judge's web texts, its weakest level of the code exam)
  2. material   a teacher (local llama.cpp model) splits the topic into subtopics and explains each of them in every
                language of the topic (for code: writes programs); for language topics also Wikipedia articles on the
                subtopics. Everything goes through the existing quality checks. 15 % of the documents (chosen by a
                hash of the text, so always the same) are set aside as the topic's exam and never trained on:
                    data/topic_v1/<slug>/material.jsonl     (read by long_train --extra-dirs)
                    data/topic_exam/<slug>.jsonl            (only measured)
                The main dataset is never changed.
  3. training   the champion is trained on its usual mix plus the topic material (a share of every batch comes from
                the topic: long_train --focus dir:data/topic_v1/<slug>)
  4. decision   the sealed judge decides, exactly as for every other attempt - a topic does not change the rules.
                Beside it, the topic exam is measured before and after (it informs the report, it decides nothing)
  5. report     the topic, where it came from, how much material, the topic exam before/after and the verdict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable

STORE = Path("evo/director/topics.json")
MATERIAL = Path("data/topic_v1")
EXAM = Path("data/topic_exam")
LANGS = ("sk", "cs", "pl", "en")
EXAM_SHARE = 0.15
MIN_DOCS = 12                 # less material than this: the topic is marked failed, not trained
GAP_REPEAT_DAYS = 7           # a gap topic is not suggested again within this many days
OPEN = ("queued", "collected", "queued-package")   # queued-package: learned as a domain package, next setting

# what a weak language is taught with: broad everyday knowledge, split into subtopics by the teacher
GAP_TEXT = {
    "sk": "všeobecné vedomosti po slovensky: príroda, veda, dejiny, technika, bežný život",
    "cs": "všeobecné znalosti česky: příroda, věda, dějiny, technika, běžný život",
    "pl": "wiedza ogólna po polsku: przyroda, nauka, historia, technika, życie codzienne",
    "en": "general knowledge in English: nature, science, history, technology, everyday life",
}
SUBTOPIC_PROMPT = {
    "sk": "Rozdeľ tému „{topic}“ na {n} konkrétnych podtém, ktoré sa dajú vysvetliť v krátkom texte. Každú podtému na "
          "samostatný riadok, 3 až 10 slov, bez číslovania a bez ďalšieho textu.",
    "cs": "Rozděl téma „{topic}“ na {n} konkrétních podtémat, která se dají vysvětlit v krátkém textu. Každé na samostatný "
          "řádek, 3 až 10 slov, bez číslování a bez dalšího textu.",
    "pl": "Podziel temat „{topic}” na {n} konkretnych podtematów, które można wyjaśnić w krótkim tekście. Każdy w osobnym "
          "wierszu, 3 do 10 słów, bez numeracji i bez innego tekstu.",
    "en": "Split the topic \"{topic}\" into {n} concrete subtopics that can each be explained in a short text. One per line, "
          "3 to 10 words, no numbering, no other text.",
    "py": "Write {n} short Python programming task ideas about \"{topic}\"{level}. One per line, 4 to 12 words, no numbering, "
          "no other text.",
}
LEVEL_WORDS = {1: " for a beginner", 2: " of intermediate difficulty", 3: " that need a small algorithm or data structure"}


# ------------------------------------------------------------------ the store

def slugify(text: str, taken: set[str] | None = None) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40].strip("-") or "tema"
    base, i = s, 2
    while taken and s in taken:
        s, i = f"{base}-{i}", i + 1
    return s


def load(path: Path = STORE) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else []
    except (OSError, ValueError):
        data = []
    return data if isinstance(data, list) else []


def save(topics: list[dict], path: Path = STORE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(topics, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def get(slug: str, path: Path = STORE) -> dict | None:
    return next((t for t in load(path) if t["slug"] == slug), None)


def update(slug: str, path: Path = STORE, **fields) -> dict | None:
    topics = load(path)
    for t in topics:
        if t["slug"] == slug:
            t.update(fields)
            save(topics, path)
            return t
    return None


def add(text: str, langs: list[str] | None = None, kind: str = "explain", source: str = "creator", path: Path = STORE,
        level: int | None = None, why: str = "", now: float | None = None) -> dict:
    """Queue a topic. The same text still open is not queued twice (the open one is returned)."""
    text = " ".join(text.split())
    if not text:
        raise ValueError("empty topic")
    if kind not in ("explain", "code"):
        raise ValueError(f"kind {kind}: explain or code")
    langs = ["py"] if kind == "code" else [l for l in (langs or ["sk"]) if l in LANGS] or ["sk"]
    topics = load(path)
    for t in topics:
        if t["text"].lower() == text.lower() and t["status"] in OPEN:
            return t
    t = {"slug": slugify(text, {x["slug"] for x in topics}), "text": text, "langs": langs, "kind": kind, "source": source,
         "status": "queued", "created": time.time() if now is None else now, "date": time.strftime("%Y-%m-%d %H:%M")}
    if level:
        t["level"] = int(level)
    if why:
        t["why"] = why
    topics.append(t)
    save(topics, path)
    return t


def next_topic(topics: list[dict]) -> dict | None:
    """The topic to work on: the Creator's first, oldest first; then the director's own."""
    open_ = [t for t in topics if t.get("status") in OPEN]
    open_.sort(key=lambda t: (t.get("source") != "creator", t.get("created", 0)))
    return open_[0] if open_ else None


# ------------------------------------------------------------------ weak spots

def code_levels(solved: list[str] | None) -> dict[int, list[int]]:
    """Exam tasks solved per level: {level: [solved, total]}."""
    from evo.learning.code_tasks import task_bank

    done = set(solved or [])
    out: dict[int, list[int]] = {}
    for t in task_bank():
        if t.pool != "exam":
            continue
        row = out.setdefault(int(t.level), [0, 0])
        row[1] += 1
        row[0] += t.key in done
    return out


def find_gaps(scores: dict | None, solved: list[str] | None = None, topics: list[dict] | None = None,
              now: float | None = None) -> list[dict]:
    """Topics for the champion's weakest spots, weakest first; none that were worked on in the last days."""
    now = time.time() if now is None else now
    recent = {t["text"].lower() for t in (topics or []) if now - t.get("created", 0) < GAP_REPEAT_DAYS * 86400}
    out = []
    langs = (scores or {}).get("web_by_language") or {}
    if langs:
        lang = max(langs, key=langs.get)
        out.append({"text": GAP_TEXT[lang], "langs": [lang], "kind": "explain",
                    "why": f"najslabší jazyk na webových textoch ({lang}: strata {langs[lang]})"})
    if solved is not None:
        levels = code_levels(solved)
        weak = [(s / n, lvl) for lvl, (s, n) in levels.items() if n and s / n < 0.5]
        if weak:
            share, lvl = min(weak)
            s, n = levels[lvl]
            out.append({"text": f"programovanie v Pythone, úroveň {lvl}", "langs": ["py"], "kind": "code", "level": lvl,
                        "why": f"skúška z programovania, úroveň {lvl}: {s}/{n}"})
    return [g for g in out if g["text"].lower() not in recent]


# ------------------------------------------------------------------ material

MAIN_PAGES = {"hlavná stránka", "hlavní strana", "strona główna", "main page"}


def wiki_subtopics(topic: dict, n: int = 12, browser=None, log: Callable[[str], None] = print) -> list[str]:
    """Subtopics from Wikipedia when no teacher answers (the teachers run on the card, which a long training can
    hold for days): the topic's article in its first language, the same article in the other languages (Wikipedia's
    own language links, not a search with a foreign word), and in each the articles it links to most often.
    Every subtopic carries its language ("sk:Chlorofyl") so it is looked up only in that Wikipedia."""
    from collections import Counter
    import urllib.parse

    if browser is None:
        from nova.browse import Browser

        browser = Browser()
    langs = [l for l in topic["langs"] if l in ("sk", "cs", "pl", "en")]
    if not langs:
        return []
    try:
        hits = browser._wikipedia(topic["text"], langs[0], 1)
    except Exception as exc:  # noqa: BLE001 - no network: the teacher's (empty) answer stands
        log(f"{topic['slug']}: Wikipedia {langs[0]} not reachable ({type(exc).__name__})")
        return []
    if not hits:
        return []
    articles = {langs[0]: hits[0]["title"]}
    for lang in langs[1:]:
        try:
            q = urllib.parse.urlencode({"action": "query", "prop": "langlinks", "titles": hits[0]["title"],
                                        "lllang": lang, "format": "json", "redirects": 1})
            pages = browser._get_json(f"https://{langs[0]}.wikipedia.org/w/api.php?{q}")["query"]["pages"]
            for p in pages.values():
                for ll in p.get("langlinks") or []:
                    articles[lang] = ll.get("*") or ll.get("title")
        except Exception:  # noqa: BLE001 - that language is left out
            pass
    per_lang = max(n // len(articles), 3)
    out: list[str] = []
    for lang, title in articles.items():
        url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        try:
            page = browser.open(url)
        except Exception:  # noqa: BLE001
            continue
        out.append(f"{lang}:{title}")
        if page.get("error"):
            continue
        host = f"{lang}.wikipedia.org"
        counts: Counter = Counter()
        first: dict[str, int] = {}
        for i, link in enumerate(page.get("links") or []):
            u = urllib.parse.urlsplit(link["url"])
            if u.hostname != host or not u.path.startswith("/wiki/") or u.query:
                continue
            t = urllib.parse.unquote(u.path[6:]).replace("_", " ")
            if (":" in t or t.lower() == title.lower() or t.lower() in MAIN_PAGES or len(t) < 3 or t[:1].isdigit()):
                continue
            counts[t] += 1
            first.setdefault(t, i)
        out += [f"{lang}:{t}" for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], first[kv[0]]))][:per_lang]
    seen, uniq = set(), []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            uniq.append(t)
    return uniq[:n]


def split_lang(sub: str, langs: list[str]) -> tuple[list[str], str]:
    """"sk:Chlorofyl" -> (["sk"], "Chlorofyl"); a teacher's subtopic stays in every language of the topic."""
    if len(sub) > 3 and sub[2] == ":" and sub[:2] in ("sk", "cs", "pl", "en"):
        return [sub[:2]], sub[3:]
    return langs, sub


def subtopics(client, topic: dict, n: int = 12, browser=None, log: Callable[[str], None] = print) -> list[str]:
    """The teacher splits the topic into concrete subtopics; when it gives nothing usable (not running, an error),
    Wikipedia's links around the topic do (language topics); the topic itself is the last resort."""
    from evo.learning.teacher_corpus import parse_topics

    lang = "py" if topic["kind"] == "code" else topic["langs"][0]
    prompt = SUBTOPIC_PROMPT[lang].format(topic=topic["text"], n=n, level=LEVEL_WORDS.get(topic.get("level"), ""))
    try:
        answer = client.chat([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=900)
        found = parse_topics(answer, set(), n)
    except Exception:
        found = []
    if not found and topic["kind"] != "code":
        found = wiki_subtopics(topic, n, browser, log)
        if found:
            log(f"{topic['slug']}: no teacher answered - {len(found)} subtopics from Wikipedia")
    return found or [topic["text"]]


def exam_part(text: str, share: float = EXAM_SHARE) -> bool:
    """Is this document in the topic's exam? Decided by its content, so the split never changes."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 1000 < share * 1000


def _records(directory: Path) -> list[dict]:
    out = []
    for p in sorted(Path(directory).glob("*.jsonl")) if Path(directory).exists() else []:
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("text") or "").strip() and str(r.get("status", "")).upper() != "REJECTED":
                out.append(r)
    return out


def collect(topic: dict, registry=None, web_cfg: dict | None = None, root: Path = MATERIAL, exam_root: Path = EXAM,
            max_minutes: float = 40, n_sub: int = 30, client_factory: Callable | None = None,
            generate_fn: Callable | None = None, web_fn: Callable | None = None, log: Callable[[str], None] = print,
            browser=None) -> dict:
    """Material for one topic: teacher texts (+ Wikipedia for language topics), split into training and exam."""
    from evo.learning import teacher_corpus as tc

    slug, code = topic["slug"], topic["kind"] == "code"
    raw = Path(root) / slug / "_raw"                     # a subfolder: long_train reads only the files directly in <slug>/
    if registry is None:
        from evo.learning.teacher_registry import TeacherRegistry

        registry = TeacherRegistry.load()
    client_factory = client_factory or tc.TeacherClient
    teacher = registry.choose("code" if code else "language", "py" if code else topic["langs"][0])
    subs = subtopics(client_factory(registry, teacher), topic, n_sub, browser, log)
    log(f"{slug}: teacher {teacher}, {len(subs)} subtopics: {'; '.join(subs)[:300]}")
    pairs = [(lang, t) for s in subs for ls, t in [split_lang(s, topic["langs"])] for lang in ls]
    tasks = [tc.Task(teacher, "code" if code else "explain", lang, t) for lang, t in pairs]
    (generate_fn or tc.generate)(tasks, registry, out_dir=raw / "teacher", client_factory=client_factory, log=log,
                                 max_minutes=max_minutes * (1.0 if code else 0.75))
    if not code:
        from evo.learning import web_learner as wl

        cfg = dict(web_cfg if web_cfg is not None else wl.load_config())
        cfg["max_mb_per_round"] = max(float(cfg.get("max_mb_per_round", 2.0)), 6.0)     # a topic needs more than a round
        try:
            (web_fn or wl.learn_from_web)([wl.Topic(lang, t, f"topic:{slug}") for lang, t in pairs],
                                          cfg, out_dir=raw / "web", log=log)
        except Exception as exc:                          # no network: teacher texts alone
            log(f"{slug}: web collection failed ({type(exc).__name__}: {exc})")
    docs = _records(raw / "teacher") + _records(raw / "web")
    train = [d for d in docs if not exam_part(d["text"])]
    exam = [d for d in docs if exam_part(d["text"])]
    (Path(root) / slug).mkdir(parents=True, exist_ok=True)
    Path(exam_root).mkdir(parents=True, exist_ok=True)
    keep = ("lang", "text", "prompt", "reasoning", "source", "topic", "teacher", "url")
    with (Path(root) / slug / "material.jsonl").open("w", encoding="utf-8") as f:
        for d in train:
            f.write(json.dumps({k: d[k] for k in keep if k in d}, ensure_ascii=False) + "\n")
    with (Path(exam_root) / f"{slug}.jsonl").open("w", encoding="utf-8") as f:
        for d in exam:
            f.write(json.dumps({k: d[k] for k in keep if k in d}, ensure_ascii=False) + "\n")
    return {"teacher": teacher, "subtopics": subs, "train_docs": len(train), "exam_docs": len(exam),
            "chars": sum(len(d["text"]) for d in docs)}


# ------------------------------------------------------------------ the topic exam

def exam_loss(checkpoint: str | Path, slug: str, tokenizer: str | Path, exam_root: Path = EXAM, device: str = "auto",
              max_rows: int = 400) -> float | None:
    """Mean loss per token on the topic's exam documents (never trained on). None when there is no exam."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from evo.corpus.build_text_v1 import encode_doc
    from evo.corpus.sources import Document
    from nova.generate import load_checkpoint_model
    from nova.tokenizer import NovaTokenizer, pack_sequences

    path = Path(exam_root) / f"{slug}.jsonl"
    docs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []
    if not docs:
        return None
    tok = NovaTokenizer.load(tokenizer)
    enc = [encode_doc(tok, Document(d["lang"], "teacher", d["text"], prompt=d.get("prompt", ""))) for d in docs]
    rows = np.array(list(pack_sequences(enc, 128))[:max_rows], dtype=np.int64)
    if not len(rows):
        return None
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model, _ = load_checkpoint_model(checkpoint)
    model = model.float().to(device).eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            x = torch.from_numpy(rows[i:i + 32]).to(device)
            out = model(x)
            logits = (out[0] if isinstance(out, (tuple, list)) else out).float()
            total += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum"))
            count += x[:, 1:].numel()
    return round(total / count, 4)


# ------------------------------------------------------------------ the command

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="queue a topic")
    a.add_argument("text")
    a.add_argument("--lang", default="sk", help="comma list of sk,cs,pl,en (ignored for --kind code)")
    a.add_argument("--kind", default="explain", choices=["explain", "code"])
    sub.add_parser("list")
    g = sub.add_parser("gaps", help="the champion's weakest spots now (nothing is queued)")
    g.add_argument("--state", default="evo/director/state.json")
    c = sub.add_parser("collect", help="collect the material of a queued topic (run by the director)")
    c.add_argument("slug")
    c.add_argument("--max-minutes", type=float, default=40)
    e = sub.add_parser("exam", help="the topic exam before and after (run by the director)")
    e.add_argument("slug")
    e.add_argument("--before", required=True)
    e.add_argument("--after", required=True)
    e.add_argument("--tokenizer", default="")
    for p in (a, sub.choices["list"], g, c, e):
        p.add_argument("--store", default=str(STORE))
    args = ap.parse_args(argv)
    store = Path(getattr(args, "store", STORE))

    if args.cmd == "add":
        t = add(args.text, args.lang.split(","), args.kind, "creator", store)
        print(f"queued: {t['slug']} ({t['kind']}, {','.join(t['langs'])}) - the director takes it at its next step")
    elif args.cmd == "list":
        for t in load(store):
            ex = t.get("exam") or {}
            print(f"{t['date']}  {t['slug']:<32} {t['source']:<8} {t['status']:<10} docs {t.get('train_docs', '-')}"
                  + (f"  exam {ex.get('before')} -> {ex.get('after')}" if ex else "")
                  + (f"  verdict {t['verdict']}" if t.get("verdict") else ""))
    elif args.cmd == "gaps":
        st = json.loads(Path(args.state).read_text(encoding="utf-8")) if Path(args.state).exists() else {}
        for gp in find_gaps((st.get("champion") or {}).get("scores"), None, load(store)):
            print(f"{gp['text']}  ({gp['why']})")
    elif args.cmd == "collect":
        t = get(args.slug, store)
        if t is None:
            raise SystemExit(f"no topic {args.slug}")
        try:
            r = collect(t, max_minutes=args.max_minutes)
        except Exception as exc:
            update(args.slug, store, status="failed", note=f"collection failed: {type(exc).__name__}: {exc}"[:300])
            raise
        ok = r["train_docs"] >= MIN_DOCS and r["exam_docs"] >= 1
        update(args.slug, store, status="collected" if ok else "failed", collected=time.time(),
               **{k: r[k] for k in ("teacher", "subtopics", "train_docs", "exam_docs", "chars")},
               **({} if ok else {"note": f"too little material: {r['train_docs']} documents for training, {r['exam_docs']} for the exam"}))
        print(json.dumps(r, ensure_ascii=False))
    elif args.cmd == "exam":
        tokz = args.tokenizer or str(Path(args.before).parent / "tokenizer.json")
        before = exam_loss(args.before, args.slug, tokz)
        after = exam_loss(args.after, args.slug, tokz)
        ex = {"before": before, "after": after,
              "percent": round((after - before) / before * 100, 2) if before and after else None}
        update(args.slug, store, exam=ex)
        print(json.dumps(ex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

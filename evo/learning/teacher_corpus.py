"""
Teacher corpus: NOVA learns from all enabled teacher models.

Every enabled teacher gets tasks from its own domains/languages
(explanations in sk/cs/pl/en, Python and Rust tasks with solutions).
Requests are grouped per teacher so the router (one model in memory)
swaps models as rarely as possible.

Content policy (Creator decision, see teachers.json "content_policy"):
NO content/topic filtering. NOVA must not learn refusals, so teacher
refusals and moralizing meta-talk are dropped. Only quality is checked.

Every answer passes a quality gate before it is kept:
    - no refusal / meta talk, not too short, not repetitive
    - expected script/language markers for natural languages
    - Python code must parse (ast); Rust code must have balanced braces
Accepted items are stored as EXPERIMENTAL in data/teacher_v1/<teacher>.jsonl
and can later be mixed into the text corpus (source "teacher").

    python -m evo.learning.teacher_corpus --plan          # show plan, loads nothing
    python -m evo.learning.teacher_corpus --per-task 2    # generate
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from evo.learning.teacher_registry import TeacherClient, TeacherRegistry

OUT_DIR = Path("data/teacher_v1")

# Hardware-light defaults: small teachers only, time box, capped reasoning.
LIGHT_TEACHERS = ("Qwen3.5-9B", "Qwen3-14B")
MAX_REASONING_CHARS = 4000
BRIEF_THINKING = " Think briefly before answering (at most a few sentences of reasoning)."

LANG_NAMES = {"sk": "slovenčine", "cs": "češtině", "pl": "po polsku", "en": "English"}

EXPLAIN_PROMPT = {
    "sk": "Vysvetli po slovensky, jasne a vecne, v 2 až 4 odsekoch: {topic}. Píš iba text vysvetlenia.",
    "cs": "Vysvětli česky, jasně a věcně, ve 2 až 4 odstavcích: {topic}. Piš pouze text vysvětlení.",
    "pl": "Wyjaśnij po polsku, jasno i rzeczowo, w 2–4 akapitach: {topic}. Napisz tylko tekst wyjaśnienia.",
    "en": "Explain in English, clearly and factually, in 2-4 paragraphs: {topic}. Write only the explanation.",
}

CODE_PROMPT = {
    "py": "Write a correct, idiomatic Python 3 solution: {topic}. Reply with one ```python code block with a docstring and a short usage example, then one sentence of explanation.",
    "rs": "Write a correct, idiomatic Rust solution: {topic}. Reply with one ```rust code block including a main() example, then one sentence of explanation.",
}

TOPICS_LANGUAGE = [
    "ako funguje fotosyntéza", "prečo je obloha modrá", "čo je gravitácia",
    "ako vzniká dúha", "čo je demokracia", "história písma",
    "ako funguje očkovanie", "čo je algoritmus", "kolobeh vody v prírode",
    "prečo sa striedajú ročné obdobia", "čo je inflácia", "ako funguje internet",
    "čo je umelá inteligencia", "ako vznikajú hory", "čo je atóm",
    "ako funguje pamäť človeka", "prečo je spánok dôležitý", "čo je ekosystém",
    "ako funguje elektrina v domácnosti", "čo je binárna sústava",
]

TOPICS_CODE = [
    "check whether a string is a palindrome", "compute the n-th Fibonacci number iteratively",
    "count word frequencies in a text", "binary search in a sorted list",
    "merge two sorted lists", "parse a CSV line with quoted fields",
    "implement a stack with push, pop and peek", "find duplicates in a list",
    "reverse the words of a sentence", "compute the greatest common divisor",
    "validate balanced parentheses", "flatten a nested list",
    "a simple LRU cache", "convert Roman numerals to integers",
    "matrix multiplication", "remove duplicate items while keeping order",
]

# Refusals / moralizing meta-talk: dropped so NOVA never learns to refuse.
REFUSAL = re.compile(
    r"(as an ai|as a language model|i cannot|i can't (help|assist|provide)|"
    r"i'm sorry, but|i am not able to|it is not appropriate|i must decline|"
    r"ako ai|ako jazykový model|nemôžem (ti )?pomôcť|nemohu (ti )?pomoci|"
    r"jako jazykový model|nie mogę pomóc|jako model językowy)",
    re.I,
)
# Distinctive letters + distinctive function words per language
# (letters/words shared between languages, like á or "je", are excluded).
LANG_MARKERS = {
    "sk": re.compile(r"[ľĺŕôä]|\b(sa|ktor\w*|sú|alebo|aj|iba|pretože|tiež)\b", re.I),
    "cs": re.compile(r"[ěřů]|\b(se|kter\w*|jsou|nebo|také|pouze|protože|jako)\b", re.I),
    "pl": re.compile(r"[ąęłńśźżć]|\b(się|jest|któr\w*|oraz|nie|są|jak)\b", re.I),
    "en": re.compile(r"\b(the|and|is|of|to|in|which|are)\b", re.I),
}
MIN_MARKERS = 3


def detect_language(text: str) -> str | None:
    counts = {lang: len(rx.findall(text)) for lang, rx in LANG_MARKERS.items()}
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    return lang if n >= MIN_MARKERS else None

CODE_BLOCK = re.compile(r"```(?:python|py|rust|rs)?\s*\n(.*?)```", re.S)


@dataclass
class Task:
    teacher: str
    kind: str      # explain | code
    lang: str      # sk cs pl en py rs
    topic: str

    @property
    def prompt(self) -> str:
        if self.kind == "explain":
            return EXPLAIN_PROMPT[self.lang].format(topic=self.topic)
        return CODE_PROMPT[self.lang].format(topic=self.topic)


# ------------------------------------------------------------------ planning

TOPICS_FILE = OUT_DIR / "topics.json"
TOPIC_PROMPT = {
    "language": "Napíš {n} krátkych tém na vysvetlenie pre zvedavého človeka (príroda, veda, technika, dejiny, bežný život, "
                "jazyk, zdravie, hospodárstvo). Každú tému na samostatný riadok, 3 až 8 slov, bez číslovania a bez ďalšieho textu. "
                "Nesmú sa opakovať tieto: {known}",
    "code": "Write {n} short programming task ideas for a beginner-to-intermediate programmer (strings, lists, numbers, "
            "dictionaries, simple algorithms, small data structures). One task per line, 4 to 10 words, no numbering, "
            "no other text. Do not repeat these: {known}",
}


def load_topics(path: Path | None = None) -> dict[str, list[str]]:
    """Topics the teachers invented themselves (added to the built-in ones)."""
    path = path or TOPICS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        data = {}
    return {"language": list(data.get("language", [])), "code": list(data.get("code", []))}


def parse_topics(answer: str, known: set[str], limit: int) -> list[str]:
    out: list[str] = []
    for line in strip_think(answer).splitlines():
        t = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip('"„“').rstrip(".")
        if 8 <= len(t) <= 90 and 2 <= len(t.split()) <= 12 and t.lower() not in known and not REFUSAL.search(t):
            out.append(t)
            known.add(t.lower())
        if len(out) >= limit:
            break
    return out


def expand_topics(client, n: int, path: Path | None = None, log: Callable[[str], None] = print) -> dict[str, int]:
    """Ask a teacher for new topics, so the corpus does not run dry. Returns how many were added per kind."""
    path = path or TOPICS_FILE
    topics = load_topics(path)
    added = {}
    for kind, builtin in (("language", TOPICS_LANGUAGE), ("code", TOPICS_CODE)):
        known = {t.lower() for t in builtin + topics[kind]}
        sample = "; ".join(sorted(known)[:25])
        try:
            answer = client.chat([{"role": "user", "content": TOPIC_PROMPT[kind].format(n=n, known=sample)}], max_tokens=1536)
        except Exception as exc:
            log(f"new topics ({kind}): {type(exc).__name__}: {exc}")
            added[kind] = 0
            continue
        new = parse_topics(answer, known, n)
        topics[kind] += new
        added[kind] = len(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(topics, indent=1, ensure_ascii=False), encoding="utf-8")
    return added


def plan(
    registry: TeacherRegistry,
    per_task: int = 1,
    seed: int = 7,
    only: tuple[str, ...] | None = None,
    langs: tuple[str, ...] | None = None,
) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    own = load_topics()
    topics_lang = list(TOPICS_LANGUAGE) + [t for t in own["language"] if t not in TOPICS_LANGUAGE]
    topics_code = list(TOPICS_CODE) + [t for t in own["code"] if t not in TOPICS_CODE]
    # topics NOVA looked up on the web because of its own mistakes
    for f in Path("data/web_v1").glob("*.jsonl") if Path("data/web_v1").exists() else []:
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                title = json.loads(line).get("title")
            except json.JSONDecodeError:
                continue
            if title and title not in topics_lang:
                topics_lang.append(title)
    for t in registry.enabled():
        if only and t.name not in only:
            continue
        if "language" in t.domains:
            for lang in ("sk", "cs", "pl", "en"):
                if lang in t.languages and (not langs or lang in langs):
                    for topic in rng.sample(topics_lang, min(per_task, len(topics_lang))):
                        tasks.append(Task(t.name, "explain", lang, topic))
        if "code" in t.domains:
            for lang in ("py", "rs"):
                if lang in t.languages and (not langs or lang in langs):
                    for topic in rng.sample(topics_code, min(per_task, len(topics_code))):
                        tasks.append(Task(t.name, "code", lang, topic))
    # grouped per teacher -> minimal model swaps on the router
    return sorted(tasks, key=lambda x: (registry.teachers[x.teacher].priority, x.teacher))


# --------------------------------------------------------------- quality gate

def _repetitive(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 20:
        return False
    grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return grams.most_common(1)[0][1] > max(3, len(words) // 40)


def quality_check(task: Task, answer: str) -> tuple[bool, str]:
    text = (answer or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if len(text) < 200:
        return False, "too_short"
    if REFUSAL.search(text[:300]):
        return False, "refusal_or_meta"
    if _repetitive(text):
        return False, "repetitive"
    if task.kind == "explain":
        if detect_language(text) != task.lang:
            return False, "wrong_language"
        return True, "ok"
    blocks = CODE_BLOCK.findall(text)
    if not blocks:
        return False, "no_code_block"
    code = blocks[0]
    if task.lang == "py":
        try:
            ast.parse(code)
        except SyntaxError:
            return False, "python_syntax_error"
    if task.lang == "rs":
        if code.count("{") != code.count("}") or "fn " not in code:
            return False, "rust_structure"
    return True, "ok"


def strip_think(answer: str) -> str:
    return re.sub(r"<think>.*?</think>", "", answer or "", flags=re.S).strip()


# ----------------------------------------------------------------- generation

def generate(
    tasks: list[Task],
    registry: TeacherRegistry,
    out_dir: Path = OUT_DIR,
    client_factory: Callable[[TeacherRegistry, str], TeacherClient] = TeacherClient,
    log: Callable[[str], None] = print,
    max_minutes: float | None = None,
    max_tokens: int = 3072,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = defaultdict(Counter)
    clients: dict[str, TeacherClient] = {}
    deadline = time.time() + max_minutes * 60 if max_minutes else None
    done = _done_keys(out_dir)

    for task in tasks:
        if deadline and time.time() > deadline:
            log("time box reached - stopping")
            break
        key = _task_key(task)
        if key in done:
            stats[task.teacher]["cached"] += 1
            continue
        client = clients.get(task.teacher) or client_factory(registry, task.teacher)
        clients[task.teacher] = client
        started = time.time()
        try:
            msgs = [{"role": "user", "content": task.prompt + BRIEF_THINKING}]
            if hasattr(client, "chat_full"):
                answer, reasoning = client.chat_full(msgs, max_tokens=max_tokens)
            else:
                answer, reasoning = client.chat(msgs), ""
        except Exception as exc:
            stats[task.teacher]["error"] += 1
            log(f"[{task.teacher}] {task.lang} ERROR {type(exc).__name__}: {exc}")
            continue
        ok, reason = quality_check(task, answer)
        if not ok and reason == "too_short" and reasoning:
            reason = "reasoning_used_all_tokens"
        stats[task.teacher][reason] += 1
        log(f"[{task.teacher}] {task.kind}/{task.lang} {reason} ({time.time() - started:.0f}s)")
        if not ok:
            continue
        text = strip_think(answer)
        if not reasoning:
            m = re.search(r"<think>(.*?)</think>", answer or "", re.S)
            reasoning = m.group(1).strip() if m else ""
        if len(reasoning) > MAX_REASONING_CHARS or REFUSAL.search(reasoning[:300]):
            reasoning = ""  # keep only short, clean reasoning
        record = {
            "key": key,
            "reasoning": reasoning,
            "id": hashlib.sha256(text.encode()).hexdigest()[:16],
            "status": "EXPERIMENTAL",
            "teacher": task.teacher,
            "kind": task.kind,
            "lang": task.lang,
            "topic": task.topic,
            "prompt": task.prompt,
            "text": text,
            "created": time.time(),
        }
        with (out_dir / f"{task.teacher}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {k: dict(v) for k, v in stats.items()}


def _task_key(task: Task) -> str:
    return hashlib.sha256(
        f"{task.teacher}|{task.kind}|{task.lang}|{task.topic}".encode()
    ).hexdigest()[:16]


def _done_keys(out_dir: Path) -> set[str]:
    keys: set[str] = set()
    for f in out_dir.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                keys.add(json.loads(line).get("key", ""))
            except json.JSONDecodeError:
                pass
    return keys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true", help="only print the plan")
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7, help="different seed = different topics")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--teachers", default=",".join(LIGHT_TEACHERS),
                    help="comma list, or 'all' (default: small teachers only)")
    ap.add_argument("--max-minutes", type=float, default=30.0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--langs", default="", help="only these languages, e.g. sk,py (default: all)")
    ap.add_argument("--new-topics", type=int, default=0,
                    help="first ask the (first) teacher for this many new topics of each kind")
    args = ap.parse_args(argv)

    registry = TeacherRegistry.load()
    only = None if args.teachers == "all" else tuple(args.teachers.split(","))
    if args.new_topics > 0 and not args.plan:
        first = (only or tuple(t.name for t in registry.enabled()))[0]
        try:
            print("new topics:", expand_topics(TeacherClient(registry, first), args.new_topics))
        except Exception as exc:
            print(f"new topics skipped: {type(exc).__name__}: {exc}")
    tasks = plan(registry, args.per_task, seed=args.seed, only=only,
                 langs=tuple(x for x in args.langs.split(",") if x) or None)

    summary = Counter((t.teacher, t.kind, t.lang) for t in tasks)
    print(f"{len(tasks)} tasks, teachers in order: "
          f"{list(dict.fromkeys(t.teacher for t in tasks))}")
    for (teacher, kind, lang), n in sorted(summary.items()):
        print(f"  {teacher:<22} {kind:<8} {lang}  x{n}")

    if args.plan:
        print("PLAN ONLY - no model was loaded")
        return 0

    stats = generate(tasks, registry, Path(args.out),
                     max_minutes=args.max_minutes, max_tokens=args.max_tokens)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

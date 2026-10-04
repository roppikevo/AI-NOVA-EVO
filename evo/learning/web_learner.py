"""
Web learner: NOVA looks up on the internet what it got wrong.

    mistakes.jsonl  ->  topics  ->  allowlisted sources  ->  quality gate
                    ->  data/web_v1/<lang>.jsonl (EXPERIMENTAL)
                    ->  used by the next self-correction round (replay)
                        and by the next corpus build (--web-dir)

Rules:
  * only hosts listed in web_sources.json are contacted (allowlist)
  * byte budget per round, delay between requests, timeouts
  * every text: language check, length, repetition, dedupe, source + license
  * nothing goes straight into the weights: it is learned in the next
    training/self-correction round and verified on the held-out exam

    python -m evo.learning.web_learner                 # plan + fetch
    python -m evo.learning.web_learner --plan          # only show topics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evo.corpus.sources import clean_wiki_text
from evo.learning.teacher_corpus import _repetitive, detect_language

CONFIG = Path(__file__).with_name("web_sources.json")
MISTAKES = Path("evo/learning/mistakes.jsonl")
OUT_DIR = Path("data/web_v1")

LANG_BY_OPTION = {"slovenčina": "sk", "čeština": "cs", "poľština": "pl", "angličtina": "en"}
STOPWORDS = {
    "sk": {"ktorý", "ktorá", "ktoré", "alebo", "tiež", "preto", "ktorých", "bola", "bolo", "boli", "jeho", "jej"},
    "cs": {"který", "která", "které", "nebo", "také", "proto", "byla", "bylo", "byly", "jeho", "její"},
    "pl": {"który", "która", "które", "oraz", "także", "była", "było", "były", "jego", "jest"},
    "en": {"which", "their", "there", "these", "those", "would", "could", "about", "after", "before"},
}


class NotAllowed(Exception):
    pass


@dataclass
class Topic:
    lang: str
    query: str | None      # None = random article in that language
    reason: str            # mistake key or "weak-language"


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def allowed_hosts(cfg: dict) -> set[str]:
    return {h for s in cfg["sources"].values() if s.get("enabled") for h in s["hosts"]}


# ------------------------------------------------------------------ topics

def keywords(text: str, lang: str, n: int = 3) -> list[str]:
    words = re.findall(r"[^\W\d_]{5,}", text)
    stop = STOPWORDS.get(lang, set())
    ranked = Counter(w for w in words if w.lower() not in stop)
    caps = [w for w, _ in ranked.most_common() if w[0].isupper()]
    rest = [w for w, _ in ranked.most_common() if not w[0].isupper()]
    return (caps + rest)[:n]


def plan_topics(mistakes: list[dict[str, Any]], max_topics: int = 12) -> list[Topic]:
    topics: list[Topic] = []
    seen: set[tuple[str, str | None]] = set()
    # biggest lesson first: mistakes NOVA was confident about
    for m in sorted(mistakes, key=lambda m: -float(m.get("confidence", 0))):
        if m.get("kind") == "cloze" and m.get("lang") in LANG_BY_OPTION.values():
            text = m["question"] + " " + m["answer"]
            if "roppik" in text or "NOVA" in text:
                continue  # identity is learned from own data, not the web
            kw = [w for w in keywords(text, m["lang"], 5) if w[0].isupper()]
            if kw:
                t = (m["lang"], " ".join(kw[:2]))
                if t not in seen:
                    seen.add(t)
                    topics.append(Topic(m["lang"], t[1], m.get("key", "")))
        elif m.get("kind") == "language":
            lang = LANG_BY_OPTION.get(m.get("answer", ""))
            if lang and (lang, None) not in seen:
                seen.add((lang, None))
                topics.append(Topic(lang, None, "weak-language"))
        if len(topics) >= max_topics:
            break
    return topics


# ---------------------------------------------------------------- fetching

class Fetcher:
    def __init__(self, cfg: dict, opener: Callable | None = None):
        self.cfg = cfg
        self.hosts = allowed_hosts(cfg)
        self.opener = opener or urllib.request.urlopen
        self.last = 0.0
        self.bytes = 0

    def get_raw(self, url: str, headers: dict | None = None) -> bytes:
        host = urllib.parse.urlparse(url).hostname or ""
        if host not in self.hosts:
            raise NotAllowed(f"host not in allowlist: {host}")
        wait = self.cfg.get("delay_seconds", 1.0) - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            url, headers={"User-Agent": self.cfg["user_agent"], **(headers or {})}
        )
        for attempt in range(int(self.cfg.get("max_retries", 3)) + 1):
            try:
                with self.opener(req, timeout=self.cfg.get("timeout_seconds", 20)) as r:
                    raw = r.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt >= int(self.cfg.get("max_retries", 3)):
                    raise
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                pause = float(retry_after) if retry_after and str(retry_after).isdigit() else 10.0 * (attempt + 1)
                time.sleep(min(pause, 120.0))
        self.last = time.time()
        self.bytes += len(raw)
        return raw

    def get_json(self, url: str, headers: dict | None = None) -> dict:
        return json.loads(self.get_raw(url, headers))


# ------------------------------------------------------------- brave search

def brave_key(cfg: dict) -> str | None:
    """API key from env or a private file on the server. Never logged."""
    b = cfg["sources"].get("brave") or {}
    if not b.get("enabled"):
        return None
    key = os.environ.get(b.get("key_env", "BRAVE_API_KEY"))
    if not key and b.get("key_file"):
        path = Path(os.path.expanduser(b["key_file"]))
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
    return key or None


def _usage_path(cfg: dict) -> Path:
    b = cfg["sources"].get("brave") or {}
    return Path(os.path.expanduser(b.get("usage_file", "~/.config/nova/brave_usage.json")))


def brave_month_left(cfg: dict) -> int:
    """Queries still allowed this calendar month (hard cap, survives restarts)."""
    b = cfg["sources"].get("brave") or {}
    cap = int(b.get("max_queries_per_month", 0))
    month = time.strftime("%Y-%m")
    path = _usage_path(cfg)
    used = 0
    if path.exists():
        try:
            used = int(json.loads(path.read_text()).get(month, 0))
        except (ValueError, json.JSONDecodeError):
            used = cap  # unreadable counter -> be safe, spend nothing
    return max(cap - used, 0)


def brave_count_query(cfg: dict) -> None:
    path = _usage_path(cfg)
    month = time.strftime("%Y-%m")
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    data[month] = int(data.get(month, 0)) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def result_domains(cfg: dict) -> set[str]:
    """Search results are usable only from these (non-search) source hosts."""
    return {h for name, s in cfg["sources"].items()
            if name != "brave" and s.get("enabled") for h in s["hosts"]}


def brave_search(f: Fetcher, key: str, query: str, lang: str, count: int = 10) -> list[str]:
    q = urllib.parse.urlencode({"q": query, "count": count, "search_lang": lang,
                                "safesearch": "off"})
    data = f.get_json(f"https://api.search.brave.com/res/v1/web/search?{q}",
                      headers={"X-Subscription-Token": key, "Accept": "application/json"})
    allowed = result_domains(f.cfg)
    urls = []
    for r in (data.get("web") or {}).get("results", []):
        host = urllib.parse.urlparse(r.get("url", "")).hostname or ""
        if host in allowed:
            urls.append(r["url"])
    return urls


class _TextExtractor:
    """Minimal HTML -> text (paragraphs, headings, list items, code)."""

    KEEP = {"p", "h1", "h2", "h3", "h4", "li", "pre", "dd", "dt"}
    SKIP = {"script", "style", "nav", "header", "footer", "aside"}

    def __call__(self, html: str) -> str:
        from html.parser import HTMLParser

        out: list[str] = []
        state = {"keep": 0, "skip": 0, "buf": []}

        class P(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag in _TextExtractor.SKIP:
                    state["skip"] += 1
                if tag in _TextExtractor.KEEP:
                    state["keep"] += 1

            def handle_endtag(self, tag):
                if tag in _TextExtractor.SKIP and state["skip"]:
                    state["skip"] -= 1
                if tag in _TextExtractor.KEEP and state["keep"]:
                    state["keep"] -= 1
                    text = "".join(state["buf"]).strip()
                    if text:
                        out.append(text)
                    state["buf"] = []

            def handle_data(self, data):
                if state["keep"] and not state["skip"]:
                    state["buf"].append(data)

        P().feed(html)
        return "\n\n".join(out)


html_to_text = _TextExtractor()


def fetch_url_article(f: Fetcher, url: str) -> dict[str, str] | None:
    """Wikipedia URL -> clean extract via API; docs URL -> HTML text."""
    u = urllib.parse.urlparse(url)
    host = u.hostname or ""
    if host.endswith("wikipedia.org") and u.path.startswith("/wiki/"):
        lang = host.split(".")[0]
        title = urllib.parse.unquote(u.path[len("/wiki/"):]).replace("_", " ")
        q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                    "titles": title, "format": "json", "redirects": 1})
        pages = f.get_json(f"https://{host}/w/api.php?{q}")["query"]["pages"]
        for p in pages.values():
            if p.get("extract"):
                return {"title": p["title"], "text": p["extract"], "url": url, "lang": lang}
        return None
    text = html_to_text(f.get_raw(url).decode("utf-8", errors="replace"))
    return {"title": u.path.rsplit("/", 1)[-1] or host, "text": text, "url": url, "lang": "en"}


def wikipedia_articles(f: Fetcher, topic: Topic, limit: int = 2) -> list[dict[str, str]]:
    api = f"https://{topic.lang}.wikipedia.org/w/api.php"
    if topic.query:
        q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": topic.query,
                                    "srlimit": limit, "format": "json"})
        titles = [h["title"] for h in f.get_json(f"{api}?{q}")["query"]["search"]]
    else:
        q = urllib.parse.urlencode({"action": "query", "list": "random", "rnnamespace": 0,
                                    "rnlimit": limit, "format": "json"})
        titles = [h["title"] for h in f.get_json(f"{api}?{q}")["query"]["random"]]
    out = []
    for title in titles:
        q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                    "titles": title, "format": "json", "redirects": 1})
        pages = f.get_json(f"{api}?{q}")["query"]["pages"]
        for p in pages.values():
            if p.get("extract"):
                out.append({"title": p["title"], "text": p["extract"],
                            "url": f"https://{topic.lang}.wikipedia.org/wiki/{urllib.parse.quote(p['title'])}"})
    return out


# ----------------------------------------------------------------- quality

def accept(text: str, lang: str) -> tuple[bool, str]:
    if len(text) < 500:
        return False, "too_short"
    if detect_language(text) != lang:
        return False, "wrong_language"
    if _repetitive(text):
        return False, "repetitive"
    return True, "ok"


def _known_keys(out_dir: Path) -> set[str]:
    keys = set()
    for p in out_dir.glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(json.loads(line)["id"])
    return keys


def learn_from_web(
    topics: list[Topic],
    cfg: dict,
    out_dir: Path = OUT_DIR,
    fetcher: Fetcher | None = None,
    log=print,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = fetcher or Fetcher(cfg)
    budget = int(cfg.get("max_mb_per_round", 2.0) * 1_000_000)
    known = _known_keys(out_dir)
    stats: Counter = Counter()
    key = brave_key(cfg)
    brave_left = min(
        int((cfg["sources"].get("brave") or {}).get("max_queries_per_round", 0)),
        brave_month_left(cfg),
    ) if key else 0
    for t in topics:
        if f.bytes >= budget:
            log("byte budget reached")
            break
        try:
            articles = []
            if key and t.query and brave_left > 0:
                brave_left -= 1
                brave_count_query(cfg)
                stats["brave_queries"] += 1
                for url in brave_search(f, key, t.query, t.lang,
                                        cfg["sources"]["brave"].get("results_per_query", 10))[:2]:
                    a = fetch_url_article(f, url)
                    if a:
                        articles.append(a)
            if not articles:
                articles = wikipedia_articles(f, t)
        except NotAllowed as exc:
            stats["not_allowed"] += 1
            log(str(exc))
            continue
        except Exception as exc:
            stats["fetch_error"] += 1
            log(f"[{t.lang}] {t.query}: {type(exc).__name__}: {exc}")
            continue
        for a in articles:
            text = clean_wiki_text(a["text"])
            ok, why = accept(text, t.lang)
            doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
            if ok and doc_id in known:
                ok, why = False, "duplicate"
            stats[why] += 1
            log(f"[{t.lang}] {t.query or 'random'} -> {a['title']}: {why}")
            if not ok:
                continue
            known.add(doc_id)
            host = urllib.parse.urlparse(a["url"]).hostname or ""
            src = "docs" if host in cfg["sources"].get("docs", {}).get("hosts", []) else "wikipedia"
            rec = {"id": doc_id, "status": "EXPERIMENTAL", "lang": t.lang,
                   "source": f"web:{src}", "license": cfg["sources"][src]["license"], "url": a["url"],
                   "title": a["title"], "topic": t.query, "reason": t.reason,
                   "text": text, "fetched_at": time.time()}
            with (out_dir / f"{t.lang}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"stats": dict(stats), "bytes": f.bytes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--max-topics", type=int, default=12)
    args = ap.parse_args(argv)

    mistakes = []
    if MISTAKES.exists():
        mistakes = [json.loads(l) for l in MISTAKES.read_text(encoding="utf-8").splitlines() if l.strip()]
    topics = plan_topics(mistakes, args.max_topics)
    print(f"{len(mistakes)} remembered mistakes -> {len(topics)} topics")
    for t in topics:
        print(f"  [{t.lang}] {t.query or '(random article)'}  <- {t.reason}")
    if args.plan or not topics:
        return 0
    result = learn_from_web(topics, load_config())
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

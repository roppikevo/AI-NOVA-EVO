"""
NOVA browse: an invisible browser that NOVA (or another program) drives through an API. Nothing is shown to anybody -
pages are read, turned into clean text and handed back; on request they are queued as training material.

    python -m nova.browse open https://vrv.sk                     # one page: title, text, links
    python -m nova.browse open https://vrv.sk --follow 1 --pages 10 --train   # and its subpages, queued for training
    python -m nova.browse search "fotosyntéza" --langs sk,en      # Wikipedia search (+ Brave when a key is set up)
    python -m nova.browse serve                                   # the API on 127.0.0.1:8765 (token in ~/.config/nova)

API (every call but /health needs the header "Authorization: Bearer <token>"):
    GET  /health                               alive, pages in the store
    POST /search  {"query", "langs", "count"}  results: title, url, snippet, source
    POST /open    {"url", "train"}             one page now: title, lang, text, links
    POST /crawl   {"url", "depth", "pages", "focus", "train"}   in the background -> {"job": id}
    POST /search_read {"query", "langs", "read", "train"}       search, then read the best results (background job)
    GET  /jobs/<id>                            state and the pages read so far (text shortened)
    GET  /page?url=...                         a page from the store (full text)

Rules (the browser is honest, not hidden):
  * it says who it is (user agent NOVA-EVO), keeps robots.txt and waits between requests to the same host
  * only http/https on ports 80/443 to public addresses: the server's own and private networks are refused, also
    after a redirect (every hop is checked)
  * following links stays on the domain it started from, with a depth and page limit
  * pages behind a login or refused by the site are reported as such and not forced
  * every page keeps its source and date; a page already read is taken from the store for `cache_days`
  * nothing goes straight into the weights: "train" only queues the text (data/browse_v1/train), the director and
    the judge decide as with every other material
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

CONFIG = {
    "user_agent": "NOVA-EVO-browse/1.0 (+https://github.com/roppikevo/NOVA-EVO; research model reading public pages)",
    "delay_seconds": 2.0,          # between two requests to the same host
    "timeout_seconds": 20,
    "max_page_bytes": 3_000_000,
    "max_pages": 30,               # per crawl
    "max_depth": 2,
    "cache_days": 7,
    "min_train_chars": 400,
    "store": "data/browse_v1",
    "token_file": "~/.config/nova/browse.token",
    "brave_key_file": "~/.config/nova/brave.key",
    "brave_usage_file": "~/.config/nova/brave_usage.json",
    "brave_max_per_month": 800,
    "host": "127.0.0.1",
    "port": 8765,
}

TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class Refused(Exception):
    """The browser will not (or may not) read this address; the reason is the message."""


# ------------------------------------------------------------------ safety

def public_address(host: str, resolve: Callable = socket.getaddrinfo) -> bool:
    """True only when every address the name resolves to is a public one."""
    try:
        addrs = {str(ipaddress.ip_address(host.strip("[]")))}                 # an address written out needs no lookup
    except ValueError:
        try:
            addrs = {i[4][0] for i in resolve(host, None)}
        except (socket.gaierror, UnicodeError, OSError):
            return False
    if not addrs:
        return False
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified or (ip.version == 6 and ip.ipv4_mapped and not ip.ipv4_mapped.is_global)
                or not ip.is_global):
            return False
    return True


def check_url(url: str, resolve: Callable = socket.getaddrinfo) -> str:
    """The normalised URL, or Refused with the reason."""
    u = urllib.parse.urlsplit(url.strip())
    if u.scheme not in ("http", "https"):
        raise Refused(f"only http/https ({u.scheme or 'none'})")
    if not u.hostname:
        raise Refused("no host")
    if u.username or u.password:
        raise Refused("addresses with a login in them are not opened")
    port = u.port or (443 if u.scheme == "https" else 80)
    if port not in (80, 443):
        raise Refused(f"port {port} is not opened")
    if not public_address(u.hostname, resolve):
        raise Refused(f"{u.hostname} is not a public address")
    return urllib.parse.urlunsplit((u.scheme, u.netloc.lower(), u.path or "/", u.query, ""))


def add_scheme(url: str) -> str:
    url = url.strip()
    return url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "https://" + url


# ------------------------------------------------------------------ fetching

class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, resolve: Callable):
        self.resolve = resolve

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(urllib.parse.urljoin(req.full_url, newurl), self.resolve)   # every hop must be public too
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_fetch(url: str, cfg: dict, resolve: Callable = socket.getaddrinfo) -> dict:
    """GET with the checks; {status, url (final), type, body} - an HTTP error is a status, not an exception."""
    opener = urllib.request.build_opener(_CheckedRedirect(resolve))
    req = urllib.request.Request(url, headers={"User-Agent": cfg["user_agent"],
                                               "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
                                               "Accept-Language": "sk,cs;q=0.9,en;q=0.8,pl;q=0.7"})
    try:
        with opener.open(req, timeout=cfg["timeout_seconds"]) as r:
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            charset = r.headers.get_content_charset() or "utf-8"
            readable = ctype in TEXT_TYPES or ctype == "application/json" or not ctype
            body = r.read(cfg["max_page_bytes"] + 1) if readable else b""
            return {"status": r.status, "url": r.geturl(), "type": ctype, "charset": charset,
                    "body": body[:cfg["max_page_bytes"]], "truncated": len(body) > cfg["max_page_bytes"]}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "url": url, "type": "", "charset": "utf-8", "body": b"", "truncated": False}


# ------------------------------------------------------------------ reading HTML

class _Page(HTMLParser):
    BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "dd", "dt", "blockquote", "figcaption", "td",
             "th", "article", "section", "div", "br", "tr", "summary", "caption", "label", "main"}
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "template", "iframe",
            "button", "select", "option"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.lang, self.description = "", "", ""
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.images: list[str] = []
        self._skip = 0
        self._in_title = False
        self._a: list[Any] | None = None
        self._buf: list[str] = []

    def _flush(self):
        t = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if t:
            self.parts.append(t)
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html" and a.get("lang"):
            self.lang = a["lang"].split("-")[0].lower()
        if tag == "meta" and (a.get("name") or "").lower() == "description":
            self.description = (a.get("content") or "").strip()
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1
        if tag in self.BLOCK:
            self._flush()
        if tag == "a" and a.get("href"):
            self._a = [a["href"], []]
        if tag == "img" and a.get("alt") and not self._skip:
            self.images.append(a["alt"].strip())

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.BLOCK:
            self._flush()
        if tag == "a" and self._a is not None:
            self.links.append((self._a[0], re.sub(r"\s+", " ", "".join(self._a[1])).strip()))
            self._a = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._a is not None:
            self._a[1].append(data)
        if not self._skip:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def read_html(html: str, base_url: str) -> dict:
    p = _Page()
    p.feed(html)
    p.close()
    seen, parts = set(), []
    for t in p.parts:                                   # drop repeated boilerplate lines and lone crumbs
        key = t.lower()
        if key in seen or (len(t) < 25 and not re.search(r"[.!?:]$", t)):
            continue
        seen.add(key)
        parts.append(t)
    links, got = [], set()
    for href, text in p.links:
        u = urllib.parse.urldefrag(urllib.parse.urljoin(base_url, href.strip()))[0]
        if u.startswith(("http://", "https://")) and u not in got:
            got.add(u)
            links.append({"url": u, "text": text[:120]})
    return {"title": re.sub(r"\s+", " ", p.title).strip(), "html_lang": p.lang, "description": p.description,
            "text": "\n\n".join(parts), "links": links, "images": p.images[:50]}


def detect_lang(text: str, hint: str = "") -> str:
    try:
        from evo.learning.teacher_corpus import detect_language
        got = detect_language(text)
    except Exception:  # noqa: BLE001 - the browser works without the training package too
        got = None
    return got or hint or "?"


def same_site(a: str, b: str) -> bool:
    ha, hb = (urllib.parse.urlsplit(x).hostname or "" for x in (a, b))
    ha, hb = ha.removeprefix("www."), hb.removeprefix("www.")
    return ha == hb or ha.endswith("." + hb) or hb.endswith("." + ha)


# ------------------------------------------------------------------ the browser

class Browser:
    def __init__(self, cfg: dict | None = None, root: Path | str = ".", fetch: Callable | None = None,
                 resolve: Callable = socket.getaddrinfo, clock: Callable = time.time, sleep: Callable = time.sleep):
        self.cfg = {**CONFIG, **(cfg or {})}
        self.root = Path(root)
        self.store = self.root / self.cfg["store"]
        self.resolve = resolve
        self.fetch = fetch or (lambda url: http_fetch(url, self.cfg, resolve))
        self.clock, self.sleep = clock, sleep
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()
        self._index: dict[str, dict] | None = None

    # -- politeness
    def _wait(self, host: str):
        with self._lock:
            wait = self.cfg["delay_seconds"] - (self.clock() - self._last.get(host, -1e9))
            self._last[host] = self.clock() + max(wait, 0)
        if wait > 0:
            self.sleep(wait)

    def allowed(self, url: str) -> bool:
        u = urllib.parse.urlsplit(url)
        key = f"{u.scheme}://{u.netloc}"
        if key not in self._robots:
            rp = None
            try:
                self._wait(u.hostname or "")
                r = self.fetch(key + "/robots.txt")
                if r["status"] == 200:
                    rp = urllib.robotparser.RobotFileParser()
                    rp.parse(r["body"].decode(r.get("charset") or "utf-8", errors="replace").splitlines())
                elif r["status"] >= 500:                          # RFC 9309: server error = assume all is disallowed
                    rp = urllib.robotparser.RobotFileParser()
                    rp.disallow_all = True                            # (4xx = no rules: reading is allowed)
            except (Refused, OSError, ValueError):
                rp = None                                      # no robots.txt: reading is allowed
            self._robots[key] = rp
        rp = self._robots[key]
        return rp is None or rp.can_fetch(self.cfg["user_agent"], url)

    # -- the store
    def _load_index(self) -> dict[str, dict]:
        if self._index is None:
            self._index = {}
            f = self.store / "pages.jsonl"
            if f.exists():
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self._index[row["url"]] = row
        return self._index

    def cached(self, url: str) -> dict | None:
        row = self._load_index().get(url)
        if row and self.clock() - row["fetched"] < self.cfg["cache_days"] * 86400:
            return row
        return None

    def _save(self, page: dict):
        with self._lock:
            self.store.mkdir(parents=True, exist_ok=True)
            with (self.store / "pages.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
            self._load_index()[page["url"]] = page
            if page["final_url"] != page["url"]:
                self._index[page["final_url"]] = page

    def queue_for_training(self, page: dict) -> str:
        """Clean text into data/browse_v1/train/<lang>.jsonl (the format the corpus readers take) - not the weights."""
        text = page.get("text") or ""
        if len(text) < self.cfg["min_train_chars"]:
            return "too_short"
        try:
            from evo.learning.teacher_corpus import _repetitive
            if _repetitive(text):
                return "repetitive"
        except Exception:  # noqa: BLE001
            pass
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        d = self.store / "train"
        with self._lock:
            d.mkdir(parents=True, exist_ok=True)
            seen = d / "seen.txt"
            known = set(seen.read_text().split()) if seen.exists() else set()
            if key in known:
                return "duplicate"
            lang = page["lang"] if page["lang"] in ("sk", "cs", "pl", "en") else "other"
            with (d / f"{lang}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": key, "lang": lang, "text": text, "url": page["final_url"],
                                    "title": page["title"], "fetched": page["date"], "source": "browse",
                                    "license": "unknown (public page; check before release)",
                                    "status": "EXPERIMENTAL"}, ensure_ascii=False) + "\n")
            with seen.open("a") as f:
                f.write(key + "\n")
        return "queued"

    # -- reading
    def open(self, url: str, train: bool = False, use_cache: bool = True) -> dict:
        """One page -> {url, final_url, status, title, lang, text, links, images, date} or {error}."""
        try:
            url = check_url(add_scheme(url), self.resolve)
        except Refused as e:
            return {"url": url, "error": str(e)}
        page = self.cached(url) if use_cache else None
        if page is None:
            if not self.allowed(url):
                return {"url": url, "error": "the site's robots.txt does not allow reading this page"}
            self._wait(urllib.parse.urlsplit(url).hostname or "")
            try:
                r = self.fetch(url)
            except Refused as e:
                return {"url": url, "error": f"redirected: {e}"}
            except (OSError, ValueError) as e:
                return {"url": url, "error": f"could not load: {type(e).__name__}: {str(e)[:120]}"}
            if r["status"] in (401, 403):
                return {"url": url, "error": f"the site refused (HTTP {r['status']}) - login or blocked, not forced"}
            if r["status"] >= 400:
                return {"url": url, "error": f"HTTP {r['status']}"}
            if r["type"] and r["type"] not in TEXT_TYPES:
                return {"url": url, "error": f"not a text page ({r['type']})"}
            body = r["body"].decode(r.get("charset") or "utf-8", errors="replace")
            if r["type"] == "text/plain":
                got = {"title": url.rsplit("/", 1)[-1], "html_lang": "", "description": "", "text": body.strip(),
                       "links": [], "images": []}
            else:
                got = read_html(body, r["url"])
            page = {"url": url, "final_url": r["url"], "status": r["status"], "title": got["title"],
                    "description": got["description"], "lang": detect_lang(got["text"], got["html_lang"]),
                    "text": got["text"], "links": got["links"], "images": got["images"],
                    "truncated": r.get("truncated", False), "fetched": self.clock(),
                    "date": time.strftime("%Y-%m-%d %H:%M", time.gmtime(self.clock()))}
            if len(got["text"]) < 200 and "<script" in body.lower():
                page["note"] = "little text - the page is probably built by JavaScript (not read by this browser yet)"
            self._save(page)
        page = dict(page)
        if train:
            page["training"] = self.queue_for_training(page)
        return page

    def crawl(self, url: str, depth: int = 1, pages: int = 10, focus: str = "", train: bool = False,
              on_page: Callable | None = None, stop: Callable = lambda: False) -> list[dict]:
        """Breadth first from `url`, only on its site, up to `depth` links away and `pages` pages; with `focus`
        the links whose text or address mention its words go first."""
        depth = max(0, min(int(depth), self.cfg["max_depth"]))
        pages = max(1, min(int(pages), self.cfg["max_pages"]))
        words = [w for w in re.findall(r"\w{3,}", focus.lower())]
        start = add_scheme(url)
        queue, seen, out = [(start, 0)], {start}, []
        while queue and len(out) < pages and not stop():
            u, d = queue.pop(0)
            p = self.open(u, train=train)
            final = p.get("final_url")
            if final and final != u:
                if final in seen and any(q.get("final_url") == final for q in out):
                    continue                                   # a redirect to a page already read
                seen.add(final)
            out.append(p)
            if on_page:
                on_page(p)
            if p.get("error") or d >= depth:
                continue
            nxt = []
            for link in p["links"]:
                v = link["url"]
                if v in seen or not same_site(v, start) or re.search(r"\.(jpg|jpeg|png|gif|svg|webp|pdf|zip|mp4|mp3|"
                                                                      r"css|js|ico|xml)(\?|$)", v.lower()):
                    continue
                seen.add(v)
                score = sum(w in (link["text"] + " " + v).lower() for w in words)
                nxt.append((-score, len(nxt), v))
            queue += [(v, d + 1) for _, _, v in sorted(nxt)]
        return out

    # -- searching
    def search(self, query: str, langs: tuple[str, ...] = ("sk", "en"), count: int = 8) -> dict:
        results, notes = [], []
        key = self._brave_key()
        if key and self._brave_left() > 0:
            try:
                results += self._brave(query, langs[0] if langs else "sk", count, key)
            except Exception as e:  # noqa: BLE001
                notes.append(f"brave: {type(e).__name__}")
        elif not key:
            notes.append("web search off: no Brave key on the server (~/.config/nova/brave.key) - Wikipedia only")
        else:
            notes.append("web search off: this month's Brave budget is used up - Wikipedia only")
        for lang in langs:
            try:
                results += self._wikipedia(query, lang, max(2, count // max(len(langs), 1)))
            except Exception as e:  # noqa: BLE001
                notes.append(f"wikipedia {lang}: {type(e).__name__}")
        uniq, got = [], set()
        for r in results:
            if r["url"] not in got:
                got.add(r["url"])
                uniq.append(r)
        return {"query": query, "results": uniq, "notes": notes}

    def _get_json(self, url: str, headers: dict | None = None) -> dict:
        self._wait(urllib.parse.urlsplit(url).hostname or "")
        if headers:                                            # the API key goes only to that API
            req = urllib.request.Request(url, headers={"User-Agent": self.cfg["user_agent"], **headers})
            with urllib.request.urlopen(req, timeout=self.cfg["timeout_seconds"]) as r:
                return json.loads(r.read())
        r = self.fetch(url)
        if r["status"] != 200:
            raise OSError(f"HTTP {r['status']}")
        return json.loads(r["body"])

    def _wikipedia(self, query: str, lang: str, n: int) -> list[dict]:
        q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": n,
                                    "format": "json"})
        hits = self._get_json(f"https://{lang}.wikipedia.org/w/api.php?{q}")["query"]["search"]
        return [{"title": h["title"], "url": f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(h['title'].replace(' ', '_'))}",
                 "snippet": re.sub(r"<[^>]+>", "", h.get("snippet", "")), "source": f"wikipedia-{lang}"} for h in hits]

    def _brave_key(self) -> str | None:
        k = os.environ.get("BRAVE_API_KEY")
        p = Path(os.path.expanduser(self.cfg["brave_key_file"]))
        if not k and p.exists():
            k = p.read_text(encoding="utf-8").strip()
        return k or None

    def _brave_left(self) -> int:
        p = Path(os.path.expanduser(self.cfg["brave_usage_file"]))
        month = time.strftime("%Y-%m")
        try:
            used = int(json.loads(p.read_text()).get(month, 0)) if p.exists() else 0
        except (ValueError, json.JSONDecodeError):
            return 0
        return max(self.cfg["brave_max_per_month"] - used, 0)

    def _brave(self, query: str, lang: str, count: int, key: str) -> list[dict]:
        p = Path(os.path.expanduser(self.cfg["brave_usage_file"]))
        month = time.strftime("%Y-%m")
        data = json.loads(p.read_text()) if p.exists() else {}
        data[month] = int(data.get(month, 0)) + 1                 # counted before the call: never over the budget
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
        q = urllib.parse.urlencode({"q": query, "count": count, "search_lang": lang})
        res = self._get_json(f"https://api.search.brave.com/res/v1/web/search?{q}",
                             {"X-Subscription-Token": key, "Accept": "application/json"})
        return [{"title": r.get("title", ""), "url": r["url"], "snippet": re.sub(r"<[^>]+>", "", r.get("description", "")),
                 "source": "brave"} for r in (res.get("web") or {}).get("results", []) if r.get("url")]

    def search_read(self, query: str, langs=("sk", "en"), read: int = 3, train: bool = False,
                    on_page: Callable | None = None, stop: Callable = lambda: False) -> list[dict]:
        found = self.search(query, tuple(langs), count=max(read * 2, 6))
        out = []
        for r in found["results"]:
            if len(out) >= read or stop():
                break
            p = self.open(r["url"], train=train)
            p["found_by"] = r["source"]
            out.append(p)
            if on_page:
                on_page(p)
        return out


# ------------------------------------------------------------------ the API

def short(page: dict, chars: int = 600) -> dict:
    p = {k: v for k, v in page.items() if k not in ("links", "images", "text")}
    p["text"] = (page.get("text") or "")[:chars]
    p["chars"] = len(page.get("text") or "")
    p["links"] = len(page.get("links") or [])
    return p


def load_token(cfg: dict) -> str:
    p = Path(os.path.expanduser(cfg["token_file"]))
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_urlsafe(32) + "\n")
        os.chmod(p, 0o600)
    return p.read_text().strip()


class Jobs:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, kind: str, fn: Callable) -> str:
        jid = secrets.token_hex(4)
        job = {"id": jid, "kind": kind, "state": "running", "pages": [], "started": time.time()}
        with self.lock:
            self.jobs[jid] = job

        def run():
            try:
                fn(lambda p: job["pages"].append(p))
                job["state"] = "done"
            except Exception as e:  # noqa: BLE001
                job["state"], job["error"] = "failed", f"{type(e).__name__}: {e}"
            job["finished"] = time.time()

        threading.Thread(target=run, daemon=True).start()
        return jid


def make_handler(browser: Browser, token: str, jobs: Jobs):
    class H(BaseHTTPRequestHandler):
        server_version = "nova-browse/1.0"

        def log_message(self, fmt, *args):            # no URLs with queries in the server log
            pass

        def _send(self, code: int, obj: Any):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            got = self.headers.get("Authorization", "")
            if secrets.compare_digest(got, f"Bearer {token}"):
                return True
            self._send(401, {"error": "missing or wrong token"})
            return False

        def _body(self) -> dict:
            n = min(int(self.headers.get("Content-Length") or 0), 100_000)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urllib.parse.urlsplit(self.path)
            if u.path == "/health":
                return self._send(200, {"ok": True, "pages_stored": len(browser._load_index())})
            if not self._authorised():
                return
            if u.path.startswith("/jobs/"):
                job = jobs.jobs.get(u.path.split("/")[-1])
                if not job:
                    return self._send(404, {"error": "no such job"})
                return self._send(200, {**{k: v for k, v in job.items() if k != "pages"},
                                        "pages": [short(p) for p in job["pages"]]})
            if u.path == "/page":
                url = (urllib.parse.parse_qs(u.query).get("url") or [""])[0]
                row = browser._load_index().get(url) or browser._load_index().get(add_scheme(url))
                return self._send(200 if row else 404, row or {"error": "not in the store"})
            self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if not self._authorised():
                return
            try:
                b = self._body()
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"error": "body must be JSON"})
            p = urllib.parse.urlsplit(self.path).path
            langs = tuple(b.get("langs") or ("sk", "en"))
            if p == "/search":
                return self._send(200, browser.search(str(b.get("query", "")), langs, int(b.get("count", 8))))
            if p == "/open":
                page = browser.open(str(b.get("url", "")), train=bool(b.get("train")))
                return self._send(200 if "error" not in page else 422, page)
            if p == "/crawl":
                jid = jobs.start("crawl", lambda on: browser.crawl(str(b.get("url", "")), b.get("depth", 1),
                                                                  b.get("pages", 10), str(b.get("focus", "")),
                                                                  bool(b.get("train")), on_page=on))
                return self._send(202, {"job": jid})
            if p == "/search_read":
                jid = jobs.start("search_read", lambda on: browser.search_read(str(b.get("query", "")), langs,
                                                                              int(b.get("read", 3)), bool(b.get("train")),
                                                                              on_page=on))
                return self._send(202, {"job": jid})
            self._send(404, {"error": "unknown path"})

    return H


def serve(browser: Browser, host: str, port: int, token: str) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1") and not host.startswith("100."):
        raise SystemExit("the API listens only on this machine (127.0.0.1) or the Tailscale address (100.x)")
    return ThreadingHTTPServer((host, port), make_handler(browser, token, Jobs()))


# ------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("open")
    o.add_argument("url")
    o.add_argument("--follow", type=int, default=0, help="links deep on the same site")
    o.add_argument("--pages", type=int, default=10)
    o.add_argument("--focus", default="")
    o.add_argument("--train", action="store_true", help="queue the text for training (data/browse_v1/train)")
    o.add_argument("--full", action="store_true", help="print the whole text")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--langs", default="sk,en")
    s.add_argument("--read", type=int, default=0, help="also read this many of the results")
    s.add_argument("--train", action="store_true")
    s.add_argument("--full", action="store_true")
    v = sub.add_parser("serve")
    v.add_argument("--host", default=CONFIG["host"])
    v.add_argument("--port", type=int, default=CONFIG["port"])
    a = ap.parse_args(argv)
    br = Browser()

    def show(p: dict):
        if p.get("error"):
            print(f"✗ {p['url']}: {p['error']}")
            return
        print(f"== {p['title'] or p['final_url']}  [{p['lang']}, {len(p['text'])} znakov, {len(p['links'])} odkazov"
              + (f", tréning: {p['training']}" if p.get("training") else "") + "]")
        print(f"   {p['final_url']}" + (f"\n   ! {p['note']}" if p.get("note") else ""))
        print(p["text"] if getattr(a, "full", False) else p["text"][:800] + ("…" if len(p["text"]) > 800 else ""))
        print()

    if a.cmd == "open":
        if a.follow:
            for p in br.crawl(a.url, a.follow, a.pages, a.focus, a.train):
                show(p)
        else:
            show(br.open(a.url, train=a.train))
    elif a.cmd == "search":
        langs = tuple(x for x in a.langs.split(",") if x)
        if a.read:
            for p in br.search_read(a.query, langs, a.read, a.train):
                show(p)
        else:
            r = br.search(a.query, langs)
            for n in r["notes"]:
                print("!", n)
            for x in r["results"]:
                print(f"- [{x['source']}] {x['title']}\n  {x['url']}\n  {x['snippet'][:160]}")
    else:
        token = load_token(br.cfg)
        srv = serve(br, a.host, a.port, token)
        print(f"nova-browse API on http://{a.host}:{a.port} (token in {br.cfg['token_file']})", flush=True)
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

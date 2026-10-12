"""nova.browse: the invisible browser - safe addresses, robots.txt, clean text, staying on the site, the store, the API."""

import json
import socket
import threading
import urllib.request

import pytest

from nova import browse as B

PUBLIC = lambda host, port: [(socket.AF_INET, 0, 0, "", ("93.184.216.34", 0))]          # noqa: E731


def resolver(table):
    def r(host, port):
        if host not in table:
            raise socket.gaierror("unknown")
        return [(socket.AF_INET, 0, 0, "", (table[host], 0))]
    return r


PAGE = """<html lang="sk"><head><title>Vrv – úvod</title><meta name="description" content="Stránka o vrvení">
<script>var x = 1;</script><style>p{}</style></head><body>
<nav><a href="/menu">Menu</a></nav>
<h1>Vitajte na stránke o vrvení</h1>
<p>Toto je prvý odsek, ktorý hovorí o tom, čo sa na stránke nachádza a prečo je to dôležité.</p>
<div>Druhý blok textu je v elemente div, ako to robí veľa moderných stránok na webe dnes.</div>
<p>Viac v <a href="/clanky/fotosynteza#hore">článku o fotosyntéze</a> alebo na <a href="https://ina.sk/x">inej stránke</a>.</p>
<img src="a.png" alt="Graf rastu rastlín">
<footer>© 2026 všetky práva vyhradené a tak ďalej a tak ďalej</footer></body></html>"""


def site(pages, robots="", calls=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        if url.endswith("/robots.txt"):
            return {"status": 200 if robots else 404, "url": url, "type": "text/plain", "charset": "utf-8",
                    "body": robots.encode()}
        if url in pages:
            return {"status": 200, "url": url, "type": "text/html", "charset": "utf-8", "body": pages[url].encode()}
        return {"status": 404, "url": url, "type": "", "charset": "utf-8", "body": b""}
    return fetch


def browser(tmp_path, pages, robots="", calls=None, resolve=PUBLIC):
    return B.Browser(root=tmp_path, fetch=site(pages, robots, calls), resolve=resolve, sleep=lambda s: None)


def test_only_public_web_addresses_are_opened():
    pub = resolver({"vrv.sk": "93.184.216.34", "router": "192.168.1.1", "me": "127.0.0.1", "meta": "169.254.169.254",
                    "v6": "::1", "ts": "198.18.0.1"})
    assert B.check_url("https://vrv.sk", pub) == "https://vrv.sk/"
    for bad, why in [("http://router/", "public"), ("http://me/admin", "public"), ("http://meta/latest", "public"),
                     ("http://v6/", "public"), ("http://ts/", "public"), ("ftp://vrv.sk/", "http"),
                     ("file:///etc/passwd", "http"), ("https://vrv.sk:8080/", "port"), ("https://a:b@vrv.sk/", "login"),
                     ("https://nikde.sk/", "public")]:
        with pytest.raises(B.Refused, match=why):
            B.check_url(bad, pub)
    assert B.add_scheme("vrv.sk") == "https://vrv.sk" and B.add_scheme("http://x.sk") == "http://x.sk"


def test_a_redirect_to_the_servers_own_network_is_refused():
    h = B._CheckedRedirect(resolver({"vrv.sk": "93.184.216.34", "in": "10.0.0.5"}))
    req = urllib.request.Request("https://vrv.sk/a")
    with pytest.raises(B.Refused):
        h.redirect_request(req, None, 302, "Found", {}, "http://in/secret")
    assert h.redirect_request(req, None, 302, "Found", {}, "https://vrv.sk/b").full_url == "https://vrv.sk/b"


def test_a_page_becomes_clean_text_with_its_links_and_picture_descriptions():
    got = B.read_html(PAGE, "https://vrv.sk/")
    assert got["title"] == "Vrv – úvod" and got["html_lang"] == "sk" and got["description"] == "Stránka o vrvení"
    assert "prvý odsek" in got["text"] and "elemente div" in got["text"]
    assert "var x" not in got["text"] and "Menu" not in got["text"] and "práva vyhradené" not in got["text"]
    urls = [l["url"] for l in got["links"]]
    assert "https://vrv.sk/clanky/fotosynteza" in urls and "https://ina.sk/x" in urls          # fragment dropped
    assert got["images"] == ["Graf rastu rastlín"]


def test_open_reads_a_page_keeps_it_and_does_not_load_it_twice(tmp_path):
    calls = []
    br = browser(tmp_path, {"https://vrv.sk/": PAGE}, calls=calls)
    p = br.open("vrv.sk")
    assert p["title"] == "Vrv – úvod" and p["status"] == 200 and "prvý odsek" in p["text"] and p["date"]
    n = len(calls)
    assert br.open("https://vrv.sk/")["text"] == p["text"] and len(calls) == n                 # from the store
    again = B.Browser(root=tmp_path, fetch=site({}), resolve=PUBLIC, sleep=lambda s: None)
    assert again.open("https://vrv.sk/")["title"] == "Vrv – úvod"                              # survives a restart
    assert br.open("http://192.168.0.1/")["error"].endswith("not a public address")


def test_robots_txt_and_refusals_are_respected_and_reported(tmp_path):
    br = browser(tmp_path, {"https://vrv.sk/": PAGE, "https://vrv.sk/tajne": PAGE},
                 robots="User-agent: *\nDisallow: /tajne\n")
    assert "error" not in br.open("https://vrv.sk/")
    assert "robots.txt" in br.open("https://vrv.sk/tajne")["error"]

    def forbidden(url):
        return {"status": 403, "url": url, "type": "text/html", "charset": "utf-8", "body": b""}
    br2 = B.Browser(root=tmp_path / "2", fetch=forbidden, resolve=PUBLIC, sleep=lambda s: None)
    assert "not forced" in br2.open("https://zamknute.sk/x")["error"]


def test_following_links_stays_on_the_site_within_the_limits_and_follows_the_focus_first(tmp_path):
    pages = {
        "https://vrv.sk/": '<p>Úvodná stránka s dostatočne dlhým textom na prečítanie.</p>'
                           '<a href="/a">Počasie</a><a href="/b">Fotosyntéza rastlín</a><a href="https://ina.sk/">iná</a>'
                           '<a href="/obr.png">obrázok</a>',
        "https://vrv.sk/a": '<p>Stránka o počasí, ktorá má tiež nejaký dlhší text.</p><a href="/c">ďalej</a>',
        "https://vrv.sk/b": '<p>Stránka o fotosyntéze, ktorá má tiež nejaký dlhší text.</p>',
        "https://vrv.sk/c": '<p>Hlbšia stránka, ktorá sa pri hĺbke 1 nemá prečítať vôbec.</p>',
    }
    calls = []
    br = browser(tmp_path, pages, calls=calls)
    got = br.crawl("https://vrv.sk/", depth=1, pages=10, focus="fotosyntéza")
    assert [p["final_url"] for p in got] == ["https://vrv.sk/", "https://vrv.sk/b", "https://vrv.sk/a"]
    assert not any("ina.sk" in c or "obr.png" in c or c.endswith("/c") for c in calls)
    assert len(browser(tmp_path / "x", pages).crawl("https://vrv.sk/", depth=5, pages=2)) == 2


def test_train_only_queues_good_text_once_and_never_touches_the_weights(tmp_path):
    long_page = "<html lang='sk'><p>" + " ".join(f"Veta {i}: {w} rastie {i % 7 + 2} dní v {m}." for i, (w, m) in enumerate(
        [(w, m) for w in ("mach", "dub", "buk", "tráva", "ruža", "lipa") for m in ("lese", "záhrade", "meste", "horách", "poli")])) + "</p></html>"
    br = browser(tmp_path, {"https://vrv.sk/": long_page, "https://vrv.sk/kratka": "<p>Krátky text tu.</p>"})
    assert br.open("https://vrv.sk/", train=True)["training"] == "queued"
    assert br.open("https://vrv.sk/", train=True)["training"] == "duplicate"
    assert br.open("https://vrv.sk/kratka", train=True)["training"] == "too_short"
    rows = [json.loads(l) for f in (tmp_path / "data/browse_v1/train").glob("*.jsonl") for l in f.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["url"] == "https://vrv.sk/" and rows[0]["status"] == "EXPERIMENTAL"
    assert rows[0]["source"] == "browse" and "license" in rows[0]


def test_search_uses_wikipedia_without_a_key_and_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    wiki = {"query": {"search": [{"title": "Fotosyntéza", "snippet": "<b>Fotosyntéza</b> je proces"}]}}

    def fetch(url):
        assert "wikipedia.org/w/api.php" in url
        return {"status": 200, "url": url, "type": "application/json", "charset": "utf-8", "body": json.dumps(wiki).encode()}
    br = B.Browser({"brave_key_file": str(tmp_path / "none")}, root=tmp_path, fetch=fetch, resolve=PUBLIC, sleep=lambda s: None)
    r = br.search("fotosyntéza", ("sk",))
    assert r["results"][0]["url"] == "https://sk.wikipedia.org/wiki/Fotosynt%C3%A9za"
    assert r["results"][0]["snippet"] == "Fotosyntéza je proces" and "Brave" in r["notes"][0]


def test_the_api_needs_the_token_and_answers_in_json(tmp_path):
    br = browser(tmp_path, {"https://vrv.sk/": PAGE})
    srv = B.serve(br, "127.0.0.1", 0, "tajny-token")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None, token="tajny-token"):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {token}"} if token else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert call("/health", token=None)[0] == 200
        assert call("/open", {"url": "vrv.sk"}, token="zly")[0] == 401
        code, page = call("/open", {"url": "vrv.sk"})
        assert code == 200 and page["title"] == "Vrv – úvod"
        code, job = call("/crawl", {"url": "https://vrv.sk/", "depth": 0})
        assert code == 202
        for _ in range(50):
            st = call(f"/jobs/{job['job']}")[1]
            if st["state"] != "running":
                break
            threading.Event().wait(0.05)
        assert st["state"] == "done" and st["pages"][0]["title"] == "Vrv – úvod"
        assert call("/page?url=https://vrv.sk/")[1]["text"] == page["text"]
    finally:
        srv.shutdown()
    with pytest.raises(SystemExit):
        B.serve(br, "0.0.0.0", 0, "t")                                                       # never open to the world


def test_a_redirected_start_page_is_not_read_twice(tmp_path):
    home = '<p>Domovská stránka s dostatočne dlhým textom na čítanie.</p><a href="/">Domov</a><a href="/o-nas">O nás</a>'

    def fetch(url):
        if url.endswith("/robots.txt"):
            return {"status": 404, "url": url, "type": "", "charset": "utf-8", "body": b""}
        final = url.replace("://www.", "://")
        body = home if final == "https://vrv.sk/" else "<p>Stránka o nás s nejakým dlhším textom.</p>"
        return {"status": 200, "url": final, "type": "text/html", "charset": "utf-8", "body": body.encode()}
    br = B.Browser(root=tmp_path, fetch=fetch, resolve=PUBLIC, sleep=lambda s: None)
    got = br.crawl("https://www.vrv.sk/", depth=1, pages=5)
    assert [p["final_url"] for p in got] == ["https://vrv.sk/", "https://vrv.sk/o-nas"]

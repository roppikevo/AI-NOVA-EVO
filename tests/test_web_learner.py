"""Web learner tests with a fake network (nothing leaves the machine)."""

import io
import json

import pytest

from evo.learning import web_learner as wl

SK_TEXT = ("Bratislava je hlavné mesto Slovenska, ktoré leží na oboch brehoch Dunaja. "
           "Mesto sa rozprestiera aj na úpätí Malých Karpát a je sídlom prezidenta, "
           "parlamentu aj vlády. Bratislava je tiež dôležitým kultúrnym a hospodárskym "
           "centrom krajiny, pretože tu sídli veľa univerzít, divadiel a firiem. ") * 3


def fake_opener(req, timeout=0):
    url = req.full_url
    assert "User-agent" in req.headers
    if "list=search" in url or "list=random" in url:
        key = "search" if "list=search" in url else "random"
        return io.BytesIO(json.dumps({"query": {key: [{"title": "Bratislava"}]}}).encode())
    return io.BytesIO(json.dumps({"query": {"pages": {"1": {"title": "Bratislava",
                                                            "extract": SK_TEXT}}}}).encode())


@pytest.fixture()
def cfg():
    c = wl.load_config()
    c["delay_seconds"] = 0
    c["sources"]["brave"]["enabled"] = True  # code path tests; real config is free-only (disabled)
    return c


def test_allowlist_blocks_other_hosts(cfg):
    f = wl.Fetcher(cfg, opener=fake_opener)
    with pytest.raises(wl.NotAllowed):
        f.get_json("https://evil.example.com/x")
    assert "sk.wikipedia.org" in wl.allowed_hosts(cfg)


def test_topics_from_mistakes_confident_first():
    ms = [
        {"kind": "cloze", "lang": "sk", "question": "Mesto Bratislava leží pri rieke", "answer": "Dunaj",
         "confidence": 0.2, "key": "a"},
        {"kind": "language", "answer": "poľština", "confidence": 0.9, "key": "b"},
        {"kind": "identity", "answer": "roppik", "confidence": 0.9, "key": "c"},
    ]
    topics = wl.plan_topics(ms)
    assert topics[0].lang == "pl" and topics[0].query is None
    assert topics[1].lang == "sk" and "Bratislava" in topics[1].query
    assert len(topics) == 2  # identity mistakes are not looked up on the web


def test_learn_from_web_stores_with_license_and_dedupes(cfg, tmp_path):
    f = wl.Fetcher(cfg, opener=fake_opener)
    topics = [wl.Topic("sk", "Bratislava Dunaj", "a")]
    r1 = wl.learn_from_web(topics, cfg, tmp_path, fetcher=f, log=lambda *_: None)
    r2 = wl.learn_from_web(topics, cfg, tmp_path, fetcher=f, log=lambda *_: None)
    assert r1["stats"]["ok"] == 1 and r2["stats"]["duplicate"] == 1
    rec = json.loads((tmp_path / "sk.jsonl").read_text().splitlines()[0])
    assert rec["license"] == "CC BY-SA 4.0" and rec["url"].startswith("https://sk.wikipedia.org/")


def test_wrong_language_rejected():
    assert wl.accept(SK_TEXT, "pl") == (False, "wrong_language")
    assert wl.accept(SK_TEXT, "sk") == (True, "ok")


def brave_opener(req, timeout=0):
    url = req.full_url
    if "api.search.brave.com" in url:
        assert req.headers.get("X-subscription-token") == "TESTKEY"
        body = {"web": {"results": [
            {"url": "https://spam.example.com/bratislava", "title": "spam"},
            {"url": "https://sk.wikipedia.org/wiki/Bratislava", "title": "Bratislava"},
        ]}}
        return io.BytesIO(json.dumps(body).encode())
    return fake_opener(req, timeout)


def test_brave_used_only_as_pointer_to_allowed_domains(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "TESTKEY")
    cfg["sources"]["brave"]["usage_file"] = str(tmp_path / "usage.json")
    f = wl.Fetcher(cfg, opener=brave_opener)
    urls = wl.brave_search(f, wl.brave_key(cfg), "Bratislava", "sk")
    assert urls == ["https://sk.wikipedia.org/wiki/Bratislava"]
    r = wl.learn_from_web([wl.Topic("sk", "Bratislava", "x")], cfg, tmp_path,
                          fetcher=f, log=lambda *_: None)
    assert r["stats"]["brave_queries"] == 1 and r["stats"]["ok"] == 1


def test_no_key_means_no_brave(cfg, monkeypatch, tmp_path):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    cfg["sources"]["brave"]["key_file"] = str(tmp_path / "missing.key")
    assert wl.brave_key(cfg) is None


def test_html_to_text_keeps_content_drops_scripts():
    html = "<html><script>x=1</script><nav>menu</nav><h1>Title</h1><p>Hello <b>world</b></p><pre>code()</pre></html>"
    assert wl.html_to_text(html) == "Title\n\nHello world\n\ncode()"


def test_monthly_cap_is_enforced(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "TESTKEY")
    cfg["sources"]["brave"]["usage_file"] = str(tmp_path / "usage.json")
    cfg["sources"]["brave"]["max_queries_per_month"] = 2
    assert wl.brave_month_left(cfg) == 2
    wl.brave_count_query(cfg); wl.brave_count_query(cfg)
    assert wl.brave_month_left(cfg) == 0
    f = wl.Fetcher(cfg, opener=brave_opener)
    r = wl.learn_from_web([wl.Topic("sk", "Bratislava", "x")], cfg, tmp_path / "out",
                          fetcher=f, log=lambda *_: None)
    assert "brave_queries" not in r["stats"]  # fell back to Wikipedia search


def test_real_config_is_free_only():
    c = wl.load_config()
    assert c["sources"]["brave"]["enabled"] is False
    assert wl.brave_key(c) is None


def test_429_is_retried_with_backoff(cfg, monkeypatch):
    import urllib.error
    calls = {"n": 0}

    def flaky(req, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {"Retry-After": "0"}, None)
        return fake_opener(req, timeout)

    monkeypatch.setattr(wl.time, "sleep", lambda s: None)
    f = wl.Fetcher(cfg, opener=flaky)
    assert "query" in f.get_json("https://sk.wikipedia.org/w/api.php?action=query&list=random")
    assert calls["n"] == 2


def test_identity_and_lowercase_topics_skipped():
    ms = [{"kind": "cloze", "lang": "pl", "question": "Kto cię stworzył? Stworzył mnie", "answer": "roppik", "confidence": 0.9},
          {"kind": "cloze", "lang": "sk", "question": "sa tam naďalej nachádzajú", "answer": "stromy", "confidence": 0.8}]
    assert wl.plan_topics(ms) == []

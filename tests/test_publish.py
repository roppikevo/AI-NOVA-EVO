"""Publishing: the public commit is a clean snapshot with its own history; problems stop it."""

import json
import subprocess

import pytest

from evo.engine import publish

MAIL = "someone@g" + "mail.com"      # split so that this file itself stays publishable


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "work"
    r.mkdir()
    _git(r, "init", "-q", "-b", "master")
    _git(r, "config", "user.name", "Private Name")
    _git(r, "config", "user.email", MAIL)
    files = {"README.md": "# project\n", "nova/core.py": "x = 1\n", "evo/releases/A-v1/MODEL.json": json.dumps({"frozen": "2026-01-01"}),
             "evo/releases/A-v1/nova_model.pt": "W" * 2000, "evo/old/seed_1.pt": "old checkpoint", "evo/run.err": "trace",
             "evo/engine/STOP_AUTOPILOT": "", "evo/learning/arch-chain.sh": "echo", "fix.patch": "diff"}
    for name, text in files.items():
        (r / name).parent.mkdir(parents=True, exist_ok=True)
        (r / name).write_text(text)
    _git(r, "add", "-A", "-f")
    _git(r, "commit", "-qm", "private work")
    return r


def test_public_commit_is_clean_and_has_no_private_history(repo):
    out = publish.build("first public version", cwd=repo)
    assert out["changed"] and not out["problems"] and out["new_releases"] == ["A-v1"]
    files = set(publish.tracked(repo, publish.PUBLIC_REF))
    assert files == {"README.md", "nova/core.py", "evo/releases/A-v1/MODEL.json", "evo/releases/A-v1/nova_model.pt"}
    log = _git(repo, "log", "--format=%an <%ae> | %cn <%ce> | %s", publish.PUBLIC_REF)
    assert log == f"{publish.AUTHOR[0]} <{publish.AUTHOR[1]}> | {publish.AUTHOR[0]} <{publish.AUTHOR[1]}> | first public version"
    assert MAIL not in _git(repo, "log", "--format=%ae%ce%B", publish.PUBLIC_REF)
    assert _git(repo, "status", "--short") == ""                       # the working folder was not touched
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "master"


def test_next_publication_adds_one_commit_and_nothing_when_unchanged(repo):
    first = publish.build("one", cwd=repo)
    assert publish.build("again", cwd=repo)["changed"] is False
    (repo / "nova" / "core.py").write_text("x = 2\n")
    _git(repo, "commit", "-qam", "private change")
    second = publish.build("two", cwd=repo)
    assert second["changed"] and second["new_releases"] == []
    assert _git(repo, "rev-list", "--count", publish.PUBLIC_REF) == "2"
    assert _git(repo, "rev-parse", f"{publish.PUBLIC_REF}^") == first["commit"]


@pytest.mark.parametrize("name,text", [("notes.md", "write to " + MAIL), ("conf.json", '{"host": "100.' + '92.1.7"}'),
                                       ("a.py", "# Co-authored" + "-by: somebody"), ("k.txt", "-----BEGIN OPENSSH PRIVATE " + "KEY-----")])
def test_problems_stop_the_publication(repo, name, text):
    (repo / name).write_text(text)
    _git(repo, "add", name)
    _git(repo, "commit", "-qm", "oops")
    out = publish.build("x", cwd=repo)
    assert out["problems"] and name in out["problems"][0] and out["commit"] is None
    assert not publish.ref_exists(publish.PUBLIC_REF, repo)


def test_weights_stop_at_the_budget_but_the_release_is_still_listed(repo):
    publish.build("one", cwd=repo)
    for n in ("B-v1", "C-v1"):
        d = repo / "evo" / "releases" / n
        d.mkdir()
        (d / "MODEL.json").write_text("{}")
        (d / "nova_model.pt").write_text(n * 500_000)                    # 2 MB each
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-qm", "two more releases")
    out = publish.build("two", cwd=repo, budget_mb=2.5)
    files = set(publish.tracked(repo, publish.PUBLIC_REF))
    assert out["without_weights"] == ["C-v1"] and "evo/releases/B-v1/nova_model.pt" in files
    assert "evo/releases/C-v1/MODEL.json" in files and "evo/releases/C-v1/nova_model.pt" not in files
    assert "evo/releases/A-v1/nova_model.pt" in files                    # what is public stays public


def test_push_only_to_a_clean_public_repository(repo, tmp_path):
    publish.build("one", cwd=repo)
    assert "no remote" in publish.push(repo)["why"]
    public = tmp_path / "public.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(public))
    _git(repo, "remote", "add", "public", str(public))
    assert publish.push(repo)["pushed"] is True
    assert _git(public, "rev-parse", "refs/heads/main") == _git(repo, "rev-parse", publish.PUBLIC_REF)
    assert publish.push(repo)["why"] == "already up to date"
    working = tmp_path / "working.git"                                    # a repository holding the private history
    _git(tmp_path, "init", "-q", "--bare", "-b", "master", str(working))
    _git(repo, "push", "-q", str(working), "master")
    _git(repo, "remote", "set-url", "public", str(working))
    assert publish.push(repo)["pushed"] is False
    _git(repo, "remote", "add", "origin", str(working))
    assert "working repository" in publish.push(repo)["why"]


def test_index_lists_releases(tmp_path):
    d = tmp_path / "NOVA-X-v1"
    d.mkdir()
    (d / "MODEL.json").write_text(json.dumps({"frozen": "2026-10-03 16:35:57", "scores": {"val": 3.2877, "web_val": 3.3619,
                                                                                       "code": {"solved": 40, "tasks": 79}}}))
    text = publish.index_text(tmp_path)
    assert "| [NOVA-X-v1](NOVA-X-v1/) |" in text and "3.2877" in text and "40 / 79" in text and "not in the repository" in text


def test_auto_publishes_a_new_release_and_only_then(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(repo)
    assert publish.main(["--auto"]) == 0 and not publish.ref_exists(publish.PUBLIC_REF)      # first time is by hand
    public = tmp_path / "public.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(public))
    _git(repo, "remote", "add", "public", str(public))
    assert publish.main(["--push", "-m", "first"]) == 0
    first = _git(public, "rev-parse", "refs/heads/main")
    assert publish.main(["--auto"]) == 0 and _git(public, "rev-parse", "refs/heads/main") == first
    d = repo / "evo" / "releases" / "B-v1"
    d.mkdir()
    (d / "MODEL.json").write_text(json.dumps({"frozen": "2026-02-02", "scores": {"val": 3.1}}))
    (d / "nova_model.pt").write_text("W2" * 100)
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-qm", "director: release B-v1")
    assert publish.main(["--auto"]) == 0
    assert _git(public, "rev-parse", "refs/heads/main") != first
    assert _git(public, "log", "-1", "--format=%s", "main") == "Release B-v1"
    assert "B-v1" in _git(public, "show", "main:evo/releases/README.md")


def test_releases_on_probation_or_stepped_back_do_not_go_public(tmp_path):
    (tmp_path / "evo/director").mkdir(parents=True)
    (tmp_path / "evo/director/state.json").write_text(json.dumps({"probation": {"name": "NOVA8-24M-v5"}, "reverted": ["NOVA8-24M-v3"]}))
    hide = publish.not_standing(tmp_path)
    assert hide == {"probation": {"NOVA8-24M-v5"}, "reverted": {"NOVA8-24M-v3"}}
    assert publish.not_standing(tmp_path / "nothing") == {"probation": set(), "reverted": set()}
    files = {f"evo/releases/{n}/{f}": size for n in ("NOVA8-24M-v2", "NOVA8-24M-v3", "NOVA8-24M-v5")
             for f, size in (("MODEL.json", 500), ("nova_model.pt", 60_000_000), ("core8.py", 9000))}
    files["README.md"] = 100
    public = {"evo/releases/NOVA8-24M-v3/MODEL.json", "evo/releases/NOVA8-24M-v3/nova_model.pt"}      # went out before it was stepped back
    chosen = publish.select(files, public, 100.0, hidden=hide["probation"] | hide["reverted"])
    assert sorted(publish.release_names(set(chosen["keep"]))) == ["NOVA8-24M-v2"] and "README.md" in chosen["keep"]
    assert all("NOVA8-24M-v3" not in p and "NOVA8-24M-v5" not in p for p in chosen["keep"]) and chosen["weights_mb"] == 160.0
    assert len(publish.select(files, public, 100.0)["keep"]) == 10                                   # nothing hidden: everything goes
    root = tmp_path / "evo/releases"
    for n in ("NOVA8-24M-v2", "NOVA8-24M-v3", "NOVA8-24M-v5"):
        (root / n).mkdir(parents=True)
        (root / n / "MODEL.json").write_text(json.dumps({"name": n, "frozen": "2026-10-06 03:00:00", "scores": {"val": 3.0}}))
    text = publish.index_text(root, hide)
    assert "[NOVA8-24M-v2]" in text and "[NOVA8-24M-v3]" not in text and "[NOVA8-24M-v5]" not in text
    assert "stepped back after probation" in text and "NOVA8-24M-v3." in text


def test_a_large_new_release_is_split_into_parts_and_committed(repo, monkeypatch):
    monkeypatch.chdir(repo)
    rel = repo / "evo/releases/NOVA8-BIG-v1"
    rel.mkdir(parents=True)
    (rel / "nova_model.pt").write_bytes(b"x" * 2500)
    assert publish.split_large_weights("NOVA8-BIG-v1", limit=1e9) == []          # small enough: left as it is
    paths = publish.split_large_weights("NOVA8-BIG-v1", limit=1000)
    import nova.parts as parts
    assert len(paths) == len(parts.parts_of(rel / "nova_model.pt")) >= 1
    assert all(p in _git(repo, "ls-files") for p in paths)
    assert publish.split_large_weights("NOVA8-BIG-v1", limit=1000) == []         # already in parts

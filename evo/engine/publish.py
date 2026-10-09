"""
Publish the project: the public repository gets a clean snapshot, the working repository keeps everything.

    python -m evo.engine.publish --index              # write evo/releases/README.md (table of released cores)
    python -m evo.engine.publish --check              # what would be published, what is left out, any problem
    python -m evo.engine.publish --build -m "..."     # make the public commit (local ref refs/public/main)
    python -m evo.engine.publish --push               # build if needed and push it to the remote "public"
    python -m evo.engine.publish --auto               # for cron: a release that is not public yet -> index, build, push

The public commit is built from the tree of HEAD without touching the working folder:
  * left out: checkpoints and other weights (except evo/releases/<name>/nova_model.pt), logs, patches,
    lock/stop files, operational shell chains
  * its only parent is the previous public commit - the working history never goes along
  * author and committer are the Creator with the GitHub no-reply address
  * nothing is built if the public files contain an e-mail address, a private address, a key or a token
  * weights are added while their total in the public history stays under BUDGET_MB; after that a release
    is published without its weights (MODEL.json and checksums stay) and the report says so
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

PUBLIC_REF = "refs/public/main"
REMOTE = "public"
AUTHOR = ("roppik", "roppikevo@users.noreply.github.com")
BUDGET_MB = 900
RELEASE_WEIGHTS = re.compile(r"^evo/releases/([^/]+)/nova_model\.pt(?:\.part\d\d)?$")      # whole, or in parts (nova.parts)
RELEASE_INFO = re.compile(r"^evo/releases/([^/]+)/MODEL\.json$")
DROP = [re.compile(p) for p in (
    r"\.(pt|pth|safetensors|gguf|err|lock|patch|log|tmp|bak)$", r"\.bak-", r"(^|/)STOP[A-Z_]*$",
    r"(^|/)[^/]*chain[^/]*\.sh$", r"^evo/director/.*\.sh$", r"(^|/)skip-[^/]*\.(sh|txt)$")]
# Problems stop the publication. The words are split so that this file does not match itself.
PROBLEMS = {
    "e-mail address": re.compile(r"@(g" r"mail|googlemail|seznam|azet|centrum|zoznam|outlook|hotmail|yahoo|proton|protonmail|icloud)\.[a-z]{2,}"),
    "private network address": re.compile(r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),
    "home folder of a PC": re.compile(r"[A-Za-z]:\\+Users\\+[A-Za-z0-9]"),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE " r"KEY"),
    "access token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|tskey-[A-Za-z0-9-]{10,}"),
}
UNWANTED = re.compile(r"(?i)co-authored" r"-by|\bcl" r"aude\b|anth" r"ropic")      # in code and documents only
UNWANTED_IN = (".py", ".md", ".sh", ".bat", ".txt", ".ini", ".toml")
SCAN_LIMIT = 30_000_000


def git(*args: str, cwd: str | Path = ".", env: dict | None = None, check: bool = True, data: bytes | None = None) -> bytes:
    r = subprocess.run(["git", *args], cwd=cwd, env={**os.environ, **(env or {})}, input=data, capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.decode(errors='replace')[-300:]}")
    return r.stdout


def tracked(cwd: str | Path = ".", ref: str = "HEAD") -> dict[str, int]:
    """Path -> size of every file in the tree of `ref`."""
    out = git("ls-tree", "-r", "-l", "-z", ref, cwd=cwd).decode()
    files = {}
    for row in out.split("\0"):
        if row:
            meta, path = row.split("\t", 1)
            size = meta.split()[3]
            files[path] = int(size) if size.isdigit() else 0
    return files


def ref_exists(ref: str, cwd: str | Path = ".") -> bool:
    return subprocess.run(["git", "rev-parse", "-q", "--verify", ref], cwd=cwd, capture_output=True).returncode == 0


def published_weights_mb(cwd: str | Path = ".") -> float:
    """Megabytes of release weights already in the public history (every version counts, history never shrinks)."""
    if not ref_exists(PUBLIC_REF, cwd):
        return 0.0
    seen: dict[str, str] = {}
    for row in git("rev-list", "--objects", PUBLIC_REF, cwd=cwd).decode().splitlines():
        sha, _, path = row.partition(" ")
        if RELEASE_WEIGHTS.match(path):
            seen[sha] = path
    if not seen:
        return 0.0
    out = git("cat-file", "--batch-check=%(objectsize)", cwd=cwd, data=("\n".join(seen) + "\n").encode()).decode().split()
    return sum(int(x) for x in out) / 1e6


RELEASE_FILE = re.compile(r"^evo/releases/([^/]+)/")
DIRECTOR_STATE = Path("evo/director/state.json")


def not_standing(cwd: str | Path = ".") -> dict[str, set[str]]:
    """Releases that are not (or not yet) the system's word: still on probation, or stepped back after it.

    A release on probation is published once it has stood it; one that was stepped back leaves the public tree."""
    try:
        st = json.loads((Path(cwd) / DIRECTOR_STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"probation": set(), "reverted": set()}
    return {"probation": {st["probation"]["name"]} if st.get("probation") else set(), "reverted": set(st.get("reverted") or [])}


def select(files: dict[str, int], already_public: set[str], used_mb: float, budget_mb: float = BUDGET_MB,
           hidden: set[str] | None = None) -> dict:
    """Which files of the tree go public. Weights of a new release only while the budget lasts; nothing of a
    release in `hidden` (on probation or stepped back)."""
    keep, dropped, no_weights = [], [], []
    weights: dict[str, list[str]] = {}          # a release's weights go together: one file, or all of its parts
    for path in sorted(files):
        r = RELEASE_FILE.match(path)
        if r and hidden and r.group(1) in hidden:
            dropped.append(path)
            continue
        m = RELEASE_WEIGHTS.match(path)
        if m:
            weights.setdefault(m.group(1), []).append(path)
            continue
        (dropped if any(p.search(path) for p in DROP) else keep).append(path)
    for name, paths in sorted(weights.items()):
        new = [x for x in paths if x not in already_public]
        size = sum(files[x] for x in new) / 1e6
        if not new or (all(files[x] < 95e6 for x in new) and used_mb + size <= budget_mb):
            keep.extend(paths)
            used_mb += size
        else:
            keep.extend(x for x in paths if x in already_public)
            no_weights.append(name)
    return {"keep": keep, "dropped": dropped, "without_weights": no_weights, "weights_mb": round(used_mb, 1)}


def scan(paths: list[str], cwd: str | Path = ".", ref: str = "HEAD") -> dict[str, list[str]]:
    """Problems (block the publication) and unwanted mentions in the files that would go public. Names only."""
    problems: list[str] = []
    proc = subprocess.Popen(["git", "cat-file", "--batch"], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdin and proc.stdout
    try:
        for path in paths:
            proc.stdin.write(f"{ref}:{path}\n".encode())
            proc.stdin.flush()
            head = proc.stdout.readline().split()
            if len(head) < 3 or head[1] != b"blob":
                continue
            size = int(head[2])
            body = proc.stdout.read(size)
            proc.stdout.read(1)
            if size > SCAN_LIMIT or b"\0" in body[:8000]:
                continue
            text = body.decode("utf-8", errors="replace")
            for name, pattern in PROBLEMS.items():
                if pattern.search(text):
                    problems.append(f"{name}: {path}")
            if path.endswith(UNWANTED_IN) and UNWANTED.search(text):
                problems.append(f"mention the Creator wants left out: {path}")
    finally:
        proc.stdin.close()
        proc.wait()
    return {"problems": problems}


def release_names(files: dict[str, int] | set[str]) -> set[str]:
    return {m.group(1) for p in files if (m := RELEASE_INFO.match(p))}


def _parameters(config: dict | None) -> str:
    try:
        from evo.engine.architecture_factory import build_model

        return f"{sum(p.numel() for p in build_model(config).parameters()) / 1e6:.1f} M"
    except Exception:
        return "?"


def index_text(root: Path = Path("evo/releases"), hidden: dict[str, set[str]] | None = None) -> str:
    """Table of the released cores (evo/releases/README.md)."""
    hidden = not_standing() if hidden is None else hidden
    rows = []
    for info_file in sorted(root.glob("*/MODEL.json")):
        if info_file.parent.name in hidden["probation"] | hidden["reverted"]:
            continue
        info = json.loads(info_file.read_text(encoding="utf-8"))
        s = info.get("scores") or {}
        code = s.get("code") or {}
        weights = info_file.parent / "nova_model.pt"
        rows.append((info.get("frozen", ""), f"| [{info_file.parent.name}]({info_file.parent.name}/) | {_parameters(info.get('config'))} "
                     f"| {str(info.get('frozen', '?'))[:10]} | {s.get('val', '–')} | {s.get('web_val', '–')} "
                     f"| {code.get('solved', '–')} / {code.get('tasks', '–')} "
                     f"| {f'{weights.stat().st_size / 1e6:.0f} MB' if weights.exists() else 'not in the repository'} |"))
    L = ["# Released cores", "",
         "Every release is frozen: the files never change and `SHA256SUMS` lists their checksums. "
         "Losses are measured on text no training run has seen (lower is better); the code exam has 79 tasks.", "",
         "| Release | Parameters | Frozen | Loss, dataset | Loss, web | Code exam | Weights |", "|---|---|---|---|---|---|---|"]
    L += [r for _, r in sorted(rows)]
    if hidden["reverted"]:
        L += ["", "Accepted by the judge but stepped back after probation (worse on fresh text), not kept here: "
              + ", ".join(sorted(hidden["reverted"])) + "."]
    L += ["", "Use one: `python -m nova.demo --release <name> --lang en --prompt \"The river\"` (see [INSTALL.md](../../INSTALL.md)).", "",
          "`nova_model.pt` holds the weights as 16-bit floats; the full-precision file named in `SHA256SUMS` is too large for a git repository.",
          "", "Licence: free for research, experiments and other non-commercial use; commercial use needs the creator's permission "
          "([LICENSE](../../LICENSE)).", ""]
    return "\n".join(L)


def build(message: str, cwd: str | Path = ".", budget_mb: float = BUDGET_MB) -> dict:
    """Make the public commit from HEAD. Returns what happened; changes only the local ref refs/public/main."""
    files = tracked(cwd)
    old = tracked(cwd, PUBLIC_REF) if ref_exists(PUBLIC_REF, cwd) else {}
    hide = not_standing(cwd)
    chosen = select(files, set(old), published_weights_mb(cwd), budget_mb, hide["probation"] | hide["reverted"])
    found = scan(chosen["keep"], cwd)
    out = {**{k: chosen[k] for k in ("dropped", "without_weights", "weights_mb")}, "files": len(chosen["keep"]),
           "problems": found["problems"], "commit": None, "changed": False,
           "new_releases": sorted(release_names(set(chosen["keep"])) - release_names(set(old)))}
    if found["problems"]:
        return out
    index = Path(git("rev-parse", "--git-dir", cwd=cwd).decode().strip())
    index = (index if index.is_absolute() else Path(cwd) / index) / "public.index"
    env = {"GIT_INDEX_FILE": str(index)}
    if index.exists():
        index.unlink()
    git("read-tree", "HEAD", cwd=cwd, env=env)
    leave = [p for p in files if p not in set(chosen["keep"])]
    if leave:
        git("update-index", "--force-remove", "-z", "--stdin", cwd=cwd, env=env, data=("\0".join(leave) + "\0").encode())
    tree = git("write-tree", cwd=cwd, env=env).decode().strip()
    index.unlink()
    parent = git("rev-parse", PUBLIC_REF, cwd=cwd).decode().strip() if old else None
    if parent and git("rev-parse", f"{parent}^{{tree}}", cwd=cwd).decode().strip() == tree:
        return {**out, "commit": parent}
    who = {"GIT_AUTHOR_NAME": AUTHOR[0], "GIT_AUTHOR_EMAIL": AUTHOR[1], "GIT_COMMITTER_NAME": AUTHOR[0], "GIT_COMMITTER_EMAIL": AUTHOR[1]}
    commit = git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", message, cwd=cwd, env=who).decode().strip()
    git("update-ref", PUBLIC_REF, commit, cwd=cwd)
    return {**out, "commit": commit, "changed": True}


def push(cwd: str | Path = ".", remote: str = REMOTE) -> dict:
    """Push refs/public/main to <remote>/main. Refuses a remote that is not (only) the public history."""
    if not ref_exists(PUBLIC_REF, cwd):
        return {"pushed": False, "why": "nothing built yet"}
    urls = {n: git("remote", "get-url", n, cwd=cwd, check=False).decode().strip() for n in (remote, "origin")}
    if not urls[remote]:
        return {"pushed": False, "why": f"no remote named '{remote}' (git remote add {remote} <address of the public repository>)"}
    if urls[remote] == urls["origin"]:
        return {"pushed": False, "why": "the public remote is the working repository - refusing"}
    env = {"GIT_TERMINAL_PROMPT": "0"}
    listed = subprocess.run(["git", "ls-remote", remote], cwd=cwd, env={**os.environ, **env}, capture_output=True, timeout=120)
    if listed.returncode != 0:
        return {"pushed": False, "why": "the public repository cannot be reached: " + listed.stderr.decode(errors="replace").strip()[-200:]}
    refs = {r.split("\t")[1]: r.split("\t")[0] for r in listed.stdout.decode().splitlines() if "\t" in r}
    foreign = sorted(set(refs) - {"HEAD", "refs/heads/main"})
    if foreign:
        return {"pushed": False, "why": f"the remote holds other branches or tags ({', '.join(foreign[:4])}) - it is not a clean public repository"}
    local = git("rev-parse", PUBLIC_REF, cwd=cwd).decode().strip()
    theirs = refs.get("refs/heads/main")
    if theirs == local:
        return {"pushed": False, "why": "already up to date", "commit": local}
    if theirs and subprocess.run(["git", "merge-base", "--is-ancestor", theirs, local], cwd=cwd, capture_output=True).returncode != 0:
        return {"pushed": False, "why": "the remote 'main' is not part of the local public history - refusing to overwrite it"}
    r = subprocess.run(["git", "push", remote, f"{PUBLIC_REF}:refs/heads/main"], cwd=cwd, env={**os.environ, **env},
                       capture_output=True, timeout=1800)
    if r.returncode != 0:
        return {"pushed": False, "why": r.stderr.decode(errors="replace").strip()[-300:]}
    return {"pushed": True, "commit": local}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--if-new-release", action="store_true")
    ap.add_argument("--auto", action="store_true", help="for cron: when a release is not public yet, refresh the index and push")
    ap.add_argument("-m", "--message", default="")
    args = ap.parse_args(argv)

    if args.auto:
        if not ref_exists(PUBLIC_REF):
            print("nothing is public yet - the first publication is made by hand")
            return 0
        hide = not_standing()
        public = release_names(tracked(ref=PUBLIC_REF))
        new = sorted(release_names(tracked()) - public - hide["probation"] - hide["reverted"])
        gone = sorted(public & hide["reverted"])                  # public, then stepped back: taken out of the tree
        if not new and not gone:
            print("no new release - nothing to publish" + (f" (on probation: {', '.join(sorted(hide['probation']))})" if hide["probation"] else ""))
            return 0
        Path("evo/releases/README.md").write_text(index_text(), encoding="utf-8")
        git("add", "evo/releases/README.md")
        if git("status", "--porcelain", "--", "evo/releases/README.md").strip():
            git("commit", "-qm", f"Index of released cores: {', '.join(new) or 'stepped back ' + ', '.join(gone)}", "--", "evo/releases/README.md")
        args.push, args.message = True, (f"Release {', '.join(new)}" if new else f"Stepped back after probation: {', '.join(gone)}")
    if args.index:
        Path("evo/releases/README.md").write_text(index_text(), encoding="utf-8")
        print("evo/releases/README.md written")
    if args.check:
        files = tracked()
        old = tracked(ref=PUBLIC_REF) if ref_exists(PUBLIC_REF) else {}
        hide = not_standing()
        chosen = select(files, set(old), published_weights_mb(), hidden=hide["probation"] | hide["reverted"])
        found = scan(chosen["keep"])
        print(f"public files: {len(chosen['keep'])}  ({sum(files[p] for p in chosen['keep']) / 1e6:.0f} MB), left out: {len(chosen['dropped'])}")
        print("left out: " + ", ".join(chosen["dropped"][:40]) + (" ..." if len(chosen["dropped"]) > 40 else ""))
        print(f"releases: {', '.join(sorted(release_names(files)))}; weights in public history after this: {chosen['weights_mb']} MB of {BUDGET_MB}")
        if chosen["without_weights"]:
            print("published without weights (budget): " + ", ".join(chosen["without_weights"]))
        for p in found["problems"]:
            print("PROBLEM " + p)
        return 1 if found["problems"] else 0
    if args.build or args.push:
        if args.if_new_release and ref_exists(PUBLIC_REF) and not (release_names(tracked()) - release_names(tracked(ref=PUBLIC_REF))):
            print("no new release - nothing to publish")
            return 0
        names = ", ".join(sorted(release_names(tracked())))
        result = build(args.message or f"NOVA-EVO: cores {names}")
        print(json.dumps({k: v for k, v in result.items() if k != "dropped"}, ensure_ascii=False))
        if result["problems"]:
            for p in result["problems"]:
                print("PROBLEM " + p)
            return 1
        if args.push:
            pushed = push()
            print(json.dumps(pushed, ensure_ascii=False))
            return 0 if pushed["pushed"] or pushed.get("why") == "already up to date" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

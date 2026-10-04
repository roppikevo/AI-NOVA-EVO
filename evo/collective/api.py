"""
HTTP API of a NOVA node (standard library only, no paid service, no extra packages).

    python -m evo.collective.api --id n1 --weights X.pt --tokenizer data/text_v4/tokenizer.json --port 8101

    GET  /info
    POST /score   {"seqs": [[...], ...]}                  -> {"logp": [[...], ...]}
    POST /answer  {"items": [{"context": [...], "options": [[...], ...]}]} -> {"answers": [...]}
    POST /solve   {"tasks": ["key", ...], "samples": 0}   -> {"bodies": {"key": ["...", ...]}}
    POST /next    {"context": [...], "k": 40}             -> {"top": [[id, logp], ...]}

The server listens on 127.0.0.1 by default. If NOVA_COLLECTIVE_KEY is set, every
request must carry the same value in the X-NOVA-Key header (the first step
towards signed nodes; nodes of other people are not accepted yet).
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

KEY_ENV = "NOVA_COLLECTIVE_KEY"


def make_handler(node, key: str = ""):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            return

        def _send(self, code: int, obj: Any) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            if key and self.headers.get("X-NOVA-Key") != key:
                self._send(403, {"error": "wrong or missing X-NOVA-Key"})
                return False
            return True

        def do_GET(self):
            if not self._authorised():
                return
            if self.path == "/info":
                self._send(200, node.info())
            elif hasattr(node, "dispatch"):  # peer protocol of an autonomous node (evo.collective.agent)
                try:
                    self._send(200, node.dispatch(self.path, {}))
                except KeyError:
                    self._send(404, {"error": "unknown path"})
                except Exception as exc:
                    self._send(400, {"error": f"{type(exc).__name__}: {exc}"[:300]})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if not self._authorised():
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                if self.path == "/score":
                    self._send(200, {"logp": node.score(req["seqs"])})
                elif self.path == "/answer":
                    self._send(200, {"answers": node.answer(req["items"])})
                elif self.path == "/solve":
                    self._send(200, {"bodies": node.solve(req["tasks"], int(req.get("samples", 0)))})
                elif self.path == "/next":
                    self._send(200, {"top": node.next(req["context"], int(req.get("k", 40)))})
                elif hasattr(node, "dispatch"):
                    try:
                        self._send(200, node.dispatch(self.path, req))
                    except KeyError:
                        self._send(404, {"error": "unknown path"})
                else:
                    self._send(404, {"error": "unknown path"})
            except Exception as exc:  # a bad request must not kill the node
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"[:300]})

    return Handler


def serve(node, port: int, host: str = "127.0.0.1", key: str | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(node, os.environ.get(KEY_ENV, "") if key is None else key))
    return server


class RemoteNode:
    """Same methods as Node, but over HTTP - a node can be on this server or anywhere else."""

    def __init__(self, url: str, key: str | None = None, timeout: float = 600.0) -> None:
        self.url, self.timeout = url.rstrip("/"), timeout
        self.key = os.environ.get(KEY_ENV, "") if key is None else key
        self._info: dict | None = None

    def _call(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        if self.key:
            req.add_header("X-NOVA-Key", self.key)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def info(self) -> dict:
        if self._info is None:
            self._info = self._call("/info")
        return self._info

    @property
    def node_id(self) -> str:
        return self.info()["node"]

    @property
    def specialty(self) -> str:
        return self.info().get("specialty", "")

    def score(self, seqs: list[list[int]], batch: int = 64) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(seqs), batch):
            out += self._call("/score", {"seqs": seqs[i:i + batch]})["logp"]
        return out

    def answer(self, items: list[dict], batch: int = 50) -> list[int]:
        out: list[int] = []
        for i in range(0, len(items), batch):
            out += self._call("/answer", {"items": items[i:i + batch]})["answers"]
        return out

    def solve(self, task_keys: list[str], samples: int = 0) -> dict[str, list[str]]:
        return self._call("/solve", {"tasks": task_keys, "samples": samples})["bodies"]

    def next(self, context: list[int], k: int = 40) -> list[list[float]]:
        return self._call("/next", {"context": context, "k": k})["top"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--specialty", default="")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args(argv)

    import torch

    from evo.collective.node import load_node

    torch.set_num_threads(args.threads)
    node = load_node(args.id, args.weights, args.tokenizer, args.specialty, args.device)
    server = serve(node, args.port, args.host)
    print(f"node {args.id} ({args.specialty}) listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
